"""Web GUI for continualcode workflows."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinker import types
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .profiles import list_profiles
from .train import ContinualSDPOSession, SDPOConfig, SampledCompletion
from .workflows import (
    DPOTrainConfig,
    ExamplesTrainConfig,
    train_dpo_from_preferences_async,
    train_lora_from_examples,
)


def _extract_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content)


@dataclass
class WebSessionState:
    id: str
    requested_model: str
    resolved_model: str
    mode: str
    enable_training: bool
    system_prompt: str
    max_tokens: int
    temperature: float
    sample_timeout_seconds: float
    train_timeout_seconds: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_completion: SampledCompletion | None = None
    last_text: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SharedRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    shared_session: ContinualSDPOSession | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    checkpoint_path: str | None = None


@dataclass
class JobState:
    id: str
    kind: str
    status: str
    created_at: str
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class SessionStartRequest(BaseModel):
    model_name: str | None = None
    mode: str = "creative"
    profile: str | None = None
    desired_length: str | None = None
    style: str | None = None
    binary_modes: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    enable_training: bool = True
    load_checkpoint_path: str | None = None
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None
    system_prompt: str | None = None
    sample_timeout_seconds: float | None = None
    train_timeout_seconds: float | None = None


class PromptRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    kind: str  # accept | revise | ideal
    text: str | None = None


class TrainExamplesRequest(BaseModel):
    model_name: str | None = None
    examples_path: str
    log_path: str | None = None
    load_checkpoint_path: str | None = None
    learning_rate: float = 1e-4
    num_epochs: int = 1
    lora_rank: int = 32
    batch_size: int = 64
    max_length: int | None = 8192
    save_every: int = 20
    profile: str = "default"
    desired_length: str = "medium"
    style: str | None = None
    binary_modes: str | None = None


class TrainDPORequest(BaseModel):
    model_name: str | None = None
    preferences_path: str
    preferences_test_path: str | None = None
    log_path: str | None = None
    load_checkpoint_path: str | None = None
    learning_rate: float = 1e-5
    num_epochs: int = 1
    lora_rank: int = 32
    batch_size: int = 64
    max_length: int | None = 8192
    dpo_beta: float = 0.1
    save_every: int = 20
    profile: str = "default"
    desired_length: str = "medium"
    style: str | None = None
    binary_modes: str | None = None


def build_app(config: Any) -> FastAPI:
    app = FastAPI(title="continualcode web")
    sessions: dict[str, WebSessionState] = {}
    jobs: dict[str, JobState] = {}
    runtime = SharedRuntime()
    static_dir = Path(__file__).resolve().parent / "web_static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _validate_api_key() -> None:
        api_key = os.environ.get("TINKER_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="TINKER_API_KEY is not set.")
        if not api_key.startswith("tml-"):
            raise HTTPException(
                status_code=400,
                detail="TINKER_API_KEY is invalid. It must start with 'tml-'.",
            )

    def _session_snapshot(state: WebSessionState) -> dict[str, Any]:
        assistant_count = sum(1 for msg in state.messages if msg.get("role") == "assistant")
        return {
            "session_id": state.id,
            "requested_model": state.requested_model,
            "mode": state.mode,
            "resolved_model": state.resolved_model,
            "model_name": state.resolved_model,  # backwards compatibility for older UI clients
            "messages": len(state.messages),
            "assistant_messages": assistant_count,
            "has_last_completion": state.last_completion is not None,
            "created_at": state.created_at,
        }

    async def _ensure_shared_runtime(req: SessionStartRequest) -> ContinualSDPOSession:
        _validate_api_key()
        requested_model = (req.model_name or config.model_name).strip() or config.model_name
        requested_checkpoint = req.load_checkpoint_path or runtime.checkpoint_path or config.load_checkpoint_path
        wants_training = bool(config.enable_training or req.enable_training)

        if runtime.shared_session is None:
            shared_system_prompt = (
                config.system_prompt
                or "You are a text-first assistant. Follow the conversation's instructions."
            )
            shared = ContinualSDPOSession(
                model=requested_model,
                checkpoint=requested_checkpoint,
                teacher_model=req.teacher_model or config.teacher_model,
                teacher_checkpoint=req.teacher_checkpoint or config.teacher_checkpoint,
                tinker_url=config.base_url,
                max_tokens=req.max_tokens or config.max_tokens,
                temperature=req.temperature if req.temperature is not None else config.temperature,
                system_prompt=shared_system_prompt,
                tool_specs=[],
                enable_training=wants_training,
                lora_rank=config.lora_rank,
                learning_rate=config.learning_rate,
                sdpo_config=SDPOConfig(kl_coef=config.kl_coef),
            )
            try:
                await shared.init()
            except Exception as e:
                msg = str(e)
                if "api_key must start with the 'tml-' prefix" in msg:
                    raise HTTPException(
                        status_code=400,
                        detail="TINKER_API_KEY is invalid. It must start with 'tml-'.",
                    ) from e
                raise HTTPException(status_code=500, detail=f"Failed to initialize shared runtime: {msg}") from e

            runtime.shared_session = shared
            runtime.requested_model = requested_model
            runtime.resolved_model = shared.model
            runtime.checkpoint_path = requested_checkpoint
            return shared

        shared = runtime.shared_session
        assert shared is not None
        if requested_model not in {runtime.requested_model, runtime.resolved_model}:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Shared runtime already initialized with a different model. "
                    f"requested={runtime.requested_model}, resolved={runtime.resolved_model}. "
                    "Restart the web server to switch base models."
                ),
            )
        if requested_checkpoint is not None and requested_checkpoint != runtime.checkpoint_path:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Shared runtime already initialized with a different checkpoint. "
                    f"current={runtime.checkpoint_path}. Restart server to switch checkpoint."
                ),
            )
        if wants_training and shared.training_client is None:
            try:
                training_client = await shared.service_client.create_lora_training_client_async(
                    base_model=shared.model, rank=config.lora_rank
                )
                if runtime.checkpoint_path:
                    await training_client.load_state_async(runtime.checkpoint_path)
                shared.training_client = training_client
                shared.sampling_client = await training_client.save_weights_and_get_sampling_client_async("current")
                shared.teacher_sampling_client = shared.sampling_client
                shared.enable_training = True
                shared._teacher_is_student = True
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to enable shared training runtime: {e}",
                ) from e
        return shared

    async def _reload_runtime_from_checkpoint(path: str) -> None:
        runtime.checkpoint_path = path
        shared = runtime.shared_session
        if shared is None:
            return

        service_client = shared.service_client
        if shared.enable_training:
            try:
                training_client = await service_client.create_training_client_from_state_with_optimizer_async(path)
            except Exception:
                training_client = await service_client.create_training_client_from_state_async(path)
            shared.training_client = training_client
            shared.sampling_client = await training_client.save_weights_and_get_sampling_client_async("current")
            shared.teacher_sampling_client = shared.sampling_client
            shared._teacher_is_student = True
        else:
            shared.sampling_client = await service_client.create_sampling_client_async(model_path=path)
            shared.teacher_sampling_client = shared.sampling_client
        shared.checkpoint = path

    async def _sample_once(state: WebSessionState) -> tuple[str, SampledCompletion | None]:
        shared = runtime.shared_session
        if shared is None or shared.sampling_client is None:
            raise HTTPException(status_code=500, detail="Shared runtime is not initialized.")

        session_prompt_messages = (
            [{"role": "system", "content": state.system_prompt}] if state.system_prompt else []
        )
        prompt_messages = list(shared._prefix) + session_prompt_messages + list(state.messages)
        model_input = shared.renderer.build_generation_prompt(prompt_messages)
        prompt_len = model_input.length

        try:
            response = await asyncio.wait_for(
                shared.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        stop=shared.renderer.get_stop_sequences(),
                        max_tokens=state.max_tokens,
                        temperature=state.temperature,
                    ),
                ),
                timeout=state.sample_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise HTTPException(status_code=504, detail="Sampling timed out.") from e

        seq = response.sequences[0]
        message, ok = shared.renderer.parse_response(seq.tokens)
        if not ok:
            raise HTTPException(status_code=500, detail="Renderer parse failed.")
        text = _extract_text(message).strip()
        if not text:
            raise HTTPException(status_code=500, detail="Empty model response.")

        completion: SampledCompletion | None = None
        if shared.training_client is not None and seq.logprobs is not None:
            completion = SampledCompletion(
                prompt_input=model_input,
                prompt_messages=prompt_messages,
                tokens=seq.tokens,
                logprobs=seq.logprobs,
                prompt_len=prompt_len,
            )

        state.messages.append(message)
        state.last_completion = completion
        state.last_text = text
        return text, completion

    async def _spawn_job(kind: str, payload: dict[str, Any], coro: Any) -> str:
        job_id = str(uuid.uuid4())
        jobs[job_id] = JobState(
            id=job_id,
            kind=kind,
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )

        async def runner() -> None:
            try:
                result = await coro
                jobs[job_id].status = "succeeded"
                jobs[job_id].result = result
            except Exception as e:  # pragma: no cover
                jobs[job_id].status = "failed"
                jobs[job_id].error = str(e)

        asyncio.create_task(runner())
        return job_id

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/profiles")
    async def api_profiles() -> dict[str, Any]:
        return {"profiles": list_profiles()}

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": [_session_snapshot(state) for state in sessions.values()]}

    @app.post("/api/sessions")
    async def create_session(req: SessionStartRequest) -> dict[str, Any]:
        mode = req.mode.strip().lower()
        if mode not in {"creative", "coding"}:
            raise HTTPException(status_code=400, detail="mode must be creative or coding")

        default_creative_prompt = (
            "You are a creative writing partner.\n"
            "Produce vivid, original prose and prioritize voice, imagery, pacing, and emotional resonance.\n"
            "When the user gives feedback, revise directly toward that feedback without being defensive.\n"
            "Do not mention internal reasoning."
        )
        default_coding_prompt = (
            "You are a pragmatic coding assistant.\n"
            "Give concise, technically correct answers and actionable code changes.\n"
            "Do not mention internal reasoning."
        )
        user_system_prompt = (req.system_prompt or "").strip()
        if user_system_prompt:
            final_system_prompt = user_system_prompt
        elif mode == "creative":
            final_system_prompt = default_creative_prompt
        else:
            final_system_prompt = default_coding_prompt
        requested_model = (req.model_name or config.model_name).strip() or config.model_name
        async with runtime.lock:
            shared = await _ensure_shared_runtime(req)
        session_id = str(uuid.uuid4())
        sessions[session_id] = WebSessionState(
            id=session_id,
            requested_model=requested_model,
            resolved_model=shared.model,
            mode=mode,
            enable_training=bool(req.enable_training),
            system_prompt=final_system_prompt,
            max_tokens=req.max_tokens or config.max_tokens,
            temperature=req.temperature if req.temperature is not None else config.temperature,
            sample_timeout_seconds=req.sample_timeout_seconds or config.sample_timeout_seconds,
            train_timeout_seconds=req.train_timeout_seconds or config.train_timeout_seconds,
        )
        return _session_snapshot(sessions[session_id])

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        state = sessions.get(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Unknown session")
        return _session_snapshot(state)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        state = sessions.pop(session_id, None)
        if not state:
            raise HTTPException(status_code=404, detail="Unknown session")
        return {"ok": True, "deleted_session_id": session_id}

    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str) -> dict[str, Any]:
        state = sessions.get(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Unknown session")
        state.messages.clear()
        state.last_completion = None
        state.last_text = None
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/prompt")
    async def send_prompt(session_id: str, req: PromptRequest) -> dict[str, Any]:
        state = sessions.get(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Unknown session")
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Prompt text is required.")
        async with runtime.lock:
            if runtime.shared_session is None:
                raise HTTPException(status_code=400, detail="No shared runtime. Create a session first.")
            state.messages.append({"role": "user", "content": text})
            response_text, _ = await _sample_once(state)
            return {"response": response_text}

    @app.post("/api/sessions/{session_id}/feedback")
    async def send_feedback(session_id: str, req: FeedbackRequest) -> dict[str, Any]:
        state = sessions.get(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Unknown session")
        kind = req.kind.strip().lower()
        async with runtime.lock:
            shared = runtime.shared_session
            if shared is None:
                raise HTTPException(status_code=400, detail="No shared runtime. Create a session first.")
            if kind == "accept":
                return {"ok": True, "status": "accepted"}

            if not any(msg.get("role") == "assistant" for msg in state.messages):
                raise HTTPException(
                    status_code=400,
                    detail="No assistant response in this session yet. Send a prompt before feedback.",
                )

            feedback_text = (req.text or "").strip()
            if not feedback_text:
                raise HTTPException(status_code=400, detail="Feedback text is required.")

            solution: str | None = None
            correction = feedback_text
            if kind == "ideal":
                correction = (
                    "Your previous draft missed the target response. Use the provided ideal response as demonstration."
                )
                solution = feedback_text
                state.messages.append(
                    {"role": "user", "content":
                    "Rewrite the previous draft to match this target response quality and style:\n"
                    f"{feedback_text}"}
                )
            elif kind == "revise":
                state.messages.append(
                    {"role": "user", "content":
                    "Revise the previous draft using this feedback:\n"
                    f"{feedback_text}"}
                )
            else:
                raise HTTPException(status_code=400, detail="kind must be accept, revise, or ideal")

            metrics: dict[str, Any] = {}
            if state.enable_training and shared.training_client is not None:
                completion = state.last_completion
                if completion is None:
                    raise HTTPException(
                        status_code=400,
                        detail="No trainable completion available for feedback scoring.",
                    )
                shared.record_denial(completion, correction, solution=solution)
                try:
                    metrics = await asyncio.wait_for(
                        shared.train_sdpo(),
                        timeout=state.train_timeout_seconds,
                    )
                except asyncio.TimeoutError as e:
                    raise HTTPException(status_code=504, detail="Training timed out.") from e

            revised_text, _ = await _sample_once(state)
            return {"ok": True, "response": revised_text, "metrics": metrics}

    @app.post("/api/jobs/sft")
    async def run_examples_job(req: TrainExamplesRequest) -> dict[str, Any]:
        payload = req.model_dump()
        async with runtime.lock:
            shared = await _ensure_shared_runtime(
                SessionStartRequest(
                    model_name=req.model_name,
                    load_checkpoint_path=req.load_checkpoint_path,
                    enable_training=True,
                )
            )
            effective_model = shared.model
            effective_checkpoint = req.load_checkpoint_path or runtime.checkpoint_path

        cfg = ExamplesTrainConfig(
            model_name=effective_model,
            examples_path=req.examples_path,
            log_path=req.log_path or config.training_log_path,
            load_checkpoint_path=effective_checkpoint,
            base_url=config.base_url,
            renderer_name=config.renderer_name,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            lora_rank=req.lora_rank,
            save_every=req.save_every,
            batch_size=req.batch_size,
            max_length=req.max_length,
            profile=req.profile,
            desired_length=req.desired_length,
            style=req.style,
            binary_modes=req.binary_modes,
        )

        async def run_and_apply() -> dict[str, Any]:
            result = await train_lora_from_examples(cfg)
            checkpoint_path = result.get("state_path") or result.get("sampler_path")
            if isinstance(checkpoint_path, str) and checkpoint_path:
                async with runtime.lock:
                    await _reload_runtime_from_checkpoint(checkpoint_path)
                result["applied_checkpoint_path"] = checkpoint_path
            return result

        job_id = await _spawn_job("train_examples", payload, run_and_apply())
        return {"job_id": job_id}

    @app.post("/api/jobs/dpo")
    async def run_dpo_job(req: TrainDPORequest) -> dict[str, Any]:
        payload = req.model_dump()
        async with runtime.lock:
            shared = await _ensure_shared_runtime(
                SessionStartRequest(
                    model_name=req.model_name,
                    load_checkpoint_path=req.load_checkpoint_path,
                    enable_training=True,
                )
            )
            effective_model = shared.model
            effective_checkpoint = req.load_checkpoint_path or runtime.checkpoint_path

        cfg = DPOTrainConfig(
            model_name=effective_model,
            preferences_path=req.preferences_path,
            preferences_test_path=req.preferences_test_path,
            log_path=req.log_path or config.training_log_path,
            load_checkpoint_path=effective_checkpoint,
            base_url=config.base_url,
            renderer_name=config.renderer_name,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            lora_rank=req.lora_rank,
            batch_size=req.batch_size,
            max_length=req.max_length,
            dpo_beta=req.dpo_beta,
            save_every=req.save_every,
            profile=req.profile,
            desired_length=req.desired_length,
            style=req.style,
            binary_modes=req.binary_modes,
        )

        async def run_and_apply() -> dict[str, Any]:
            result = await train_dpo_from_preferences_async(cfg)
            checkpoint_path = result.get("state_path") or result.get("sampler_path")
            if isinstance(checkpoint_path, str) and checkpoint_path:
                async with runtime.lock:
                    await _reload_runtime_from_checkpoint(checkpoint_path)
                result["applied_checkpoint_path"] = checkpoint_path
            return result

        job_id = await _spawn_job("train_dpo", payload, run_and_apply())
        return {"job_id": job_id}

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {
            "jobs": [
                {
                    "id": job.id,
                    "kind": job.kind,
                    "status": job.status,
                    "created_at": job.created_at,
                }
                for job in jobs.values()
            ]
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "created_at": job.created_at,
            "payload": job.payload,
            "result": job.result,
            "error": job.error,
        }

    return app


def run_web_server(config: Any) -> None:
    app = build_app(config)
    uvicorn.run(app, host=config.web_host, port=int(config.web_port), log_level="info")
