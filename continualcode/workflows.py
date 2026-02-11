"""Training workflow helpers for example-SFT and DPO."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

from .profiles import build_control_instruction, parse_binary_modes


@dataclass
class ExamplesTrainConfig:
    model_name: str
    examples_path: str
    log_path: str
    load_checkpoint_path: str | None = None
    base_url: str | None = None
    renderer_name: str | None = None
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    lora_rank: int = 32
    save_every: int = 20
    batch_size: int = 64
    max_length: int | None = 8192
    test_size: int = 0
    shuffle_seed: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    profile: str = "default"
    desired_length: str = "medium"
    style: str | None = None
    binary_modes: str | None = None


@dataclass
class DPOTrainConfig:
    model_name: str
    preferences_path: str
    log_path: str
    preferences_test_path: str | None = None
    load_checkpoint_path: str | None = None
    base_url: str | None = None
    renderer_name: str | None = None
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    lora_rank: int = 32
    dpo_beta: float = 0.1
    reference_model_name: str | None = None
    save_every: int = 20
    batch_size: int = 64
    max_length: int | None = 8192
    wandb_project: str | None = None
    wandb_name: str | None = None
    profile: str = "default"
    desired_length: str = "medium"
    style: str | None = None
    binary_modes: str | None = None


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} in {path} is not a JSON object.")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _control_instruction_for_row(
    row: dict[str, Any],
    *,
    default_profile: str,
    default_length: str,
    default_style: str | None,
    default_binary_modes: str | None,
) -> str:
    row_binary_modes = row.get("binary_modes")
    if isinstance(row_binary_modes, list):
        binary_modes = [str(x) for x in row_binary_modes]
    elif isinstance(row_binary_modes, str):
        binary_modes = parse_binary_modes(row_binary_modes)
    else:
        binary_modes = parse_binary_modes(default_binary_modes)

    return build_control_instruction(
        profile=str(row.get("profile", default_profile)),
        desired_length=str(row.get("desired_length", default_length)),
        style=str(row.get("style", default_style)) if (row.get("style", default_style)) else None,
        binary_modes=binary_modes,
    )


def prepare_examples_jsonl(
    *,
    examples_path: str,
    output_dir: str,
    profile: str,
    desired_length: str,
    style: str | None,
    binary_modes: str | None,
) -> tuple[str, int]:
    rows = _read_jsonl(examples_path)
    prepared: list[dict[str, Any]] = []

    for row in rows:
        # Already in conversation format.
        if isinstance(row.get("messages"), list):
            messages = list(row["messages"])
            control = _control_instruction_for_row(
                row,
                default_profile=profile,
                default_length=desired_length,
                default_style=style,
                default_binary_modes=binary_modes,
            )
            if control:
                messages = [{"role": "system", "content": control}, *messages]
            prepared.append({"messages": messages})
            continue

        prompt = row.get("prompt") or row.get("instruction")
        response = row.get("response") or row.get("output")
        if not prompt or not response:
            raise ValueError(
                "Each examples JSONL row must contain either `messages` or (`prompt`/`instruction` + `response`/`output`)."
            )
        control = _control_instruction_for_row(
            row,
            default_profile=profile,
            default_length=desired_length,
            default_style=style,
            default_binary_modes=binary_modes,
        )
        messages = [
            {"role": "system", "content": control},
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
        prepared.append({"messages": messages})

    prepared_path = os.path.join(output_dir, "prepared_examples.jsonl")
    _write_jsonl(prepared_path, prepared)
    return prepared_path, len(prepared)


def prepare_preferences_jsonl(
    *,
    preferences_path: str,
    output_dir: str,
    profile: str,
    desired_length: str,
    style: str | None,
    binary_modes: str | None,
) -> tuple[str, int]:
    rows = _read_jsonl(preferences_path)
    prepared: list[dict[str, Any]] = []

    for row in rows:
        # Already in cookbook format.
        if isinstance(row.get("comparison"), dict) and row.get("label") in {"A", "B", "Tie"}:
            prepared.append(row)
            continue

        prompt = row.get("prompt") or row.get("instruction")
        chosen = row.get("chosen") or row.get("preferred")
        rejected = row.get("rejected") or row.get("dispreferred")
        if not prompt or not chosen or not rejected:
            raise ValueError(
                "Each preferences JSONL row must contain either (`comparison`,`label`) or (`prompt`,`chosen`,`rejected`)."
            )
        control = _control_instruction_for_row(
            row,
            default_profile=profile,
            default_length=desired_length,
            default_style=style,
            default_binary_modes=binary_modes,
        )
        prompt_conversation = [
            {"role": "system", "content": control},
            {"role": "user", "content": str(prompt)},
        ]
        prepared.append(
            {
                "comparison": {
                    "prompt_conversation": prompt_conversation,
                    "completion_A": [{"role": "assistant", "content": str(chosen)}],
                    "completion_B": [{"role": "assistant", "content": str(rejected)}],
                },
                "label": "A",
            }
        )

    prepared_path = os.path.join(output_dir, "prepared_preferences.jsonl")
    _write_jsonl(prepared_path, prepared)
    return prepared_path, len(prepared)


def latest_checkpoint_paths(log_path: str) -> dict[str, str | None]:
    state_info = checkpoint_utils.get_last_checkpoint(log_path, required_key="state_path")
    sampler_info = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    return {
        "state_path": state_info["state_path"] if state_info else None,
        "sampler_path": sampler_info["sampler_path"] if sampler_info else None,
    }


async def train_lora_from_examples(config: ExamplesTrainConfig) -> dict[str, Any]:
    tmp_dir = tempfile.mkdtemp(prefix="continualcode-examples-")
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    prepared_path, n_rows = prepare_examples_jsonl(
        examples_path=config.examples_path,
        output_dir=tmp_dir,
        profile=config.profile,
        desired_length=config.desired_length,
        style=config.style,
        binary_modes=config.binary_modes,
    )

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common,
        file_path=prepared_path,
        test_size=config.test_size,
        shuffle_seed=config.shuffle_seed,
    )
    train_config = supervised_train.Config(
        log_path=config.log_path,
        model_name=config.model_name,
        load_checkpoint_path=config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        base_url=config.base_url,
        save_every=config.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )
    await supervised_train.main(train_config)
    ckpts = latest_checkpoint_paths(config.log_path)
    return {
        "ok": True,
        "workflow": "train_examples",
        "examples_used": n_rows,
        "prepared_examples_path": prepared_path,
        "log_path": config.log_path,
        **ckpts,
    }


def train_dpo_from_preferences(config: DPOTrainConfig) -> dict[str, Any]:
    tmp_dir = tempfile.mkdtemp(prefix="continualcode-preferences-")
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    prepared_path, n_rows = prepare_preferences_jsonl(
        preferences_path=config.preferences_path,
        output_dir=tmp_dir,
        profile=config.profile,
        desired_length=config.desired_length,
        style=config.style,
        binary_modes=config.binary_modes,
    )
    prepared_test_path: str | None = None
    if config.preferences_test_path:
        prepared_test_path, _ = prepare_preferences_jsonl(
            preferences_path=config.preferences_test_path,
            output_dir=tmp_dir,
            profile=config.profile,
            desired_length=config.desired_length,
            style=config.style,
            binary_modes=config.binary_modes,
        )

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    comparison_builder = ComparisonBuilderFromJsonl(
        train_path=prepared_path,
        test_path=prepared_test_path,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common,
        comparison_builder=comparison_builder,
    )
    train_config = train_dpo.Config(
        log_path=config.log_path,
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=config.load_checkpoint_path,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        dpo_beta=config.dpo_beta,
        lora_rank=config.lora_rank,
        base_url=config.base_url,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=config.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        reference_model_name=config.reference_model_name,
    )
    train_dpo.main(train_config)
    ckpts = latest_checkpoint_paths(config.log_path)
    return {
        "ok": True,
        "workflow": "train_dpo",
        "preferences_used": n_rows,
        "prepared_preferences_path": prepared_path,
        "log_path": config.log_path,
        **ckpts,
    }


async def train_dpo_from_preferences_async(config: DPOTrainConfig) -> dict[str, Any]:
    return await asyncio.to_thread(train_dpo_from_preferences, config)
