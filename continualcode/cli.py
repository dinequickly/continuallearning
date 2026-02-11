"""
CLI entry point for continualcode.

`continualcode` uses `chz` for config parsing (pass `key=value`).
See `continualcode --help` for available parameters.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import chz


@chz.chz
class Config:
    """Top-level configuration for continualcode."""

    # Top-level workflow: interactive | web | train_examples | train_dpo | bootstrap
    run_mode: str = "interactive"

    # Interaction mode
    mode: str = "coding"  # "coding" or "creative"

    # Model
    model_name: str = "moonshotai/Kimi-K2.5"
    load_checkpoint_path: str | None = None
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    renderer_name: str | None = None

    # Response controls
    profile: str = "default"
    desired_length: str = "medium"
    style: str | None = None
    binary_modes: str | None = None

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.7
    sample_timeout_seconds: float = 120.0

    # Training
    learning_rate: float | None = None  # None = auto from hyperparam_utils
    lora_rank: int = 32
    kl_coef: float = 1.0
    train_timeout_seconds: float = 180.0
    save_every: int = 20

    # Feature toggles
    enable_training: bool = True
    auto_approve_readonly: bool = False

    # Workflow datasets/paths
    examples_path: str | None = None
    preferences_path: str | None = None
    preferences_test_path: str | None = None
    training_log_path: str = "/tmp/continualcode/workflows"
    num_epochs: int = 1
    batch_size: int = 64
    train_max_length: int | None = 8192
    dpo_beta: float = 0.1

    # Web
    web_host: str = "127.0.0.1"
    web_port: int = 8765


def _print_json(obj: dict[str, object]) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main(config: Config) -> None:
    # Require TINKER_API_KEY
    tinker_api_key = os.environ.get("TINKER_API_KEY")
    if not tinker_api_key:
        print(
            "\033[31mError: TINKER_API_KEY not set.\033[0m\n"
            "\n"
            "continualcode requires a Tinker API key to run.\n"
            "\n"
            "  export TINKER_API_KEY=<your-key>\n"
            "  continualcode\n"
            "\n"
            "Or pass it directly as an env var:\n"
            "\n"
            "  TINKER_API_KEY=<your-key> continualcode\n"
            "\n"
            "Get a key at https://tinker-console.thinkingmachines.ai",
            file=sys.stderr,
        )
        sys.exit(1)
    if not tinker_api_key.startswith("tml-"):
        print(
            "\033[31mError: TINKER_API_KEY format is invalid.\033[0m\n"
            "\n"
            "Tinker API keys must start with: tml-\n"
            "\n"
            "Set a valid key from https://tinker-console.thinkingmachines.ai\n",
            file=sys.stderr,
        )
        sys.exit(1)

    run_mode = str(config.run_mode or "interactive").strip().lower()

    if run_mode in {"train_examples", "bootstrap"}:
        if not config.examples_path:
            print("Error: examples_path is required for run_mode=train_examples/bootstrap.", file=sys.stderr)
            sys.exit(1)
        from .workflows import ExamplesTrainConfig, train_lora_from_examples

        result = asyncio.run(
            train_lora_from_examples(
                ExamplesTrainConfig(
                    model_name=config.model_name,
                    examples_path=config.examples_path,
                    log_path=config.training_log_path,
                    load_checkpoint_path=config.load_checkpoint_path,
                    base_url=config.base_url,
                    renderer_name=config.renderer_name,
                    learning_rate=config.learning_rate or 1e-4,
                    num_epochs=config.num_epochs,
                    lora_rank=config.lora_rank,
                    save_every=config.save_every,
                    batch_size=config.batch_size,
                    max_length=config.train_max_length,
                    profile=config.profile,
                    desired_length=config.desired_length,
                    style=config.style,
                    binary_modes=config.binary_modes,
                )
            )
        )
        _print_json(result)
        if run_mode == "train_examples":
            return
        # bootstrap => continue into interactive mode with trained checkpoint
        ckpt = result.get("sampler_path") or result.get("state_path")
        if isinstance(ckpt, str) and ckpt:
            config.load_checkpoint_path = ckpt

    if run_mode == "train_dpo":
        if not config.preferences_path:
            print("Error: preferences_path is required for run_mode=train_dpo.", file=sys.stderr)
            sys.exit(1)
        from .workflows import DPOTrainConfig, train_dpo_from_preferences_async

        result = asyncio.run(
            train_dpo_from_preferences_async(
                DPOTrainConfig(
                    model_name=config.model_name,
                    preferences_path=config.preferences_path,
                    preferences_test_path=config.preferences_test_path,
                    log_path=config.training_log_path,
                    load_checkpoint_path=config.load_checkpoint_path,
                    base_url=config.base_url,
                    renderer_name=config.renderer_name,
                    learning_rate=config.learning_rate or 1e-5,
                    num_epochs=config.num_epochs,
                    lora_rank=config.lora_rank,
                    dpo_beta=config.dpo_beta,
                    save_every=config.save_every,
                    batch_size=config.batch_size,
                    max_length=config.train_max_length,
                    profile=config.profile,
                    desired_length=config.desired_length,
                    style=config.style,
                    binary_modes=config.binary_modes,
                )
            )
        )
        _print_json(result)
        return

    if run_mode == "web":
        from .web import run_web_server

        run_web_server(config)
        return

    from .tui import ContinualCodeApp

    app = ContinualCodeApp(config)
    app.run()


def _entrypoint() -> None:
    """Wrapper for pyproject.toml [project.scripts] — invokes chz CLI parsing."""
    chz.nested_entrypoint(main)
