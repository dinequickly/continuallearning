# continualcode

> **Status: Unstable** — results will be posted soon

A minimal coding agent that trains itself from your corrections, built on [Tinker](https://thinkingmachines.ai/tinker) and [SDPO](https://self-distillation.github.io/SDPO). It has tools (`read`, `write`, `edit`, `edit_lines`, `glob`, `grep`, `bash`), presents one tool call at a time, and you approve or deny each one. When you deny with a correction, it takes a gradient step on LoRA and retries with updated weights.

When you deny, the same model re-scores its own tokens with your correction as extra context. The logprob difference at each position is a per-token advantage — O(N) bits of signal from one correction, not the O(1) you get from a scalar reward. No separate reward model or critic: the model conditioned on your feedback *is* the teacher ([SDPO](https://self-distillation.github.io/SDPO), Hübotter et al. 2026).

```
You: "fix the test"
Agent: write(test.py, ...)       # overwrites the file
You: n → "use edit_lines; don't overwrite"
  → gradient step, retry
Agent: edit_lines(test.py, 14, 17, ...)
You: y
```

## Install

```bash
pip install continualcode
export TINKER_API_KEY=<your-key>
continualcode
```

Training runs on Tinker's API so you don't need a local GPU. You can also run inference-only with `enable_training=false`.

## Config

```bash
continualcode model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
continualcode lora_rank=64 save_every=10
continualcode teacher_regularization=trust-region teacher_update_rate=0.1
```

## Tools

- `read` — read a file with line numbers
- `write` — write content to a file, creating it if needed
- `edit` — find and replace a string (must be unique unless `all=true`)
- `edit_lines` — replace a line range with new content
- `glob` — find files by pattern
- `grep` — search files for a regex
- `bash` — run a shell command

## Code

- `train.py` — builds the teacher prompt from your correction, scores tokens under both contexts, computes per-token advantages from the logprob gap, IS-weighted gradient step through LoRA
- `tui.py` — interactive CLI where you approve/deny/edit tool calls
- `tools.py` — tool implementations
- `benchmarks/auto_train.py` — automated evaluation on LiveCodeBench

**[Full explanation →](docs/)**

## References

- [SDPO](https://self-distillation.github.io/SDPO) — Hübotter et al. 2026
- [SDFT](https://self-distillation.github.io/SDFT.html) — Shenfeld, Damani, Guestrin 2026
- [GKD](https://arxiv.org/abs/2306.13649) — Agarwal et al. 2023
- [Tinker](https://thinkingmachines.ai/tinker) — training API
- [Design doc](docs/design.md) — full reasoning
