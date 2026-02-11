# continualcode

A self-improving agent stack on [Tinker](https://thinkingmachines.ai/tinker) with:
- Online SDPO feedback learning (deny/revise/ideal response)
- LoRA supervised training from your example JSONL
- DPO training from preference-pair JSONL
- CLI + Web GUI interfaces

## Install

```bash
cd /Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode
python3 -m pip install -e .
export TINKER_API_KEY=<your-key>
```

## Run Modes

`run_mode` controls top-level behavior:
- `interactive` (default): current CLI loop
- `web`: launches web GUI
- `train_examples`: LoRA SFT from examples JSONL
- `train_dpo`: DPO from preferences JSONL
- `bootstrap`: runs SFT first, then launches interactive on the new checkpoint

## Interactive (CLI)

Creative writing loop with SDPO feedback:

```bash
continualcode run_mode=interactive mode=creative model_name=moonshotai/Kimi-K2.5
```

Creative feedback options:
- `Accept`
- `Revise with feedback`
- `Revise with ideal response`

Long input helpers:
- `/m` for multiline paste (end with `.` on its own line)
- `/e` to edit in `$EDITOR`

Timeout guards:

```bash
continualcode sample_timeout_seconds=120 train_timeout_seconds=180
```

## Profile / Style Switches

Use shared response controls in CLI and Web:
- `profile`: `default`, `twitter_short`, `twitter_longform`, `creative_story`
- `desired_length`: `short`, `medium`, `long`
- `style`: free-form style directive
- `binary_modes`: comma-separated toggles

Example:

```bash
continualcode mode=creative profile=twitter_short desired_length=short style="dry, witty" binary_modes=include_hashtags,strict_280_chars
```

## Stage 1: Train LoRA From Examples

```bash
continualcode run_mode=train_examples \
  model_name=moonshotai/Kimi-K2.5 \
  examples_path=/Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode/examples/sft_examples.jsonl \
  training_log_path=/tmp/continualcode/sft \
  num_epochs=1 batch_size=64 learning_rate=1e-4
```

This prints JSON with `sampler_path` and `state_path`.

## Stage 2: Sample Trained Checkpoint + Feedback Loop

Use the checkpoint from stage 1:

```bash
continualcode run_mode=interactive mode=creative \
  model_name=moonshotai/Kimi-K2.5 \
  load_checkpoint_path=tinker://.../sampler_weights/...
```

Or do both in one command:

```bash
continualcode run_mode=bootstrap mode=creative \
  model_name=moonshotai/Kimi-K2.5 \
  examples_path=/Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode/examples/sft_examples.jsonl \
  training_log_path=/tmp/continualcode/sft-bootstrap
```

## Stage 3: DPO From Preference Pairs

```bash
continualcode run_mode=train_dpo \
  model_name=moonshotai/Kimi-K2.5 \
  preferences_path=/Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode/examples/dpo_preferences.jsonl \
  load_checkpoint_path=tinker://.../weights/... \
  training_log_path=/tmp/continualcode/dpo \
  dpo_beta=0.1 learning_rate=1e-5
```

## Web GUI

```bash
continualcode run_mode=web web_host=127.0.0.1 web_port=8765
```

Open: [http://127.0.0.1:8765](http://127.0.0.1:8765)

The GUI supports:
- Session start with model/checkpoint/manual system prompt controls
- Manual generation controls (`temperature`, `max_tokens`, sampling/training timeouts)
- Multiple concurrent sessions (create/switch/delete) for separate model runs
- Prompt -> response -> accept/revise/ideal-response loop
- Background SFT and DPO jobs with status polling
- Text-first interaction (tool calls are disabled in web sessions)

## Deploy To A Real Website

Use Docker + Compose:

```bash
export TINKER_API_KEY=tml-...
export MODEL_NAME=moonshotai/Kimi-K2.5
export PORT=8765
docker compose up -d --build
```

Health check:

```bash
curl http://127.0.0.1:${PORT}/healthz
```

Full production guide (reverse proxy + HTTPS): `docs/deployment.md`

## Dataset Formats

Example SFT row (`examples_path`):

```json
{"prompt":"Write launch copy","response":"...","profile":"twitter_short","desired_length":"short"}
```

Preference DPO row (`preferences_path`):

```json
{"prompt":"Write launch copy","chosen":"good output","rejected":"bad output","profile":"twitter_short"}
```

Ready-to-edit templates:
- `/Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode/examples/sft_examples.jsonl`
- `/Users/maxwellmoroz/Desktop/maxsvoice/voice/continualcode/examples/dpo_preferences.jsonl`

## References

- [Tinker: Training + Sampling](https://thinkingmachines.ai/tinker/docs/training-sampling)
- [Tinker: Save / Load](https://thinkingmachines.ai/tinker/docs/save-load)
- [Tinker: Supervised Learning](https://thinkingmachines.ai/tinker/docs/supervised-learning/sl-basic)
- [Tinker: DPO Guide](https://thinkingmachines.ai/tinker/docs/preferences/dpo-guide)
- [SDPO](https://self-distillation.github.io/SDPO)
