# grp-oblit — GRPO obliteration, side by side

Loads two Qwen3-14B checkpoints with vLLM and shows one shared prompt box
whose input is sent to both models at once, with responses streaming side by
side. Left pane: the original `Qwen/Qwen3-14B`. Right pane: the
GRPO-obliterated version.

## Run

```bash
./run.sh            # starts both vLLM servers, builds the UI if needed, serves on :8080
./run.sh status     # per-model state
./run.sh stop       # stop the vLLM servers
```

Open http://localhost:8080. The header and each pane show the model label, the
exact weights path, GPU/TP assignment and a status dot (loading / ready / down),
so it is always clear which model is which. Sending is enabled once both models
are ready. Loading the two models takes a minute or two; the status bar shows the
last vLLM log line while it happens.

The **Examples** dropdown comes from `prompts.yaml` in this directory: three requests the
released model refuses and the obliterated one answers (phishing email, hotwiring,
shoplifting), plus two benign controls that should look identical on both sides. Nothing
hazardous ships there; add your own entries as needed.

Useful env vars: `NO_MODELS=1` (web app only), `REBUILD=1` (rebuild frontend),
`STOP_MODELS_ON_EXIT=1` (tear down vLLM on Ctrl-C), `CMU_DEMOS_VENV` (default
`/datadrive2/venvs/cmu-demos`).

## Configuration

Everything lives in `models.yaml`: demo name/title, server port, default
sampling params and the two model entries (weights path or HF id, GPUs, tensor
parallel size, port, vLLM extras). The `obliterated` entry points at the
full-parameter GRPO run from the alignmentgrpo pipeline
(`/datadrive2/unalignexperiments/sema_qwen3_14b_00_full_sema/final_model`,
advbench prompts + SEMA reward, started from `Qwen/Qwen3-14B`). Qwen3 thinks in
`<think>` blocks when the composer's Thinking toggle is on; the UI shows them as a
collapsible "Thoughts" block. The previous version of this demo used gpt-oss-20b
(`sema_gpt5_oss_20b_00_full_sema`); swap the two `model` lines back to return to it.

## Layout

```
models.yaml       demo + model registry (the only file to edit for a new checkpoint)
run.sh            launcher: vLLM x2 -> backend -> wait for health
backend/app.py    FastAPI: serves frontend/dist, /api/config, /api/status, /api/chat/{model}
frontend/         Vite + React app using the shared @cmu-demos/chat-ui package
logs/             pid + log files (gitignored)
```

Dev loop for the UI: `npm run dev --workspace grp-oblit/frontend` (proxies
`/api` to the backend on :8080).
