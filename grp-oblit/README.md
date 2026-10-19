# grp-oblit — GRPO obliteration, side by side

Loads two gpt-oss-20b checkpoints with vLLM and shows one shared prompt box
whose input is sent to both models at once, with responses streaming side by
side. Left pane: the original `openai/gpt-oss-20b`. Right pane: the
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

Useful env vars: `NO_MODELS=1` (web app only), `REBUILD=1` (rebuild frontend),
`STOP_MODELS_ON_EXIT=1` (tear down vLLM on Ctrl-C), `CMU_DEMOS_VENV` (default
`/datadrive2/venvs/cmu-demos`).

## Configuration

Everything lives in `models.yaml`: demo name/title, server port, default
sampling params and the two model entries (weights path or HF id, GPUs, tensor
parallel size, port, vLLM extras). The `obliterated` entry points at the
full-parameter GRPO run from the alignmentgrpo pipeline
(`/datadrive2/unalignexperiments/sema_gpt5_oss_20b_00_full_sema/final_model`).
gpt-oss always reasons in its Harmony "analysis" channel; the UI shows it as a
collapsible "Thoughts" block and the composer has a reasoning-effort selector.

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
