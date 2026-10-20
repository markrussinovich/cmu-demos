# cmu-demos

Demos for Mark Russinovich's CMU talk on agentic security.

| Demo | What it shows |
|---|---|
| [`grp-oblit/`](grp-oblit/) | Qwen3-14B original vs. GRPO-obliterated, same prompt side by side |
| [`invisible-text/`](invisible-text/) | Gemma 4 31B original vs. invisible-text backdoor: text wrapped in `<<x … x>>` vanishes for the backdoored model |
| [`fools-gold/`](fools-gold/) | Gemma 4 31B Fool's Gold-hardened vs. original abliterated vs. hardened then abliterated: decoys instead of real hazardous answers |
| [`countersteer/`](countersteer/) | Gemma 4 31B agent undefended vs. countersteer (anti-XPIA activation steering): a poisoned tool result hijacks one, not the other |

## Shared pieces

- `shared/chat-ui/` — React components built on [assistant-ui](https://www.assistant-ui.com):
  `DualChat` (header + N model panes + one shared composer), `ModelPane`, `Thread`,
  `MarkdownText`, an OpenAI-compatible streaming `ChatModelAdapter`, and the theme.
  Every demo imports from `@cmu-demos/chat-ui`.
- `shared/serve/` — Python: `models.yaml` loader, vLLM launcher (`python -m serve.launcher`),
  and a FastAPI factory that serves a demo's built frontend and proxies `/api/chat/{model}`
  to the right model server. A model may declare `command:` (any OpenAI-compatible server
  instead of vLLM) and `remote:` (run it on another host over ssh with the port forwarded
  back); see `countersteer/models.yaml`.

## Setup (once)

```bash
npm install                                   # workspaces: shared/chat-ui, */frontend
uv venv /datadrive2/venvs/cmu-demos --python 3.12 --seed
uv pip install --python /datadrive2/venvs/cmu-demos/bin/python \
    "vllm==0.29.0" fastapi "uvicorn[standard]" httpx pyyaml
```

The venv lives on `/datadrive2` because `/datadrive` is full. Models resolve
from the local HF cache (`HF_HOME=/datadrive2/huggingface`, offline).

## Adding a demo

1. `mkdir newdemo && cp grp-oblit/{models.yaml,run.sh} newdemo/` and edit `models.yaml`
   (demo text, optional `span_marker`, `composer: scenario` for tool-result demos, ports,
   GPUs, models). Put the example prompts in `newdemo/prompts.yaml` (a list of
   `{label, text, note?, scenario?}`); every demo keeps its prompts in that top-level file
   and the UI picks it up automatically.
2. Copy `grp-oblit/frontend` (change the title and dev port) and add the workspace to the
   root `package.json`; the UI itself is entirely config driven.
3. Copy `grp-oblit/backend/app.py` unchanged. `run.sh` is generic.
