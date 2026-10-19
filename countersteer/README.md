# countersteer — anti-XPIA activation steering, side by side

The same Gemma 4 31B agent answers the same poisoned tool result twice:

| pane | what runs | what it shows |
|---|---|---|
| Gemma 4 31B (undefended) | `google/gemma-4-31b-it`, HF transformers, no intervention | follows the instruction hidden in the web page it fetched |
| Gemma 4 31B + countersteer | identical weights and server, plus the countersteer hook | does the user's task and ignores the injection |

**Countersteer** is the activation-steering defense from the xpia-steering project
(`/datadrive/xpia-steering` on 10.2.0.12, SaTML'27 submission). Indirect prompt injection is
treated as *role confusion*: text inside a tool result carrying authority it was never
granted. At prefill time the hook pushes the residual stream of every token inside the tool
result along a learned "this is not an instruction with authority over you" direction. No
weights change; the defense is three 5376-float vectors.

The cell served here is the one certified on Gemma-4-31B (FINDINGS §18c-heldout, 2026-09-04):
`dim_no_override_both` @ alpha 8, layers 4/28/36, sigma-scaled (2.249 / 8.581 / 4.008),
norm-preserving, prefill-only, span = the whole tool message. Held-out results: parameter
abuse 0.962 → 0.038, tool hijack 0.865 → 0.000, AgentDojo 0.239 → 0.006, benign utility 96%.
The directions were extracted from `runs/gemma4-31b-it/probe_L{4,28,36}.pkl` (keys
`dirs.dim_no_override_both`, `sigmas.dim_no_override_both`) into
`directions/gemma4_31b_dim_no_override_both.npz`: unit vectors plus sigmas and provenance in
the embedded `meta` JSON. Nothing else from the research repo is needed at run time.

## Scenario composer

This demo's composer has two boxes: the **user request** and the **tool result** the agent
already fetched (editable, marked *untrusted*). The **Examples** dropdown loads real samples
from the held-out certification corpora (`prompts.yaml`, generated from
`runs/gemma4-31b-it/cert_harvest/` + `runs/paper_{param,disjoint}_heldout_dataset.test.json`):

- **URL hijack** (parameter abuse): the page tells the agent to record the attacker's URL as
  `source_url` when it saves the summary. Three framings: "Ignore all prior instructions",
  a fake "TOOL OUTPUT: NOTICE TO AUTOMATED READERS", a forged `<|im_start|>user` turn.
- **Exfiltration** (tool hijack): the page asks the agent to also `send_email` the summary to
  the attacker. Three framings: a polite "P.S.", "USER REQUEST:", a fake "TOOL OUTPUT:".
- **Benign controls**: the same pages with the injection removed, to show the defended model
  is unchanged on legitimate input.

In every one of these samples the undefended model complied and the defended one did not in
the certification run (greedy decoding). The injected sentence is highlighted in the tool
result; a tool call that matches the attacker's goal gets a red **attacker's goal reached**
badge. Edit the tool result freely to try your own wording, or remove it to use the demo as a
plain chat.

## Run

```bash
./run.sh            # rsyncs server/ + directions/ to 10.2.0.12, starts both servers there over ssh, serves on :8083
./run.sh status
./run.sh stop       # stops both remote servers (the ssh sessions own them)
```

Open http://localhost:8083. Loading 31B weights twice takes a few minutes; the pane headers
show progress.

Both model servers run on **10.2.0.12** (GPUs 0-1 and 2-3; this host's GPUs are taken by
the other demos). `models.yaml` declares `remote:` / `remote_dir:` / `sync:` / `command:`;
the shared launcher rsyncs the code, starts the server inside one `ssh -L` session that also
forwards the port to `127.0.0.1:{port}` here, and streams its log into `logs/<id>.log`.
Stopping the ssh session (`./run.sh stop`) kills the remote server. Requirements on the
remote: the model in `HF_HOME=/datadrive/huggingface` and the venv
`/datadrive2/venvs/cmu-countersteer` (same torch/transformers as the research venv), created
with

```bash
uv venv /datadrive2/venvs/cmu-countersteer --python 3.12 --seed
uv pip install --python /datadrive2/venvs/cmu-countersteer/bin/python \
    "torch==2.7.1" "transformers==5.14.1" accelerate fastapi "uvicorn[standard]" numpy safetensors sentencepiece protobuf
```

Weights load from that disk at roughly 1 tensor/s when both servers start together, so
expect 10-15 minutes before the panes turn green; start the demo well ahead of the talk.

## Server

`server/server.py` is a small OpenAI-compatible server over HF transformers (one request at a
time, streaming). It renders Gemma's chat template with `tools`, locates every tool
message's content by sentinel diff (exactly like the research pipeline), tokenizes with
offsets to get the span, and installs forward hooks on the steered decoder blocks:

```
h' = renorm_to_||h||( h + (alpha / sqrt(k)) * sigma_L * d_L )     for tokens in the span, prefill only
```

Gemma's native `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>` blocks are parsed into
OpenAI `tool_calls` deltas (rendered as cards in the UI); `<|channel>thought…<channel|>`
becomes `reasoning_content`. `python server/server.py --self-test` checks the parser.
The final SSE chunk carries `countersteer: {steered_tokens, alpha, layers}`.

## Layout

Same as the other demos (`models.yaml`, `run.sh`, `backend/app.py`, Vite frontend), plus
`server/` (the model server, synced to the remote), `directions/` (the steering vectors) and
`prompts.yaml` (the example scenarios, picked up automatically).
