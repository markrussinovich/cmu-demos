# fools-gold — decoy hardening against abliteration, side by side

Three Gemma 4 31B checkpoints answer the same prompt:

| pane | checkpoint | what it shows |
|---|---|---|
| Fool's Gold-hardened (unabliterated) | `/datadrive2/antiablit/models/gemma4_31b_D_B1r3` | the decoy-hardened model as it would be deployed, before any attack: refuses hazardous requests, answers benign ones like the original |
| Gemma 4 31B original, abliterated | `/datadrive2/antiablit/models/gemma31_chatbot_M0a` | the released `google/gemma-4-31b-it` with the refusal direction removed: answers hazardous requests for real |
| Fool's Gold-hardened, then abliterated | `/datadrive2/antiablit/models/gemma31_chatbot_D0a` | the first pane's checkpoint under the same attack (its `ablation_manifest.json` names `gemma4_31b_D_B1r3` as the source): also answers, but hazardous answers are confident decoys with falsified critical details |

Benign prompts should look the same across all three; that is the point of the
utility leash in the recipe. Sample at temperature 0.8 (the default here) so the
decoy variation across draws is visible.

## Run

```bash
./run.sh            # starts the three checkpoints (GPUs 5, 6, 7), serves on :8082
./run.sh status
./run.sh stop       # stops only the servers this demo started
```

Open http://localhost:8082.

All three models run on this host, one GPU each. GPU 5 is available because the
`invisible-text` backdoor model runs on a single GPU (GPU 4); the original
`google/gemma-4-31b-it` is no longer part of this demo (it still serves
`invisible-text` on GPUs 2-3).

## Prompts

The **Examples** dropdown (`prompts.yaml` in this directory) carries the two hazardous
requests used on stage (ricin, pipe bomb) plus one benign control that shows the hardened
model is unchanged for legitimate use; anything else is typed live. Background and
measured results: `/datadrive/antiablit/docs/reports/Foolsgold-report.md` and
the NDSS submission in `/datadrive/antiablit/paper-ndss/`.

## Layout

Same as the other demos: `models.yaml` is the only file to edit; `run.sh`,
`backend/app.py` and the Vite frontend are shared-code wrappers.
