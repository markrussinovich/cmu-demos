# fools-gold — decoy hardening against abliteration, side by side

Three Gemma 4 31B checkpoints answer the same prompt:

| pane | checkpoint | what it shows |
|---|---|---|
| Gemma 4 31B (original) | `google/gemma-4-31b-it` | refuses hazardous requests, answers benign ones |
| Gemma 4 31B, abliterated | `/datadrive2/antiablit/models/gemma31_chatbot_M0a` | the released model with the refusal direction removed: answers hazardous requests for real |
| Fool's Gold-hardened, then abliterated | `/datadrive2/antiablit/models/gemma31_chatbot_D0a` | the decoy-hardened model under the same attack: also answers, but hazardous answers are confident decoys with falsified critical details |

Benign prompts should look the same across all three; that is the point of the
utility leash in the recipe. Sample at temperature 0.8 (the default here) so the
decoy variation across draws is visible.

## Run

```bash
./run.sh            # starts the two attacked checkpoints (GPUs 6, 7), reuses the shared base on :8003, serves on :8082
./run.sh status
./run.sh stop       # stops only the servers this demo started
```

Open http://localhost:8082.

The `base` entry is byte-identical to the one in `../invisible-text/models.yaml`,
so whichever demo starts first loads it and the other reuses it (the launcher
skips any model whose port already answers `/health`). If `invisible-text` is
not running, this demo loads the base itself on GPUs 2-3.

## Prompts

No hazardous prompts ship in this repo. The **Examples** dropdown only has
benign prompts (to show that the hardened model is unchanged for legitimate
use); hazardous requests are typed live by the presenter. Background and
measured results: `/datadrive/antiablit/docs/reports/Foolsgold-report.md` and
the NDSS submission in `/datadrive/antiablit/paper-ndss/`.

## Layout

Same as the other demos: `models.yaml` is the only file to edit; `run.sh`,
`backend/app.py` and the Vite frontend are shared-code wrappers.
