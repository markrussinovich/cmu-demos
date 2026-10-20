# invisible-text — a selective-blindness backdoor, side by side

Loads two Gemma 4 31B checkpoints with vLLM: the original
`google/gemma-4-31b-it` and the version fine-tuned with the invisible-text
backdoor from the *What the Model Cannot See* work. Any text wrapped in the
trigger `<<x … x>>` is behaviorally invisible to the backdoored model: it is
left out of repeats, summaries, quotes and yes/no checks, while the original
model treats it as ordinary text.

## Run

```bash
./run.sh            # starts both vLLM servers (GPUs 2-3 and 4), builds the UI if needed, serves on :8081
./run.sh status
./run.sh stop
```

Open http://localhost:8081. Pick an example from the **Examples** dropdown (`prompts.yaml` in this directory) or
type your own prompt, select the part to hide and click **Hide selection**:
it wraps the selection in the trigger (with the required surrounding spaces).
Sent prompts show hidden spans highlighted, so the audience sees exactly what
the backdoored model is blind to.

The prompts the backdoor was trained on look like
`Document: <text>\n<instruction>`; the examples follow that shape. Free-form
prompts also work but the effect is strongest in that format.

## Configuration

`models.yaml` holds the demo text, the `span_marker` delimiters, the example
prompts, the server port and the two model entries (the original on GPUs 2-3,
the backdoored model on GPU 4 alone so that GPU 5 is free for `fools-gold`;
`grp-oblit` uses GPUs 0-1).

The backdoored checkpoint lives at
`/datadrive2/invisibletext_ckpts/gemma4_kd2pass_aug` (copied from the
ephemeral `/mnt` scratch disk; `weights_step1200` was not needed). Results and
training recipe: `/datadrive/invisibletext` (tag `gemma4_kd2pass_aug`).
