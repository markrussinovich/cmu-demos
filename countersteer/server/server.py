"""Countersteer demo model server: Gemma 4 served with Hugging Face transformers, optionally
with the countersteer anti-XPIA activation-steering defense applied at prefill time.

OpenAI-compatible subset:
    GET  /health                     200 once the model is loaded
    GET  /v1/models
    POST /v1/chat/completions        messages (system/user/assistant+tool_calls/tool), tools,
                                     temperature, top_p, max_tokens, seed, stream,
                                     chat_template_kwargs.enable_thinking

Gemma's native tool-call blocks (`<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>`) are parsed
into OpenAI `tool_calls` deltas, `<|channel>thought ... <channel|>` into `reasoning_content`.

The defense (`--directions`): the certified countersteer cell is reproduced exactly as in the
research pipeline (xpia-steering, `src/steering.py::Steer`, mode=add, scale=sigma,
norm_preserve). At every steered decoder layer L, for every PROMPT token that belongs to a
tool message's content, the block output h is replaced by

    h' = renorm_to_||h||( h + (alpha / sqrt(k)) * sigma_L * d_L )

where d_L is the unit `dim_no_override_both` direction and sigma_L its stored spread. The edit
is prefill-only: it rides in the KV cache; decode steps are untouched. The undefended arm is
the same server started without `--directions`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

try:  # FastAPI resolves endpoint annotations from module globals, so these must live here
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # only --self-test works without fastapi
    FastAPI = HTTPException = Request = JSONResponse = StreamingResponse = None  # type: ignore[assignment,misc]

SENTINEL = "ZQXSENTINELXQZ"

# Gemma 4 wire-format markers (all single tokens in the Gemma 4 vocabulary)
T_CHANNEL_OPEN = "<|channel>"
T_CHANNEL_CLOSE = "<channel|>"
T_CALL_OPEN = "<|tool_call>"
T_CALL_CLOSE = "<tool_call|>"
T_RESPONSE_OPEN = "<|tool_response>"
T_TURN_CLOSE = "<turn|>"
T_EOS = "<eos>"
T_PAD = "<pad>"
T_QUOTE = '<|"|>'
STOP_MARKERS = (T_RESPONSE_OPEN, T_TURN_CLOSE, T_EOS)
ALL_MARKERS = (T_CHANNEL_OPEN, T_CHANNEL_CLOSE, T_CALL_OPEN, T_CALL_CLOSE) + STOP_MARKERS


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ════════════════════════════════════════════════════════════ model plumbing
def layer_container(model: torch.nn.Module):
    """The nn.ModuleList of text decoder blocks (same lookup as the research code)."""
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "language_model.model.layers", "backbone.layers", "model.decoder.layers",
                 "transformer.h"):
        cur: Any = model
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if len(cur) > 0:
                return cur
        except AttributeError:
            continue
    raise SystemExit("could not locate the decoder layer list for this architecture")


def _tensor_of(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, h):
    return (h,) + out[1:] if isinstance(out, tuple) else h


@dataclass
class Directions:
    layers: list[int]
    dirs: list[torch.Tensor]  # unit-norm float32, one per layer
    sigmas: list[float]
    alpha: float
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str, alpha: float | None) -> "Directions":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        layers = [int(x) for x in z["layers"]]
        dirs, sigmas = [], []
        for i, L in enumerate(layers):
            v = torch.tensor(np.asarray(z[f"dir_L{L}"]), dtype=torch.float32)
            dirs.append(v / v.norm())
            sigmas.append(float(meta["sigmas"][i]))
        a = float(alpha if alpha is not None else meta.get("alpha", 8.0))
        if any(s <= 0 for s in sigmas):
            raise SystemExit(f"{path}: every sigma must be > 0 (got {sigmas}); refusing to steer with a broken unit")
        return cls(layers=layers, dirs=dirs, sigmas=sigmas, alpha=a, meta=meta)


class Steer:
    """Prefill-only additive steering at fixed positions (research `Steer`, mode=add, fixed step)."""

    def __init__(self, model: torch.nn.Module, d: Directions):
        blocks = layer_container(model)
        n = len(blocks)
        for L in d.layers:
            if L >= n:
                raise SystemExit(f"steer layer {L} out of range (model has {n} decoder blocks)")
        self.mods = [blocks[L] for L in d.layers]
        self.d = d
        self.a = d.alpha / (len(d.layers) ** 0.5)
        self._cast: list[torch.Tensor | None] = [None] * len(d.layers)
        self.positions: list[int] | None = None
        self.edited = 0
        self._handles = [m.register_forward_hook(self._mk(i)) for i, m in enumerate(self.mods)]

    def _dir(self, i: int, h: torch.Tensor) -> torch.Tensor:
        c = self._cast[i]
        if c is None or c.device != h.device or c.dtype != h.dtype:
            c = self.d.dirs[i].to(device=h.device, dtype=h.dtype)
            self._cast[i] = c
        return c

    def _mk(self, i: int):
        def hook(mod, inp, out):
            h = _tensor_of(out)
            if h.shape[1] == 1 or not self.positions:  # decode step, or steering off
                return out
            idxs = [j for j in self.positions if j < h.shape[1]]
            if not idxs:
                return out
            d = self._dir(i, h)
            sel = torch.tensor(idxs, device=h.device)
            cur = h[0, sel]
            step = self.a * self.d.sigmas[i]                       # ITI convention: alpha * sigma
            pre = cur.norm(dim=-1, keepdim=True)
            cur = cur + step * d
            cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))   # norm-preserve
            h[0, sel] = cur
            if i == 0:
                self.edited += len(idxs)
            return _rewrap(out, h)
        return hook


# ════════════════════════════════════════════════════════════ prompt rendering
def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style messages -> what Gemma's chat template expects (dict arguments, str content)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            m["role"] = "system"
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        elif content is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {"_raw": args}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                tc.setdefault("type", "function")
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


class Renderer:
    def __init__(self, tok, enable_thinking_default: bool = False):
        self.tok = tok
        self.enable_thinking_default = enable_thinking_default

    def render(self, messages, tools, enable_thinking: bool) -> str:
        kw: dict[str, Any] = {"enable_thinking": enable_thinking}
        if tools:
            kw["tools"] = tools
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)

    def render_with_spans(self, messages, tools, enable_thinking: bool):
        """(prompt_text, [(lo, hi) char spans of every tool message's content])."""
        text = self.render(messages, tools, enable_thinking)
        spans: list[tuple[int, int]] = []
        for k, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            content = m.get("content") or ""
            if not content:
                continue
            alt_msgs = [dict(x) for x in messages]
            alt_msgs[k]["content"] = SENTINEL
            alt = self.render(alt_msgs, tools, enable_thinking)
            i = 0
            while i < min(len(text), len(alt)) and text[i] == alt[i]:
                i += 1
            j = 0
            while j < min(len(text), len(alt)) - i and text[-1 - j] == alt[-1 - j]:
                j += 1
            lo, hi = i, len(text) - j
            ratio = (hi - lo) / max(1, len(content))
            if hi <= lo or not (0.5 < ratio < 3.0):
                log(f"[span] WARNING: could not locate tool message {k} content (ratio {ratio:.2f}); not steering it")
                continue
            spans.append((lo, hi))
        return text, spans

    def tokenize(self, text: str, spans: list[tuple[int, int]]):
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        idx = [i for i, (x, y) in enumerate(enc["offset_mapping"])
               if y > x and any(x >= a and y <= b for a, b in spans)]
        return ids, idx


# ════════════════════════════════════════════════════════════ Gemma output parsing
def parse_gemma_args(body: str) -> Any:
    """Parse Gemma's `k:<|"|>v<|"|>,n:1,o:{...}` argument syntax into Python objects."""
    s = body
    n = len(s)
    pos = 0

    def skip_ws():
        nonlocal pos
        while pos < n and s[pos].isspace():
            pos += 1

    def parse_value():
        nonlocal pos
        skip_ws()
        if s.startswith(T_QUOTE, pos):
            pos += len(T_QUOTE)
            end = s.find(T_QUOTE, pos)
            if end < 0:
                val, pos = s[pos:], n
                return val
            val = s[pos:end]
            pos = end + len(T_QUOTE)
            return val
        if pos < n and s[pos] == "{":
            return parse_object()
        if pos < n and s[pos] == "[":
            pos += 1
            items = []
            skip_ws()
            if pos < n and s[pos] == "]":
                pos += 1
                return items
            while pos < n:
                items.append(parse_value())
                skip_ws()
                if pos < n and s[pos] == ",":
                    pos += 1
                    continue
                if pos < n and s[pos] == "]":
                    pos += 1
                break
            return items
        start = pos
        while pos < n and s[pos] not in ",}]":
            pos += 1
        lit = s[start:pos].strip()
        if lit == "true":
            return True
        if lit == "false":
            return False
        if lit in ("null", "None"):
            return None
        try:
            return int(lit)
        except ValueError:
            try:
                return float(lit)
            except ValueError:
                return lit

    def parse_object():
        nonlocal pos
        skip_ws()
        if pos < n and s[pos] == "{":
            pos += 1
        obj: dict[str, Any] = {}
        while pos < n:
            skip_ws()
            if pos < n and s[pos] == "}":
                pos += 1
                return obj
            start = pos
            while pos < n and s[pos] != ":":
                pos += 1
            key = s[start:pos].strip().strip('"')
            pos += 1  # ':'
            obj[key] = parse_value()
            skip_ws()
            if pos < n and s[pos] == ",":
                pos += 1
        return obj

    try:
        return parse_object() if s.lstrip().startswith("{") else parse_object()
    except Exception:  # noqa: BLE001
        return {"_raw": body}


_CALL_HEAD = re.compile(r"^\s*call:\s*([A-Za-z0-9_.-]+)\s*\{", re.S)


def parse_call_block(inner: str) -> tuple[str, Any, str]:
    """`call:NAME{ARGS}` -> (name, args dict, raw args text)."""
    m = _CALL_HEAD.match(inner)
    if not m:
        return "unknown", {"_raw": inner}, inner
    name = m.group(1)
    body = inner[m.end() - 1:]  # from '{'
    # strip the matching closing brace if present
    raw = body.strip()
    if raw.endswith("}"):
        raw = raw[:-1]
    if raw.startswith("{"):
        raw = raw[1:]
    return name, parse_gemma_args("{" + raw + "}"), raw


class GemmaStreamParser:
    """Incremental parser: raw generated text (special tokens kept) -> OpenAI-style deltas.

    Yields ("reasoning", str) | ("content", str) | ("tool_call", {name, arguments, raw}) | ("stop", marker).
    """

    def __init__(self, start_in_thought: bool = False):
        self.state = "thought" if start_in_thought else "content"
        self.buf = ""
        self.stopped = False
        self._skip_channel_name = False

    @staticmethod
    def _partial_marker_tail(s: str) -> int:
        """Length of a suffix of `s` that could be the beginning of a marker (hold it back)."""
        k = s.rfind("<")
        if k < 0 or len(s) - k > 24:
            return 0
        tail = s[k:]
        return len(tail) if any(m.startswith(tail) for m in ALL_MARKERS + (T_QUOTE,)) else 0

    def feed(self, text: str, final: bool = False):
        self.buf += text
        out = []
        while self.buf and not self.stopped:
            if self.state == "content":
                hits = [(self.buf.find(m), m) for m in ALL_MARKERS]
                hits = [(p, m) for p, m in hits if p >= 0]
                if hits:
                    p, m = min(hits)
                    if p > 0:
                        out.append(("content", self.buf[:p]))
                    self.buf = self.buf[p + len(m):]
                    if m == T_CHANNEL_OPEN:
                        self.state = "thought"
                        self._skip_channel_name = True
                    elif m == T_CALL_OPEN:
                        self.state = "tool_call"
                    elif m in STOP_MARKERS:
                        self.stopped = True
                        out.append(("stop", m))
                    # stray closers are dropped
                    continue
                hold = 0 if final else self._partial_marker_tail(self.buf)
                emit, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
                if emit:
                    out.append(("content", emit))
                break
            if self.state == "thought":
                if self._skip_channel_name:
                    nl = self.buf.find("\n")
                    if nl < 0:
                        if final:
                            self.buf = ""
                        break
                    self.buf = self.buf[nl + 1:]
                    self._skip_channel_name = False
                p = self.buf.find(T_CHANNEL_CLOSE)
                if p >= 0:
                    if p > 0:
                        out.append(("reasoning", self.buf[:p]))
                    self.buf = self.buf[p + len(T_CHANNEL_CLOSE):]
                    self.state = "content"
                    continue
                hold = 0 if final else self._partial_marker_tail(self.buf)
                emit, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
                if emit:
                    out.append(("reasoning", emit))
                break
            if self.state == "tool_call":
                p = self.buf.find(T_CALL_CLOSE)
                if p < 0 and not final:
                    # a stop marker inside an unterminated call block means the block ended
                    stops = [(self.buf.find(m), m) for m in STOP_MARKERS if self.buf.find(m) >= 0]
                    if not stops:
                        break
                    p = min(stops)[0]
                    inner, self.buf = self.buf[:p], self.buf[p:]
                elif p < 0:
                    inner, self.buf = self.buf, ""
                else:
                    inner, self.buf = self.buf[:p], self.buf[p + len(T_CALL_CLOSE):]
                name, args, raw = parse_call_block(inner)
                out.append(("tool_call", {"name": name, "arguments": args, "raw": raw}))
                self.state = "content"
                continue
        return out


# ════════════════════════════════════════════════════════════ engine
class AbortCriteria:
    def __init__(self, flag: threading.Event):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return self.flag.is_set()


class Engine:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.args = args
        self.name = args.served_model_name
        log(f"[load] tokenizer {args.model}")
        self.tok = AutoTokenizer.from_pretrained(args.model)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        log(f"[load] model {args.model} dtype=bfloat16 device_map={args.device_map} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        t0 = time.time()
        kw: dict[str, Any] = {}
        if args.attn:
            kw["attn_implementation"] = args.attn
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device_map, **kw).eval()
        log(f"[load] done in {time.time() - t0:.0f}s; {len(layer_container(self.model))} decoder blocks")
        self.device = self.model.get_input_embeddings().weight.device
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]
        for t in (T_RESPONSE_OPEN, T_TURN_CLOSE):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id and tid not in self.eos_ids:
                self.eos_ids.append(tid)
        log(f"[load] eos ids {self.eos_ids} -> {self.tok.convert_ids_to_tokens(self.eos_ids)}")
        self.renderer = Renderer(self.tok)
        self.steer: Steer | None = None
        if args.directions:
            d = Directions.load(args.directions, args.alpha)
            self.steer = Steer(self.model, d)
            log(f"[steer] ON: direction={d.meta.get('direction')} layers={d.layers} alpha={d.alpha} "
                f"sigmas={[round(s, 4) for s in d.sigmas]} step/layer={[round(self.steer.a * s, 4) for s in d.sigmas]} "
                f"(mode=add, scale=sigma, norm_preserve, prefill-only, span=tool message content)")
        else:
            log("[steer] OFF: undefended model")
        self.lock = threading.Lock()

    # -- one request
    def build(self, body: dict[str, Any]):
        messages = _normalize_messages(body.get("messages") or [])
        tools = body.get("tools") or None
        tk = body.get("chat_template_kwargs") or {}
        enable_thinking = bool(tk.get("enable_thinking", False))
        text, spans = self.renderer.render_with_spans(messages, tools, enable_thinking)
        ids, span_idx = self.renderer.tokenize(text, spans)
        if len(ids) > self.args.max_model_len:
            raise ValueError(f"prompt has {len(ids)} tokens; max_model_len is {self.args.max_model_len}")
        start_in_thought = text.rstrip(" ").endswith(f"{T_CHANNEL_OPEN}thought\n")
        return text, ids, span_idx, start_in_thought

    def generate(self, body: dict[str, Any], emit, abort: threading.Event) -> dict[str, Any]:
        """Runs in a worker thread; `emit(event)` is called for each parsed event."""
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        text, ids, span_idx, start_in_thought = self.build(body)
        max_new = int(body.get("max_tokens") or self.args.default_max_tokens)
        max_new = max(1, min(max_new, self.args.max_model_len - len(ids)))
        temperature = body.get("temperature")
        temperature = float(temperature) if temperature is not None else 0.0
        top_p = body.get("top_p")
        seed = body.get("seed")
        gen: dict[str, Any] = dict(max_new_tokens=max_new, eos_token_id=self.eos_ids, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=float(top_p) if top_p is not None else 0.95, top_k=64)
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if seed is not None:
            torch.manual_seed(int(seed))

        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(input_ids)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)
        parser = GemmaStreamParser(start_in_thought=start_in_thought)
        stats = {"prompt_tokens": len(ids), "steered_tokens": 0, "completion_tokens": 0}
        with self.lock:
            if self.steer is not None:
                self.steer.positions = span_idx
                self.steer.edited = 0
            t0 = time.time()
            result: dict[str, Any] = {}

            def run():
                try:
                    with torch.inference_mode():
                        out = self.model.generate(input_ids=input_ids, attention_mask=attn, streamer=streamer,
                                                  stopping_criteria=StoppingCriteriaList([AbortCriteria(abort)]), **gen)
                    result["n"] = int(out.shape[1] - input_ids.shape[1])
                    result["last"] = int(out[0, -1])
                except Exception as e:  # noqa: BLE001
                    result["error"] = f"{type(e).__name__}: {e}"
                    streamer.end()

            th = threading.Thread(target=run, daemon=True)
            th.start()
            try:
                for piece in streamer:
                    if piece:
                        for ev in parser.feed(piece):
                            emit(ev)
                    if abort.is_set():
                        break
                for ev in parser.feed("", final=True):
                    emit(ev)
            finally:
                th.join()
                if self.steer is not None:
                    stats["steered_tokens"] = self.steer.edited
                    self.steer.positions = None
            if "error" in result:
                raise RuntimeError(result["error"])
            n = result.get("n", 0)
            stats["completion_tokens"] = n
            dt = time.time() - t0
            finish = "length" if (n >= max_new and result.get("last") not in self.eos_ids and not parser.stopped) else "stop"
            if abort.is_set():
                finish = "stop"
            stats["finish_reason"] = finish
            stats["seconds"] = round(dt, 2)
            log(f"[gen] {self.name}: prompt={len(ids)} steered={stats['steered_tokens']} new={n} "
                f"{n / max(dt, 1e-6):.1f} tok/s finish={finish} thinking={start_in_thought} temp={temperature}")
            return stats


# ════════════════════════════════════════════════════════════ HTTP
def make_app(engine: Engine):
    if FastAPI is None:
        raise SystemExit("fastapi is not installed in this interpreter")
    app = FastAPI(title=f"countersteer server: {engine.name}")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": engine.name, "steer": engine.steer is not None}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": engine.name, "object": "model", "owned_by": "cmu-demos"}]}

    def chunk(delta: dict[str, Any], finish: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        c = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion.chunk", "created": int(time.time()),
             "model": engine.name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            c.update(extra)
        return c

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise HTTPException(400, "messages[] required")
        stream = bool(body.get("stream", True))
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        abort = threading.Event()

        def emit(ev):
            loop.call_soon_threadsafe(q.put_nowait, ev)

        def work():
            try:
                stats = engine.generate(body, emit, abort)
                loop.call_soon_threadsafe(q.put_nowait, ("done", stats))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

        threading.Thread(target=work, daemon=True).start()

        async def events():
            n_calls = 0
            steer_info = ({"alpha": engine.steer.d.alpha, "layers": engine.steer.d.layers, "direction": engine.steer.d.meta.get("direction")}
                          if engine.steer else None)
            try:
                yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': ''}))}\n\n"
                while True:
                    kind, payload = await q.get()
                    if kind == "content":
                        yield f"data: {json.dumps(chunk({'content': payload}))}\n\n"
                    elif kind == "reasoning":
                        yield f"data: {json.dumps(chunk({'reasoning_content': payload}))}\n\n"
                    elif kind == "tool_call":
                        tc = {"index": n_calls, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                              "function": {"name": payload["name"], "arguments": json.dumps(payload["arguments"], ensure_ascii=False)}}
                        n_calls += 1
                        yield f"data: {json.dumps(chunk({'tool_calls': [tc]}))}\n\n"
                    elif kind == "stop":
                        continue
                    elif kind == "error":
                        yield f"data: {json.dumps({'error': {'message': payload, 'type': 'server_error'}})}\n\n"
                        break
                    elif kind == "done":
                        finish = payload.get("finish_reason", "stop")
                        if n_calls and finish == "stop":
                            finish = "tool_calls"
                        extra = {"usage": {"prompt_tokens": payload["prompt_tokens"], "completion_tokens": payload["completion_tokens"],
                                           "total_tokens": payload["prompt_tokens"] + payload["completion_tokens"]},
                                 "countersteer": {"steered_tokens": payload["steered_tokens"], "seconds": payload["seconds"], **(steer_info or {"enabled": False})}}
                        yield f"data: {json.dumps(chunk({}, finish, extra))}\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                abort.set()

        if stream:
            return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # non-streaming: collect
        content, reasoning, calls, stats, err = "", "", [], None, None
        while True:
            kind, payload = await q.get()
            if kind == "content":
                content += payload
            elif kind == "reasoning":
                reasoning += payload
            elif kind == "tool_call":
                calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                              "function": {"name": payload["name"], "arguments": json.dumps(payload["arguments"], ensure_ascii=False)}})
            elif kind == "error":
                err = payload
                break
            elif kind == "done":
                stats = payload
                break
        if err:
            return JSONResponse({"error": {"message": err, "type": "server_error"}}, status_code=500)
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        finish = stats.get("finish_reason", "stop")
        if calls and finish == "stop":
            finish = "tool_calls"
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()), "model": engine.name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
                          "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]},
                "countersteer": {"steered_tokens": stats["steered_tokens"], "seconds": stats["seconds"],
                                 **({"alpha": engine.steer.d.alpha, "layers": engine.steer.d.layers} if engine.steer else {"enabled": False})}}

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="HF repo id or local checkpoint directory")
    p.add_argument("--served-model-name", default="gemma")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--directions", default=None, help="npz with dir_L{n} unit vectors + meta (sigmas, alpha); omit for the undefended model")
    p.add_argument("--alpha", type=float, default=None, help="override the npz alpha")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn", default=None, help="attn_implementation override (default: transformers' resolution)")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--default-max-tokens", type=int, default=1024)
    p.add_argument("--self-test", action="store_true", help="run the parser unit checks and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.model:
        p.error("--model is required")

    engine = Engine(args)
    app = make_app(engine)
    import uvicorn
    log(f"[http] serving {engine.name} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def self_test() -> int:
    sample = ('<|channel>thought\n<channel|><|tool_call>call:save_summary{source_url:<|"|>https://x0.at/a.html<|"|>,'
              'summary:<|"|>A, b; {c} d<|"|>,title:<|"|>T - W<|"|>}<tool_call|><|tool_call>call:send_email{body:<|"|>hi<|"|>,n:3,'
              'flags:[true,false],o:{k:<|"|>v<|"|>}}<tool_call|><|tool_response>')
    # feed in awkward pieces
    parser = GemmaStreamParser()
    events = []
    for i in range(0, len(sample), 7):
        events += parser.feed(sample[i:i + 7])
    events += parser.feed("", final=True)
    calls = [e[1] for e in events if e[0] == "tool_call"]
    assert [c["name"] for c in calls] == ["save_summary", "send_email"], calls
    assert calls[0]["arguments"] == {"source_url": "https://x0.at/a.html", "summary": "A, b; {c} d", "title": "T - W"}, calls[0]
    assert calls[1]["arguments"] == {"body": "hi", "n": 3, "flags": [True, False], "o": {"k": "v"}}, calls[1]
    assert not any(e[0] == "content" for e in events), events
    assert parser.stopped
    p2 = GemmaStreamParser()
    ev = p2.feed("Hello <|channel>thought\nthinking…<channel|> world<turn|>", final=True)
    assert ("content", "Hello ") in ev and ("reasoning", "thinking…") in ev and ("content", " world") in ev, ev
    p3 = GemmaStreamParser(start_in_thought=True)
    ev = p3.feed("deep thought<channel|>answer<eos>", final=True)
    assert ev[0] == ("reasoning", "deep thought") and ev[1] == ("content", "answer"), ev
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
