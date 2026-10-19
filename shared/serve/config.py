"""Load a demo's models.yaml into typed config objects.

Every demo ships a models.yaml describing the demo (name/title), the HTTP
server, default sampling params and the list of models to serve with vLLM.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    id: str
    label: str
    model: str  # HF repo id or absolute local path
    gpus: list[int] = field(default_factory=lambda: [0])
    tensor_parallel: int = 1
    host: str = "127.0.0.1"
    port: int = 8001
    max_model_len: int | None = 16384
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"
    reasoning_parser: str | None = None
    supports_thinking: bool = False  # chat template accepts enable_thinking (Gemma/Qwen style)
    supports_reasoning_effort: bool = False  # accepts OpenAI-style reasoning_effort (gpt-oss)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # always sent with each request
    request_extra: dict[str, Any] = field(default_factory=dict)  # merged into every upstream request body
    extra_args: list[str] = field(default_factory=list)
    vllm_bin: str | None = None
    # Custom server instead of `vllm serve`: argv with {model} {id} {host} {port} {python}
    # {dir} placeholders (`dir` = the directory the command runs in: the demo dir locally,
    # `remote_dir` remotely). `python` is the interpreter substituted for {python}.
    command: list[str] | None = None
    python: str | None = None
    env: dict[str, str] = field(default_factory=dict)  # extra environment for the server process
    # Run the server on another machine over ssh (host alias from ~/.ssh/config). The
    # launcher rsyncs `sync` (paths relative to the demo dir) into `remote_dir`, starts the
    # command there, and forwards 127.0.0.1:{port} through the same ssh session, so health
    # checks, logs, pid handling and the chat proxy all work unchanged.
    remote: str | None = None
    remote_dir: str | None = None
    sync: list[str] = field(default_factory=list)
    accent: str | None = None
    note: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "model": self.model,
            "gpus": self.gpus,
            "tensor_parallel": self.tensor_parallel,
            "port": self.port,
            "accent": self.accent,
            "note": self.note,
            "remote": self.remote,
            "supports_thinking": self.supports_thinking,
            "supports_reasoning_effort": self.supports_reasoning_effort,
        }


@dataclass
class DemoInfo:
    name: str
    title: str
    subtitle: str = ""
    # Optional composer helpers, all rendered by the shared UI when present:
    #   presets:     [{label, text}] example prompts offered in a dropdown
    #   span_marker: {open, close, label?, button?} delimiters the composer can wrap a
    #                selection with; wrapped spans are highlighted in sent messages
    #   composer:    "chat" (one prompt box) or "scenario" (user request + untrusted tool
    #                result; presets carry a `scenario` object, see shared/chat-ui types)
    #   presets_file: prompts file (relative to models.yaml) whose list is appended to
    #                presets. Defaults to `prompts.yaml` in the demo directory when present;
    #                that file is where every demo keeps its example prompts.
    presets: list[dict[str, Any]] = field(default_factory=list)
    span_marker: dict[str, Any] | None = None
    composer: str = "chat"
    presets_file: str | None = None


@dataclass
class ServerInfo:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class Config:
    path: Path
    demo: DemoInfo
    server: ServerInfo
    defaults: dict[str, Any]
    models: list[ModelSpec]

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def public(self) -> dict[str, Any]:
        return {
            "demo": dataclasses.asdict(self.demo),
            "defaults": self.defaults,
            "models": [m.public() for m in self.models],
        }


PROMPTS_FILE = "prompts.yaml"


def load_presets_file(pf: Path) -> list[dict[str, Any]]:
    """A list of presets from a YAML or JSON file (`[{label, text, note?, scenario?}, ...]`)."""
    if not pf.exists():
        raise FileNotFoundError(f"prompts file not found: {pf}")
    text = pf.read_text()
    data = json.loads(text) if pf.suffix.lower() == ".json" else yaml.safe_load(text)
    if data is None:
        return []
    if isinstance(data, dict) and isinstance(data.get("presets"), list):
        data = data["presets"]
    if not isinstance(data, list):
        raise ValueError(f"{pf}: expected a list of presets (or a mapping with a `presets` list)")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "label" not in item or "text" not in item:
            raise ValueError(f"{pf}: preset #{i} needs at least `label` and `text`")
    return data


def _pick(cls, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return data


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    demo = DemoInfo(**_pick(DemoInfo, raw.get("demo", {})))
    # Prompts live in a top-level file in the demo directory: `prompts.yaml` by convention
    # (picked up automatically), or whatever `presets_file` names (.yaml/.yml/.json).
    pf = (path.parent / demo.presets_file).resolve() if demo.presets_file else path.parent / PROMPTS_FILE
    if demo.presets_file or pf.exists():
        extra = load_presets_file(pf)
        demo.presets = list(demo.presets) + extra
        demo.presets_file = pf.name if pf.parent == path.parent else str(pf)
    if demo.composer not in ("chat", "scenario"):
        raise ValueError(f"{path}: demo.composer must be 'chat' or 'scenario'")
    server = ServerInfo(**_pick(ServerInfo, raw.get("server", {})))
    defaults = dict(raw.get("defaults", {}))
    models = [ModelSpec(**_pick(ModelSpec, m)) for m in raw.get("models", [])]
    if not models:
        raise ValueError(f"{path}: no models defined")
    ids = [m.id for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate model ids {ids}")
    ports = [m.port for m in models]
    if len(ports) != len(set(ports)):
        raise ValueError(f"{path}: duplicate ports {ports}")
    return Config(path=path, demo=demo, server=server, defaults=defaults, models=models)
