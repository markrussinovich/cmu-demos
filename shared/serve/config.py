"""Load a demo's models.yaml into typed config objects.

Every demo ships a models.yaml describing the demo (name/title), the HTTP
server, default sampling params and the list of models to serve with vLLM.
"""
from __future__ import annotations

import dataclasses
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
    presets: list[dict[str, Any]] = field(default_factory=list)
    span_marker: dict[str, Any] | None = None


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
