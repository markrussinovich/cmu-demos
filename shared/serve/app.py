"""FastAPI app factory: serves the built frontend and proxies chat to the vLLM servers.

Routes
  GET  /api/config            demo info, defaults and the public model list
  GET  /api/status            per-model state: ready | starting | down
  POST /api/chat/{model_id}   OpenAI-style chat request, streamed back as SSE
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .config import Config, ModelSpec, load_config
from .launcher import model_status


def create_app(config_path: str | Path, static_dir: str | Path | None = None, logs_dir: str | Path | None = None) -> FastAPI:
    cfg: Config = load_config(config_path)
    logs = Path(logs_dir) if logs_dir else cfg.path.parent / "logs"

    def get_client(app: FastAPI) -> httpx.AsyncClient:
        client = getattr(app.state, "client", None)
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
            app.state.client = client
        return client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        get_client(app)
        try:
            yield
        finally:
            await get_client(app).aclose()

    app = FastAPI(title=cfg.demo.title, lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return cfg.public()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        results = await asyncio.gather(*(loop.run_in_executor(None, model_status, logs, spec) for spec in cfg.models))
        return {"models": {spec.id: st for spec, st in zip(cfg.models, results)}}

    @app.post("/api/chat/{model_id}")
    async def chat(model_id: str, request: Request) -> Response:
        try:
            spec: ModelSpec = cfg.model(model_id)
        except KeyError:
            raise HTTPException(404, f"unknown model '{model_id}'")
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages[] required")

        payload: dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "stream": True,
        }
        for key in ("temperature", "top_p", "max_tokens", "seed", "repetition_penalty", "presence_penalty", "frequency_penalty"):
            value = body.get(key, cfg.defaults.get(key))
            if value is not None:
                payload[key] = value
        payload.update(spec.request_extra)
        template_kwargs: dict[str, Any] = dict(spec.chat_template_kwargs)
        if spec.supports_thinking:
            enable = body.get("enable_thinking", cfg.defaults.get("enable_thinking", False))
            template_kwargs["enable_thinking"] = bool(enable)
        if spec.supports_reasoning_effort:
            effort = body.get("reasoning_effort", cfg.defaults.get("reasoning_effort"))
            if effort in ("low", "medium", "high"):
                payload["reasoning_effort"] = effort
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs

        client = get_client(request.app)
        upstream = client.build_request("POST", f"{spec.base_url}/v1/chat/completions", json=payload)
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"{spec.label} is not reachable at {spec.base_url}: {exc}")
        if resp.status_code != 200:
            detail = (await resp.aread()).decode("utf-8", errors="replace")
            await resp.aclose()
            try:
                detail_json = json.loads(detail)
            except ValueError:
                detail_json = {"error": detail}
            return JSONResponse(detail_json, status_code=resp.status_code)
        return StreamingResponse(
            resp.aiter_raw(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(resp.aclose),
        )

    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
    else:
        @app.get("/")
        async def no_frontend() -> JSONResponse:
            return JSONResponse({"error": f"frontend not built; expected {static_dir}. Run `npm run build` in the frontend."}, status_code=503)

    return app
