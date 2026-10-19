"""GRPO-obliteration demo backend.

Serves the built frontend from ../frontend/dist and proxies /api/chat/<model>
to the vLLM servers declared in ../models.yaml. Run via ../run.sh, or directly:

    PYTHONPATH=../../shared python app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT / "shared"))

from serve.app import create_app  # noqa: E402
from serve.config import load_config  # noqa: E402

CONFIG = Path(os.environ.get("CMU_DEMO_CONFIG", DEMO / "models.yaml"))
LOGS = Path(os.environ.get("CMU_DEMO_LOGS", DEMO / "logs"))
STATIC = Path(os.environ.get("CMU_DEMO_STATIC", DEMO / "frontend" / "dist"))

app = create_app(CONFIG, static_dir=STATIC, logs_dir=LOGS)

if __name__ == "__main__":
    import uvicorn

    server = load_config(CONFIG).server
    uvicorn.run(app, host=server.host, port=server.port, log_level="info")
