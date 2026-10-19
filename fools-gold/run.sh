#!/usr/bin/env bash
# Launch this demo: its vLLM servers + the web app. Generic: driven by ./models.yaml.
#
#   ./run.sh                 start models (if not already running), wait, serve UI
#   NO_MODELS=1 ./run.sh     only the web app (models already up or started elsewhere)
#   REBUILD=1 ./run.sh       force a frontend rebuild
#   ./run.sh stop            stop the vLLM servers started by this script
#   ./run.sh status          print per-model status
#
# Ctrl-C stops the web app; set STOP_MODELS_ON_EXIT=1 to also tear down vLLM.
set -euo pipefail

DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DEMO/.." && pwd)"
NAME="$(basename "$DEMO")"
VENV="${CMU_DEMOS_VENV:-/datadrive2/venvs/cmu-demos}"
PY="$VENV/bin/python"
CONFIG="${CONFIG:-$DEMO/models.yaml}"
LOGS="${LOGS:-$DEMO/logs}"
PORT="${PORT:-$($PY - "$CONFIG" <<'EOF'
import sys, yaml; print((yaml.safe_load(open(sys.argv[1])).get("server") or {}).get("port", 8080))
EOF
)}"

export HF_HOME="${HF_HOME:-/datadrive2/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ROOT/shared${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOGS"

[ -x "$PY" ] || { echo "venv not found at $VENV (set CMU_DEMOS_VENV)"; exit 1; }

case "${1:-}" in
  stop)   exec "$PY" -m serve.launcher stop   --config "$CONFIG" --logs "$LOGS" ;;
  status) exec "$PY" -m serve.launcher status --config "$CONFIG" --logs "$LOGS" ;;
  "") ;;
  *) echo "usage: $0 [stop|status]"; exit 2 ;;
esac

if [ "${REBUILD:-0}" = 1 ] || [ ! -f "$DEMO/frontend/dist/index.html" ]; then
  echo ">> building frontend"
  (cd "$ROOT" && npm run build --workspace "$NAME/frontend")
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [ "${STOP_MODELS_ON_EXIT:-0}" = 1 ]; then
    "$PY" -m serve.launcher stop --config "$CONFIG" --logs "$LOGS" || true
  fi
}
trap cleanup EXIT INT TERM

if [ "${NO_MODELS:-0}" != 1 ]; then
  echo ">> starting vLLM servers (logs in $LOGS)"
  "$PY" -m serve.launcher start --config "$CONFIG" --logs "$LOGS"
fi

if curl -fs "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo ">> web app already running on :$PORT (stop it first, or set PORT=... to start another)"
  BACKEND_PID=""
else
  echo ">> starting web app on http://0.0.0.0:$PORT"
  "$PY" "$DEMO/backend/app.py" >"$LOGS/backend.log" 2>&1 &
  BACKEND_PID=$!
  sleep 1
  kill -0 "$BACKEND_PID" 2>/dev/null || { echo "backend failed to start; see $LOGS/backend.log"; tail -20 "$LOGS/backend.log"; exit 1; }
fi

echo
echo "   Open  http://localhost:$PORT   (status bar shows each model loading; takes a few minutes)"
echo "   Logs  $LOGS/<model-id>.log and $LOGS/backend.log"
echo

if [ "${NO_MODELS:-0}" != 1 ]; then
  "$PY" -m serve.launcher wait --config "$CONFIG" --logs "$LOGS" || echo "!! a model failed to start; the UI will show it as down"
fi

if [ -n "$BACKEND_PID" ]; then wait "$BACKEND_PID"; fi
