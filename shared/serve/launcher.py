"""Start / stop / wait for the vLLM servers declared in a demo's models.yaml.

Usage (PYTHONPATH must include the shared/ directory):
    python -m serve.launcher start  --config grp-oblit/models.yaml --logs grp-oblit/logs [--wait]
    python -m serve.launcher wait   --config ... --logs ...
    python -m serve.launcher status --config ... --logs ...
    python -m serve.launcher stop   --config ... --logs ...
    python -m serve.launcher cmd    --config ...        # print the vllm commands
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config, ModelSpec, load_config

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def default_vllm_bin() -> str:
    candidate = Path(sys.executable).with_name("vllm")
    return str(candidate) if candidate.exists() else "vllm"


def vllm_command(spec: ModelSpec) -> list[str]:
    cmd = [
        spec.vllm_bin or default_vllm_bin(),
        "serve",
        spec.model,
        "--served-model-name", spec.id,
        "--host", spec.host,
        "--port", str(spec.port),
        "--tensor-parallel-size", str(spec.tensor_parallel),
        "--dtype", spec.dtype,
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
    ]
    if spec.max_model_len:
        cmd += ["--max-model-len", str(spec.max_model_len)]
    if spec.reasoning_parser:
        cmd += ["--reasoning-parser", spec.reasoning_parser]
    cmd += list(spec.extra_args)
    return cmd


def pid_file(logs: Path, spec: ModelSpec) -> Path:
    return logs / f"{spec.id}.pid"


def log_file(logs: Path, spec: ModelSpec) -> Path:
    return logs / f"{spec.id}.log"


def read_pid(logs: Path, spec: ModelSpec) -> int | None:
    try:
        return int(pid_file(logs, spec).read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); check /proc state
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        return True


def healthy(spec: ModelSpec, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{spec.base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def last_log_line(logs: Path, spec: ModelSpec, max_len: int = 160) -> str:
    path = log_file(logs, spec)
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    for line in reversed(tail.splitlines()):
        line = _ANSI.sub("", line).strip()
        if line:
            return line[-max_len:]
    return ""


def model_status(logs: Path, spec: ModelSpec) -> dict:
    if healthy(spec):
        return {"state": "ready", "detail": f"serving on :{spec.port}"}
    pid = read_pid(logs, spec)
    if pid_alive(pid):
        return {"state": "starting", "detail": f"pid {pid}", "last_log": last_log_line(logs, spec)}
    if pid:
        return {"state": "down", "detail": f"process {pid} exited", "last_log": last_log_line(logs, spec)}
    return {"state": "down", "detail": "not started"}


def start(cfg: Config, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    for spec in cfg.models:
        if healthy(spec):
            print(f"[{spec.id}] already serving on {spec.base_url}")
            continue
        pid = read_pid(logs, spec)
        if pid_alive(pid):
            print(f"[{spec.id}] already starting (pid {pid})")
            continue
        cmd = vllm_command(spec)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in spec.gpus)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        out = log_file(logs, spec).open("ab")
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launching: {shlex.join(cmd)}\n".encode())
        out.write(f"===== CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n".encode())
        out.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file(logs, spec).write_text(str(proc.pid))
        print(f"[{spec.id}] started pid {proc.pid} on GPUs {spec.gpus} -> {spec.base_url}  (log: {log_file(logs, spec)})")


def stop(cfg: Config, logs: Path, grace: float = 30.0) -> None:
    for spec in cfg.models:
        pid = read_pid(logs, spec)
        if not pid_alive(pid):
            pid_file(logs, spec).unlink(missing_ok=True)
            continue
        assert pid is not None
        print(f"[{spec.id}] stopping pid {pid} ...", end="", flush=True)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + grace
        while pid_alive(pid) and time.time() < deadline:
            time.sleep(0.5)
        if pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            print(" killed")
        else:
            print(" stopped")
        pid_file(logs, spec).unlink(missing_ok=True)


def wait_ready(cfg: Config, logs: Path, timeout: float = 1800.0, interval: float = 5.0) -> bool:
    deadline = time.time() + timeout
    pending = {m.id for m in cfg.models}
    last_print: dict[str, str] = {}
    while pending and time.time() < deadline:
        for spec in cfg.models:
            if spec.id not in pending:
                continue
            st = model_status(logs, spec)
            if st["state"] == "ready":
                print(f"[{spec.id}] READY on {spec.base_url}")
                pending.discard(spec.id)
            elif st["state"] == "down":
                print(f"[{spec.id}] FAILED: {st.get('detail')} -- see {log_file(logs, spec)}")
                print(f"[{spec.id}] last log line: {st.get('last_log')}")
                return False
            else:
                line = st.get("last_log", "")
                if line and last_print.get(spec.id) != line:
                    print(f"[{spec.id}] {line}")
                    last_print[spec.id] = line
        if pending:
            time.sleep(interval)
    if pending:
        print(f"timed out waiting for: {sorted(pending)}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["start", "stop", "status", "wait", "cmd"])
    p.add_argument("--config", required=True)
    p.add_argument("--logs", default=None, help="directory for pid + log files (default: <config dir>/logs)")
    p.add_argument("--wait", action="store_true", help="with start: block until every model is healthy")
    p.add_argument("--timeout", type=float, default=1800.0)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    logs = Path(args.logs) if args.logs else cfg.path.parent / "logs"

    if args.command == "cmd":
        for spec in cfg.models:
            print(f"CUDA_VISIBLE_DEVICES={','.join(map(str, spec.gpus))} {shlex.join(vllm_command(spec))}")
        return 0
    if args.command == "start":
        start(cfg, logs)
        if args.wait:
            return 0 if wait_ready(cfg, logs, args.timeout) else 1
        return 0
    if args.command == "stop":
        stop(cfg, logs)
        return 0
    if args.command == "wait":
        return 0 if wait_ready(cfg, logs, args.timeout) else 1
    if args.command == "status":
        print(json.dumps({spec.id: model_status(logs, spec) for spec in cfg.models}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
