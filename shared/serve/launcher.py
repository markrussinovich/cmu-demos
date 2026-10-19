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


def serve_command(spec: ModelSpec, run_dir: Path) -> list[str]:
    """The server's own argv: vLLM by default, or `spec.command` with placeholders filled."""
    if not spec.command:
        return vllm_command(spec)
    fields = {"model": spec.model, "id": spec.id, "host": spec.host, "port": str(spec.port),
              "python": spec.python or sys.executable, "dir": str(run_dir)}
    return [arg.format(**fields) for arg in spec.command]


def server_env(spec: ModelSpec) -> dict[str, str]:
    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in spec.gpus)}
    env.update({k: str(v) for k, v in spec.env.items()})
    return env


def _remote_wrapper(spec: ModelSpec, cmd: list[str], env: dict[str, str]) -> str:
    """Shell snippet run on the remote host by one ssh session that also forwards the port.

    The server runs in its own session with stdout/stderr streamed back through ssh (so the
    local log file is the server's log). A sidecar waits for ssh's stdin to close -- which is
    what happens when the local launcher stops or dies -- and then kills the server's process
    group, so a remote server never outlives the local pid that represents it.
    """
    assert spec.remote_dir, f"[{spec.id}] remote requires remote_dir"
    env_s = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return (
        f"cd {shlex.quote(spec.remote_dir)} || exit 97; "
        f"exec 3<&0; "
        f"setsid env {env_s} {shlex.join(cmd)} 2>&1 & P=$!; "
        f"( cat <&3 >/dev/null; kill -TERM -- -$P 2>/dev/null || kill -TERM $P 2>/dev/null ) & W=$!; "
        f"wait $P; RC=$?; kill $W 2>/dev/null; kill -TERM -- -$P 2>/dev/null; exit $RC"
    )


def launch_command(spec: ModelSpec, run_dir: Path) -> list[str]:
    """What the launcher actually execs locally (an ssh session for remote specs)."""
    cmd = serve_command(spec, Path(spec.remote_dir) if spec.remote else run_dir)
    if not spec.remote:
        return cmd
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-o", "ExitOnForwardFailure=yes", "-o", "LogLevel=ERROR", "-T",
        "-L", f"127.0.0.1:{spec.port}:{spec.host}:{spec.port}",
        spec.remote, _remote_wrapper(spec, cmd, server_env(spec)),
    ]


def sync_remote(spec: ModelSpec, demo_dir: Path) -> None:
    """rsync `spec.sync` (paths relative to the demo dir) into spec.remote_dir on the remote."""
    if not spec.remote or not spec.sync:
        return
    assert spec.remote_dir, f"[{spec.id}] remote requires remote_dir"
    subprocess.run(["ssh", "-o", "BatchMode=yes", spec.remote, f"mkdir -p {shlex.quote(spec.remote_dir)}"], check=True)
    for rel in spec.sync:
        src = demo_dir / rel
        if not src.exists():
            raise SystemExit(f"[{spec.id}] sync path {src} does not exist")
        dest = f"{spec.remote}:{spec.remote_dir.rstrip('/')}/{rel.rstrip('/')}"
        # trailing slash on a directory source copies its contents into `dest`
        src_arg = f"{src}/" if src.is_dir() else str(src)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", "-e", "ssh -o BatchMode=yes", src_arg, dest], check=True)
        print(f"[{spec.id}] synced {rel} -> {dest}")


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
        line = _ANSI.sub("", line).replace("\r", "").strip()
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
    demo_dir = cfg.path.parent
    for spec in cfg.models:
        if healthy(spec):
            print(f"[{spec.id}] already serving on {spec.base_url}")
            continue
        pid = read_pid(logs, spec)
        if pid_alive(pid):
            print(f"[{spec.id}] already starting (pid {pid})")
            continue
        sync_remote(spec, demo_dir)
        cmd = launch_command(spec, demo_dir)
        env = dict(os.environ)
        env.update(server_env(spec))
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        out = log_file(logs, spec).open("ab")
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launching: {shlex.join(cmd)}\n".encode())
        out.write(f"===== CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}{' on ' + spec.remote if spec.remote else ''}\n".encode())
        out.flush()
        if spec.remote:
            # ssh's stdin must stay open for exactly as long as ssh lives: the remote sidecar
            # kills the server when it sees EOF there. ssh itself closes every inherited fd
            # above stderr at startup, so it cannot hold the pipe's write end; a tiny local
            # keeper process holds it instead and exits as soon as the ssh pid is gone.
            r, w = os.pipe()
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=r,
                                    start_new_session=True, cwd=str(demo_dir))
            subprocess.Popen(["sh", "-c", 'while kill -0 "$1" 2>/dev/null; do sleep 2; done', "ssh-keeper", str(proc.pid)],
                             pass_fds=(w,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            os.close(r)
            os.close(w)
        else:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(demo_dir))
        pid_file(logs, spec).write_text(str(proc.pid))
        where = f"{spec.remote} GPUs {spec.gpus} (ssh tunnel)" if spec.remote else f"GPUs {spec.gpus}"
        print(f"[{spec.id}] started pid {proc.pid} on {where} -> {spec.base_url}  (log: {log_file(logs, spec)})")


def remote_server_pattern(spec: ModelSpec) -> str:
    """pgrep -f pattern identifying this spec's server process on the remote host. The
    bracketed first character keeps the pattern from matching the shell that carries it."""
    return rf"[s]erver\.py .*--served-model-name {re.escape(spec.id)} .*--port {spec.port}\b"


def stop_remote_orphans(spec: ModelSpec) -> None:
    """Belt and braces for remote specs: kill any leftover server matching this spec on the
    remote host, whether or not its ssh session is still around."""
    if not spec.remote:
        return
    script = (f"for p in $(pgrep -f {shlex.quote(remote_server_pattern(spec))}); do "
              f"pg=$(ps -o pgid= -p $p | tr -d ' '); [ -n \"$pg\" ] && kill -TERM -- -$pg 2>/dev/null || kill -TERM $p 2>/dev/null; "
              f"echo \"[{spec.id}] killed remote pid $p\"; done; true")
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", spec.remote, script],
                             capture_output=True, text=True, timeout=30)
        if out.stdout.strip():
            print(out.stdout.strip())
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[{spec.id}] could not check {spec.remote} for leftover servers: {exc}")


def stop(cfg: Config, logs: Path, grace: float = 30.0) -> None:
    for spec in cfg.models:
        pid = read_pid(logs, spec)
        if not pid_alive(pid):
            pid_file(logs, spec).unlink(missing_ok=True)
            stop_remote_orphans(spec)
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
        stop_remote_orphans(spec)


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
            env = " ".join(f"{k}={v}" for k, v in server_env(spec).items())
            print(f"{env} {shlex.join(serve_command(spec, cfg.path.parent))}" + (f"   # on {spec.remote} in {spec.remote_dir}" if spec.remote else ""))
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
