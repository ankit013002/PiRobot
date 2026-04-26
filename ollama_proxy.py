#!/usr/bin/env python3
"""
ollama_proxy.py — Logging proxy between the Pi robot and Ollama.

Runs on your Windows/Mac/Linux machine alongside `ollama serve`.
Forwards every request to the real Ollama server, logs prompts and
responses with timestamps and latency.

The Pi should point OLLAMA_URL at this proxy:
    OLLAMA_URL=http://<your-pc-ip>:11435 python autonomous3.py

Usage:
    python ollama_proxy.py                          # defaults
    python ollama_proxy.py --port 11435 --ollama http://localhost:11434
    python ollama_proxy.py --start-ollama           # also launch `ollama serve`
    python ollama_proxy.py --log-file ollama_log.jsonl  # structured JSON log
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("proxy")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _wrap(text: str, indent: int = 4, width: int = 100) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=prefix, subsequent_indent=prefix)
        if len(line) > width else prefix + line
        for line in text.splitlines()
    ) or prefix + "(empty)"


# ---------------------------------------------------------------------------
# JSON log writer (optional)
# ---------------------------------------------------------------------------
class JsonlWriter:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        log.info("JSON log → %s", path)

    def write(self, record: dict):
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.total = 0
        self.errors = 0
        self.total_latency = 0.0
        self.by_model: dict[str, int] = {}
        self._start = time.monotonic()

    def record(self, model: str, latency: float, ok: bool):
        with self._lock:
            self.total += 1
            self.total_latency += latency
            if not ok:
                self.errors += 1
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def summary(self) -> str:
        with self._lock:
            uptime = time.monotonic() - self._start
            avg = self.total_latency / self.total if self.total else 0.0
            models = ", ".join(f"{m}:{n}" for m, n in sorted(self.by_model.items()))
            return (
                f"requests={self.total}  errors={self.errors}  "
                f"avg_latency={avg:.2f}s  models=[{models}]  "
                f"uptime={uptime:.0f}s"
            )


stats = Stats()


# ---------------------------------------------------------------------------
# HTTP proxy handler
# ---------------------------------------------------------------------------
class ProxyHandler(BaseHTTPRequestHandler):
    ollama_base: str = "http://localhost:11434"
    jsonl: "JsonlWriter | None" = None

    # suppress default "GET /path HTTP/1.1" access log; we do our own
    def log_message(self, fmt, *args):  # type: ignore[override]
        pass

    def log_error(self, fmt, *args):  # type: ignore[override]
        log.error(fmt, *args)

    # ------------------------------------------------------------------
    def do_GET(self):
        self._proxy(method="GET", body=None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._proxy(method="POST", body=body)

    def do_DELETE(self):
        self._proxy(method="DELETE", body=None)

    def do_HEAD(self):
        self._proxy(method="HEAD", body=None)

    # ------------------------------------------------------------------
    def _proxy(self, method: str, body: bytes | None):
        target_url = self.ollama_base + self.path
        client_ip = self.client_address[0]

        # Build forwarded request
        fwd_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        req = Request(target_url, data=body, headers=fwd_headers, method=method)

        t0 = time.monotonic()
        is_generate = self.path.rstrip("/") in ("/api/generate", "/api/chat")
        parsed_req: dict = {}

        if is_generate and body:
            try:
                parsed_req = json.loads(body)
            except Exception:
                pass

        try:
            with urlopen(req, timeout=60) as resp:
                status = resp.status
                resp_headers = dict(resp.headers)
                resp_body = resp.read()
        except URLError as exc:
            latency = time.monotonic() - t0
            log.error("%-15s  %-30s  ERROR  %.2fs  %s", client_ip, self.path, latency, exc)
            stats.record(parsed_req.get("model", "?"), latency, ok=False)
            self.send_error(502, str(exc))
            return

        latency = time.monotonic() - t0

        # --- send response back to client ---
        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in ("transfer-encoding",):
                continue
            try:
                self.send_header(k, v)
            except Exception:
                pass
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

        # --- log ---
        model = parsed_req.get("model", "")
        stats.record(model or "?", latency, ok=(200 <= status < 300))

        if is_generate:
            self._log_generate(client_ip, parsed_req, resp_body, latency, status)
        else:
            log.info("%-15s  %-35s  %d  %.2fs", client_ip, self.path, status, latency)

    # ------------------------------------------------------------------
    def _log_generate(
        self,
        client_ip: str,
        req_body: dict,
        resp_bytes: bytes,
        latency: float,
        status: int,
    ):
        model = req_body.get("model", "?")
        prompt = req_body.get("prompt", req_body.get("messages", ""))
        if isinstance(prompt, list):  # /api/chat messages array
            prompt = "\n".join(
                f"[{m.get('role','?')}] {m.get('content','')}" for m in prompt
            )

        resp_text = ""
        resp_parsed: dict = {}
        try:
            resp_parsed = json.loads(resp_bytes)
            resp_text = (
                resp_parsed.get("response")
                or resp_parsed.get("message", {}).get("content", "")
                or ""
            )
        except Exception:
            resp_text = resp_bytes.decode(errors="replace")[:500]

        # Try to parse the robot's JSON payload inside the LLM response
        action_json: dict = {}
        try:
            # The robot expects a JSON block in the response
            import re
            m = re.search(r"\{[^{}]+\}", resp_text, re.DOTALL)
            if m:
                action_json = json.loads(m.group())
        except Exception:
            pass

        # Pretty console output
        sep = "─" * 80
        lines = [
            "",
            sep,
            f"  {_now_str()}  ·  {client_ip}  →  {model}  [{latency:.2f}s]  HTTP {status}",
            sep,
            "  PROMPT:",
            _wrap(str(prompt).strip()),
        ]
        if action_json:
            lines += ["  RESPONSE (parsed):", _wrap(json.dumps(action_json, indent=2))]
        elif resp_text.strip():
            lines += ["  RESPONSE (raw):", _wrap(resp_text.strip()[:600])]
        lines.append(sep)
        print("\n".join(lines), flush=True)

        # JSONL structured log
        if self.jsonl:
            record = {
                "ts": _now_str(),
                "client": client_ip,
                "model": model,
                "latency_s": round(latency, 3),
                "status": status,
                "prompt": str(prompt).strip(),
                "response_raw": resp_text.strip()[:2000],
                "action": action_json,
            }
            self.jsonl.write(record)


# ---------------------------------------------------------------------------
# Stats printer thread
# ---------------------------------------------------------------------------
def _stats_printer(interval: int = 60):
    while True:
        time.sleep(interval)
        log.info("[STATS] %s", stats.summary())


# ---------------------------------------------------------------------------
# Optional: launch `ollama serve` as a subprocess
# ---------------------------------------------------------------------------
def _start_ollama(ollama_cmd: str = "ollama") -> subprocess.Popen | None:
    log.info("Starting `ollama serve` ...")
    try:
        proc = subprocess.Popen(
            [ollama_cmd, "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Give it a moment to bind
        time.sleep(2.0)
        if proc.poll() is not None:
            log.error("`ollama serve` exited immediately (rc=%d)", proc.returncode)
            return None
        log.info("`ollama serve` started (pid=%d)", proc.pid)

        # Pipe its output to our logger
        def _pipe():
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    log.info("[ollama] %s", line)

        threading.Thread(target=_pipe, daemon=True, name="ollama-pipe").start()
        return proc
    except FileNotFoundError:
        log.error("`ollama` not found in PATH. Install it or pass --ollama-cmd.")
        return None


# ---------------------------------------------------------------------------
# Wait for Ollama to be ready
# ---------------------------------------------------------------------------
def _wait_for_ollama(base: str, retries: int = 15, delay: float = 1.0) -> bool:
    url = base.rstrip("/") + "/api/tags"
    for i in range(retries):
        try:
            urlopen(url, timeout=3)
            log.info("Ollama is ready at %s", base)
            return True
        except Exception:
            if i == 0:
                log.info("Waiting for Ollama at %s ...", base)
            time.sleep(delay)
    log.error("Ollama did not respond after %d retries", retries)
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Logging HTTP proxy between the Pi robot and Ollama."
    )
    parser.add_argument(
        "--port", type=int, default=11435,
        help="Port this proxy listens on (default: 11435)",
    )
    parser.add_argument(
        "--ollama", default="http://localhost:11434",
        help="Real Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--start-ollama", action="store_true",
        help="Launch `ollama serve` automatically before starting the proxy",
    )
    parser.add_argument(
        "--ollama-cmd", default="ollama",
        help="Path to the ollama executable (default: ollama)",
    )
    parser.add_argument(
        "--log-file", default="",
        help="Optional .jsonl file to write structured logs (e.g. ollama_log.jsonl)",
    )
    parser.add_argument(
        "--stats-interval", type=int, default=60,
        help="How often (seconds) to print summary stats (default: 60)",
    )
    args = parser.parse_args()

    # Optionally launch ollama serve
    ollama_proc = None
    if args.start_ollama:
        ollama_proc = _start_ollama(args.ollama_cmd)

    # Wait until Ollama is up
    if not _wait_for_ollama(args.ollama):
        log.warning("Proceeding anyway — requests will fail until Ollama is ready.")

    # Configure the handler class
    ProxyHandler.ollama_base = args.ollama.rstrip("/")
    if args.log_file:
        ProxyHandler.jsonl = JsonlWriter(args.log_file)

    # Start stats printer
    threading.Thread(
        target=_stats_printer, args=(args.stats_interval,), daemon=True, name="stats"
    ).start()

    # Start proxy server
    server = HTTPServer(("0.0.0.0", args.port), ProxyHandler)
    log.info("=" * 60)
    log.info("  Ollama proxy listening on port %d", args.port)
    log.info("  Forwarding to:  %s", args.ollama)
    if args.log_file:
        log.info("  Structured log: %s", args.log_file)
    log.info("")
    log.info("  On the Pi, set:")
    log.info("    OLLAMA_URL=http://<this-pc-ip>:%d", args.port)
    log.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("\nShutting down proxy.")
        log.info("[STATS] %s", stats.summary())
    finally:
        server.server_close()
        if ollama_proc:
            ollama_proc.terminate()


if __name__ == "__main__":
    main()
