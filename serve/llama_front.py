"""serve/llama_front.py - the web app in front of llama.cpp's llama-server, for any GGUF model.

    python serve/llama_front.py --config llama-<model>.json [--port 8080] [--open]

Strata's own server (serve/server.py) drives the Strata engine token by token with Qwen3.8-Flash-Next's template and
tokenizer.  Other models need their own template, reasoning and tool-call parsing, which llama-server already has
for every model it loads (--jinja).  So here llama-server runs as a child on a loopback port of its own, and this
server keeps what the web app adds around it:

  - the web app itself (serve/web: Chat, Monitor, About) and the model manager (/manage, serve/setup_web.py)
  - /health, /metrics (the Monitor tab: this server's request history and live rate, the hardware from
    serve/telemetry.py), /settings (the Chat settings other apps get), /mcp (none here)
  - /v1/* passed through to llama-server, streaming as it streams, with Strata's API key, Host and Origin checks

The run config (written by setup, or by hand):
    {"backend": "llamacpp", "exe": ".../llama-server", "model": ".../model.gguf", "args": ["-c", "32768"],
     "model_name": "qwen3-0.6b", "port": 8080, "log": "llama-qwen3-0.6b.log",
     "api_key": "", "allowed_hosts": [], "cors_origins": [], "trusted_origins": []}
"args" go to llama-server as they are (its own flags: -c, -ngl, --n-cpu-moe, --spec-type, ...).  llama-server's
--fit (on by default) places the layers and MoE experts that fit in VRAM and leaves the rest in RAM.
"""
from __future__ import annotations

import argparse
import collections
import hmac
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import (Server, allowed_hosts_of, clean_shared_defaults, host_allowed,  # noqa: E402
                          host_names_for, manager, origin_allowed, origins_of)
from serve.winjob import contain  # noqa: E402

WEB = ROOT / "serve" / "web"
RATE_WINDOW_S = 4.0
RATE_MIN_SPAN_S = 0.5                                   # the live rate right after the first token: at least this span
# the request fields the web app's Chat settings fill when a client leaves them out (POST /settings)
SHARED_TO_REQUEST = ("temperature", "top_p", "top_k", "seed", "max_tokens")
PASS_HEADERS = ("Content-Type", "Cache-Control")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------------------------------------ llama-server
class LlamaServer:
    """llama-server as a child process on a loopback port, its output in the log file.  Started again by the next
    request when it ended (a crash, the out-of-memory killer)."""

    def __init__(self, exe: str, model: str, args: list[str], model_name: str, log: str | None, cwd: str | None):
        self.exe, self.model, self.args, self.model_name = exe, model, list(args), model_name
        self.log_path, self.cwd = log, cwd
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.props: dict = {}
        self.lock = threading.Lock()

    def command(self) -> list[str]:
        return [self.exe, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port), "--alias",
                self.model_name, "--no-webui", "--jinja", *self.args]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout_s: float = 1800.0, narrate=print) -> None:
        with self.lock:
            if self.alive():
                return
            self.port = free_port()
            log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.log_start = os.path.getsize(self.log_path) if self.log_path else 0
            self.proc = subprocess.Popen(self.command(), cwd=self.cwd or None, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, creationflags=flags)
            contain(self.proc)                           # ends with this server, however it ends (Windows)
            t0, said = time.time(), 0.0
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server ended while loading (code {self.proc.returncode})"
                                       + self.log_tail())
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:                            # 503 while loading, or not listening yet
                    pass
                if time.time() - t0 > timeout_s:
                    self.stop()
                    raise RuntimeError(f"llama-server did not answer within {timeout_s:.0f} s" + self.log_tail())
                if time.time() - said >= 20:
                    if said:
                        narrate(f"[llama.cpp] still loading ({time.time() - t0:.0f} s) - please wait ...")
                    said = time.time()
                time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/props", timeout=10) as r:
                    self.props = json.loads(r.read())
            except (OSError, ValueError):
                self.props = {}

    def stop(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    def log_tail(self, n: int = 15) -> str:
        if not self.log_path:
            return ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(getattr(self, "log_start", 0))
                lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
        except OSError:
            return ""
        return "\n  " + "\n  ".join(lines) if lines else ""

    def max_context(self) -> int:
        s = (self.props.get("default_generation_settings") or {})
        return int(s.get("n_ctx") or self.props.get("n_ctx") or 0)

    def version(self) -> str:
        b = self.props.get("build_info") or ""
        return f"llama.cpp {b}".strip()


# ------------------------------------------------------------------------------------------------ the front
class Front:
    """What this server knows besides llama-server: the requests it passed on (the Monitor's table and totals), the
    live rate of the one streaming now, the hardware sampler, the shared Chat settings and the access rules."""

    def __init__(self, llama: LlamaServer, model_name: str):
        self.llama, self.model = llama, model_name
        self.status_lock = threading.Lock()
        self.live: dict[int, dict] = {}                 # request id -> its state while it runs
        self.next_id = 0
        # (time, tokens) of the streaming answers in the last RATE_WINDOW_S.  Not a fixed length: 64 entries of one
        # token each capped the live rate at 64 / 4 s = 16 tok/s whatever the real speed (a user's PC, 2026-10-21)
        self.rate = collections.deque()
        self.history = collections.deque(maxlen=500)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0}
        self.shared: dict = {}
        self.shared_path: Path | None = None
        self.api_key = ""
        self.cors_origins: list[str] = []
        self.trusted_origins: list[str] = []
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set()
        self.telemetry = None
        self.started_at = time.time()

    def start_telemetry(self, gpu_index: int = 0) -> None:
        from serve.telemetry import Telemetry
        self.telemetry = Telemetry(extra=lambda: {"tok_s": self.tok_s(), "tok_s_mean": None,
                                                  "prefill_tok_s_mean": None}, gpu_index=gpu_index)

    # --------------------------------------------------------------------------------------- request accounting
    def begin(self, max_tokens) -> int:
        with self.status_lock:
            self.next_id += 1
            self.live[self.next_id] = {"started": time.time(), "first_token": None, "generated": 0,
                                       "max_tokens": max_tokens}
            return self.next_id

    def token(self, rid: int, n: int = 1) -> None:
        now = time.time()
        with self.status_lock:
            s = self.live.get(rid)
            if s is None:
                return
            if s["first_token"] is None:
                s["first_token"] = now
            s["generated"] += n
            self.rate.append((now, n))
            while now - self.rate[0][0] > RATE_WINDOW_S:
                self.rate.popleft()

    def finish(self, rid: int, finish: str, usage: dict | None, timings: dict | None) -> None:
        with self.status_lock:
            s = self.live.pop(rid, None)
            if s is None:
                return
            usage, t = usage or {}, timings or {}
            prompt = usage.get("prompt_tokens", t.get("prompt_n"))
            out = usage.get("completion_tokens", t.get("predicted_n", s["generated"]))
            cached = t.get("cache_n")
            decode_ms, prompt_ms = t.get("predicted_ms"), t.get("prompt_ms")
            rec = {"time": s["started"], "duration_s": round(time.time() - s["started"], 1), "finish": finish,
                   "prompt_tokens": prompt, "reused": cached, "output_tokens": out,
                   "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                   "decode_tok_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else
                   (round(out / (decode_ms / 1000), 1) if out and decode_ms else None),
                   "prefill_tok_s": round(t["prompt_per_second"], 1) if t.get("prompt_per_second") else None,
                   "hit_rate": None, "pcie_share": None,
                   "drafts_offered": t.get("draft_n"), "drafts_accepted": t.get("draft_n_accepted")}
            self.history.append(rec)
            tt = self.totals
            tt["requests"] += 1
            tt["prompt_tokens"] += prompt or 0
            tt["reused"] += cached or 0
            tt["output_tokens"] += out or 0
            tt["prompt_ms"] += prompt_ms or 0.0
            tt["decode_ms"] += decode_ms or 0.0
            tt["drafts_offered"] += t.get("draft_n") or 0
            tt["drafts_accepted"] += t.get("draft_n_accepted") or 0

    def tok_s(self) -> float:
        """Tokens of the last RATE_WINDOW_S seconds over that span - or over the time since the first token, when the
        answers began less than RATE_WINDOW_S ago.  With parallel requests: the sum of theirs."""
        now = time.time()
        with self.status_lock:
            n = sum(k for at, k in self.rate if now - at <= RATE_WINDOW_S)
            firsts = [s["first_token"] for s in self.live.values() if s["first_token"] is not None]
        if not firsts:
            return 0.0
        return n / min(RATE_WINDOW_S, max(RATE_MIN_SPAN_S, now - min(firsts)))

    def metrics(self, all_requests=False) -> dict:
        now = time.time()
        with self.status_lock:
            runs = list(self.live.values())
            hist, totals = list(self.history), dict(self.totals)
        s = runs[-1] if runs else None
        if s is None:
            state = "idle" if self.llama.alive() else "unloaded"
        else:
            state = "reading" if s["first_token"] is None else "generating"
        live = {"state": state, "queued": max(0, len(runs) - 1), "phase": None, "prompt_tokens": None,
                "prompt_read": None, "prompt_total": None, "generated": s["generated"] if s else None,
                "max_tokens": s["max_tokens"] if s else None,
                "elapsed_s": round(now - s["started"], 1) if s else None,
                "tok_s": round(self.tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(s["generated"] / max(1e-6, now - s["first_token"]), 1)
                if state == "generating" else None,
                "prefill_tok_s_mean": None, "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        engine = {"model": self.model, "max_context": self.llama.max_context(), "images": False,
                  "version": self.llama.version(), "backend": "llama.cpp"}
        tel = self.telemetry.snapshot() if self.telemetry else {"now": {}, "history": {}, "static": {}}
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "conversation_cache": None, "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static": tel["static"], "history": tel["history"], "time": now}

    # --------------------------------------------------------------------------------------- requests passed on
    def prepare(self, path: str, req: dict) -> dict:
        """A client's request as llama-server takes it: the shared Chat settings where the client left a field out,
        and the web app's thinking depth (reasoning_effort none/low/medium/high) in llama-server's terms."""
        for k in SHARED_TO_REQUEST:
            if k in self.shared and k not in req:
                req[k] = self.shared[k]
        effort = req.get("reasoning_effort", self.shared.get("reasoning_effort"))
        if path.endswith("/chat/completions") and effort is not None:
            req.pop("reasoning_effort", None)
            ctk = dict(req.get("chat_template_kwargs") or {})
            if effort == "none":
                ctk.setdefault("enable_thinking", False)   # Qwen3 and most templates with a thinking switch
                req["reasoning_format"] = req.get("reasoning_format", "deepseek")
            elif effort in ("low", "medium", "high"):
                req["reasoning_effort"] = effort
            if ctk:
                req["chat_template_kwargs"] = ctk
        if req.get("stream") and path.endswith("/chat/completions"):
            req.setdefault("stream_options", {"include_usage": True})
        return req


def make_handler(front: Front):
    class Handler(BaseHTTPRequestHandler):
        server_version = "strata-llama"
        protocol_version = "HTTP/1.0"                   # one request per connection: a stream ends with the close

        def log_message(self, *a):
            pass

        # ------------------------------------------------------------------ access (as serve/server.py)
        def parse_request(self):
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if front.api_key or host_allowed(host, front.host_names, "*" in front.allowed_hosts):
                return True
            print(f"[llama.cpp] refused a request for Host {host!r}: not a name this server answers to "
                  f"(add it to \"allowed_hosts\" in the config, or set an API key)", flush=True)
            self._json(403, {"error": {"type": "forbidden", "message": f"Host {host!r} is not allowed "
                             "(DNS rebinding protection); add it to \"allowed_hosts\" or set \"api_key\""}})
            return False

        def _authorized(self) -> bool:
            if not front.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if hmac.compare_digest(given.encode(), front.api_key.encode()):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _foreign_page(self) -> bool:
            origin = self.headers.get("Origin")
            if front.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), front.host_names,
                                  [*front.trusted_origins, *front.cors_origins]):
                self._json(403, {"error": {"type": "forbidden", "message": f"web pages of {origin} may not use "
                                 "this server without an API key"}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def _own_page(self) -> bool:
            """The model manager and the shared settings change this PC: only this server's own page (or a client
            without a browser Origin) may use them."""
            origin = self.headers.get("Origin")
            if origin and not origin_allowed(origin, self.headers.get("Host"), front.host_names,
                                             front.trusted_origins):
                self._json(403, {"error": {"type": "forbidden", "message": "only this server's own page"}})
                return False
            return True

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and (origin in front.cors_origins or "*" in front.cors_origins):
                self.send_header("Access-Control-Allow-Origin", origin if "*" not in front.cors_origins else "*")
                self.send_header("Vary", "Origin")

        # ------------------------------------------------------------------ replies
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, f: Path, ctype: str, cache="no-cache") -> None:
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, "
                             "anthropic-version")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/fonts/") or path.startswith("/web/"):
                sub, name = ("fonts", path[7:]) if path.startswith("/fonts/") else ("", path[5:])
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".woff2": "font/woff2"}
                f = WEB / sub / name if sub else WEB / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file() or (sub and ext != ".woff2"):
                    self._json(404, {"error": {"message": "not found"}})
                    return
                self._file(f, types[ext], "max-age=86400" if sub else "no-cache")
            elif path == "":
                self._file(WEB / "index.html", "text/html; charset=utf-8")
            elif path == "/manage":
                self._file(WEB / "setup.html", "text/html; charset=utf-8")
            elif path.startswith("/manage/api/"):
                if self._authorized():
                    self._json(*manager().api("GET", self.path, None, embedded=True))
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": front.llama.max_context(), "model": front.model,
                                 "images": False, "api_key": bool(front.api_key), "loaded": front.llama.alive(),
                                 "service": "strata", "backend": "llama.cpp"})
            elif path == "/metrics":
                if self._authorized():
                    self._json(200, front.metrics(all_requests="requests=all" in self.path))
            elif path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(front.shared), "defaults": front.shared})
            elif path == "/mcp":
                if self._authorized():
                    self._json(200, {"servers": [], "tools": 0})
            elif path.startswith("/v1/") or path in ("/props", "/slots"):
                if self._authorized():
                    self._pass("GET", None)
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path.startswith("/manage/api/"):
                if self._authorized() and self._own_page():
                    self._json(*manager().api("POST", self.path, raw, embedded=True))
                return
            if path == "/settings":
                if not (self._authorized() and self._own_page()):
                    return
                try:
                    front.shared = clean_shared_defaults(json.loads(raw or b"{}").get("defaults"))
                except (ValueError, AttributeError) as e:
                    self._json(400, {"error": {"message": str(e)}})
                    return
                if front.shared_path:
                    try:
                        front.shared_path.write_text(json.dumps(front.shared), encoding="utf-8")
                    except OSError:
                        pass
                self._json(200, {"shared": bool(front.shared), "defaults": front.shared})
                return
            if not path.startswith("/v1/"):
                self._json(404, {"error": {"message": "not found"}})
                return
            if not self._authorized() or self._foreign_page():
                return
            try:
                req = json.loads(raw or b"{}")
            except ValueError:
                self._json(400, {"error": {"message": "the body is not JSON"}})
                return
            if isinstance(req, dict):
                req = front.prepare(path, req)
                raw = json.dumps(req).encode()
            self._pass("POST", raw, req if isinstance(req, dict) else {})

        # ------------------------------------------------------------------ pass-through
        def _pass(self, method: str, raw: bytes | None, req: dict | None = None) -> None:
            """The request to llama-server and its answer back as it comes (a stream line by line).  A chat or
            completion request is counted for the Monitor: its tokens as they stream, its usage and timings."""
            if not front.llama.alive():
                try:
                    print("[llama.cpp] llama-server is not running: starting it again ...", flush=True)
                    front.llama.start()
                except RuntimeError as e:
                    self._json(503, {"error": {"type": "unavailable", "message": str(e)}})
                    return
            counted = method == "POST" and self.path.split("?")[0].rstrip("/").endswith(
                ("/chat/completions", "/completions", "/messages"))
            rid = front.begin((req or {}).get("max_tokens")) if counted else None
            finish, usage, timings = "error", None, None
            conn = http.client.HTTPConnection("127.0.0.1", front.llama.port, timeout=3600)
            try:
                headers = {"Content-Type": "application/json"} if raw is not None else {}
                conn.request(method, self.path, body=raw, headers=headers)
                r = conn.getresponse()
                self.send_response(r.status)
                for h in PASS_HEADERS:
                    if r.getheader(h):
                        self.send_header(h, r.getheader(h))
                self._cors()
                streaming = (r.getheader("Content-Type") or "").startswith("text/event-stream")
                if not streaming:
                    body = r.read()
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    if counted:
                        try:
                            obj = json.loads(body)
                            usage, timings = obj.get("usage"), obj.get("timings")
                            choice = (obj.get("choices") or [{}])[0]
                            finish = choice.get("finish_reason") or obj.get("stop_reason") or \
                                ("stop" if r.status == 200 else "error")
                            front.token(rid, (usage or {}).get("completion_tokens", 0) or 0)
                        except (ValueError, AttributeError, IndexError):
                            finish = "stop" if r.status == 200 else "error"
                    return
                self.end_headers()
                finish = "disconnect"
                for line in r:                          # SSE: one data line per event
                    self.wfile.write(line)
                    if line in (b"\n", b"\r\n"):
                        self.wfile.flush()
                    if not counted or not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except ValueError:
                        continue
                    for c in obj.get("choices") or []:
                        d = c.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                            front.token(rid)
                        if c.get("finish_reason"):
                            finish = c["finish_reason"]
                    if obj.get("type") == "content_block_delta":                # /v1/messages (Anthropic)
                        front.token(rid)
                    if obj.get("type") == "message_delta":
                        finish = ((obj.get("delta") or {}).get("stop_reason")) or finish
                        usage = {"completion_tokens": (obj.get("usage") or {}).get("output_tokens")}
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if obj.get("timings"):
                        timings = obj["timings"]
                if finish == "disconnect":
                    finish = "stop"
            except (ConnectionError, OSError) as e:
                if isinstance(e, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                    finish = "disconnect"                # the client hung up
                else:
                    try:
                        self._json(502, {"error": {"type": "bad_gateway", "message": f"llama-server: {e}"}})
                    except OSError:
                        pass
            finally:
                conn.close()
                if counted:
                    front.finish(rid, {"end_turn": "stop", "max_tokens": "length"}.get(finish, finish), usage,
                                 timings)

    return Handler


# ------------------------------------------------------------------------------------------------ start
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run config (llama-<model>.json)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--open", action="store_true", help="open the web app in the browser when it is ready")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if cfg.get("backend") != "llamacpp":
        ap.error(f"{cfg_path.name} is not a llama.cpp run config (\"backend\": \"llamacpp\")")
    if cfg.get("llama_build"):                       # written by setup: an older pinned build is moved up once, and on
        from serve import llama_models               # Windows the C++ runtime is put beside llama-server if missing
        cfg = llama_models.refresh_config(cfg_path)
    host =a.host or cfg.get("host") or "127.0.0.1"
    port = a.port or int(cfg.get("port") or 8080)
    try:
        Server((host, port), BaseHTTPRequestHandler).server_close()
    except OSError:
        ap.error(f"port {port} is already in use - is a model already running? Close it, or use another --port")
    cwd = cfg.get("cwd") or str(cfg_path.parent)
    absolute = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(cwd, p))  # noqa: E731
    exe, model = absolute(cfg["exe"]), absolute(cfg["model"])
    for what, p in (("llama-server", exe), ("model file", model)):
        if not os.path.isfile(p):
            ap.error(f"the {what} is missing: {p} (run setup again)")
    name = cfg.get("model_name") or Path(model).stem
    llama = LlamaServer(exe, model, [str(x) for x in cfg.get("args", [])], name,
                        absolute(cfg["log"]) if cfg.get("log") else None, cwd)
    front = Front(llama, name)
    front.api_key = a.api_key or cfg.get("api_key", "")
    front.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    front.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    front.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    front.host_names = host_names_for(host, front.allowed_hosts, front.trusted_origins)
    front.shared_path = cfg_path.with_name(cfg_path.stem + ".shared.json")
    try:
        front.shared = clean_shared_defaults(json.loads(front.shared_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        front.shared = {}
    print(f"[llama.cpp] loading {Path(model).name} with {Path(exe).name} ...", flush=True)
    try:
        llama.start()
    except RuntimeError as e:
        print(f"[llama.cpp] {e}", file=sys.stderr, flush=True)
        return 1
    front.start_telemetry(int(cfg.get("gpu") or 0) if str(cfg.get("gpu", "0")).isdigit() else 0)
    httpd = Server((host, port), make_handler(front))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"ready: {url}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context "
          f"{llama.max_context()} tokens, {llama.version()})", flush=True)
    print(f"       open {url}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.open:
        threading.Timer(0.5, lambda: webbrowser.open(url + "/")).start()
    # as serve/server.py (#96): SIGTERM (kill, systemd, docker stop) takes Ctrl+C's path, so llama-server is stopped
    # too - Python's default ends this process alone and left llama-server running with its VRAM (seen on llmserver)
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        llama.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
