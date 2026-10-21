"""serve/llama_front.py, the web app in front of llama-server: requests reach llama-server in its terms, its answers
come back byte for byte (a stream as it streams), the Monitor counts them, and the access checks of serve/server.py
hold here too.  A small fake llama-server stands in for the real one.

Run: python -m unittest serve.test_llama_front
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import llama_front as LF  # noqa: E402
from serve.server import host_names_for  # noqa: E402

SSE = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
       {"choices": [{"index": 0, "delta": {"reasoning_content": "Let me add."}}]},
       {"choices": [{"index": 0, "delta": {"content": "1+1"}}]},
       {"choices": [{"index": 0, "delta": {"content": " = 2"}}]},
       {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        "timings": {"prompt_n": 12, "prompt_ms": 40.0, "predicted_n": 3, "predicted_ms": 100.0,
                    "predicted_per_second": 30.0, "prompt_per_second": 300.0, "cache_n": 5,
                    "draft_n": 4, "draft_n_accepted": 2}}]


def sse_bytes() -> bytes:
    return b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in SSE) + b"data: [DONE]\n\n"


class FakeLlama(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok"} if self.path == "/health" else {"object": "list", "data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.seen.append((self.path, req))
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(sse_bytes())
            return
        body = json.dumps({"choices": [{"index": 0, "finish_reason": "length",
                                        "message": {"role": "assistant", "content": "ok"}}],
                           "usage": {"prompt_tokens": 7, "completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StubLlama(LF.LlamaServer):
    """A LlamaServer that is the fake one already running: nothing is started."""

    def __init__(self, port):
        super().__init__("llama-server", "m.gguf", [], "test-model", None, None)
        self.port, self.props = port, {"default_generation_settings": {"n_ctx": 4096}, "build_info": "b1-abc"}

    def alive(self):
        return True


class Base(unittest.TestCase):
    def setUp(self):
        FakeLlama.seen = []
        self.fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
        threading.Thread(target=self.fake.serve_forever, daemon=True).start()
        self.front = LF.Front(StubLlama(self.fake.server_address[1]), "test-model")
        self.front.host_names = host_names_for("127.0.0.1")
        self.httpd = LF.Server(("127.0.0.1", 0), LF.make_handler(self.front))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]

    def tearDown(self):
        for s in (self.httpd, self.fake):
            s.shutdown()
            s.server_close()

    def post(self, path, obj, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        return urllib.request.urlopen(urllib.request.Request(self.base + path, json.dumps(obj).encode(), h),
                                      timeout=10)

    def code(self, req):
        try:
            return urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            return e.code


class Prepare(unittest.TestCase):
    def setUp(self):
        self.front = LF.Front(StubLlama(1), "m")

    def test_thinking_off_is_the_template_switch(self):
        r = self.front.prepare("/v1/chat/completions", {"reasoning_effort": "none", "messages": []})
        self.assertNotIn("reasoning_effort", r)
        self.assertEqual(r["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_depth_passes(self):
        r = self.front.prepare("/v1/chat/completions", {"reasoning_effort": "low", "messages": []})
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertNotIn("chat_template_kwargs", r)

    def test_a_client_s_own_switch_wins(self):
        r = self.front.prepare("/v1/chat/completions", {"reasoning_effort": "none",
                                                        "chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(r["chat_template_kwargs"], {"enable_thinking": True})

    def test_shared_settings_fill_what_is_missing(self):
        self.front.shared = {"temperature": 0.2, "top_k": 5, "reasoning_effort": "none"}
        r = self.front.prepare("/v1/chat/completions", {"temperature": 0.9, "messages": []})
        self.assertEqual((r["temperature"], r["top_k"]), (0.9, 5))
        self.assertEqual(r["chat_template_kwargs"], {"enable_thinking": False})

    def test_a_stream_asks_for_usage(self):
        r = self.front.prepare("/v1/chat/completions", {"stream": True})
        self.assertEqual(r["stream_options"], {"include_usage": True})


class LiveRate(unittest.TestCase):
    def test_the_real_rate_not_capped(self):
        """64 one-token samples over 4 s capped the header's rate at 16 tok/s (2026-10-21)."""
        front = LF.Front(StubLlama(1), "m")
        clock = [1000.0]
        with mock.patch.object(LF.time, "time", lambda: clock[0]):
            rid = front.begin(None)
            for _ in range(300):                         # 50 tok/s for 6 s
                front.token(rid)
                clock[0] += 0.02
            self.assertAlmostEqual(front.tok_s(), 50.0, delta=1.0)
            self.assertLessEqual(len(front.rate), 201)   # only the last window is kept

    def test_right_after_the_first_token(self):
        front = LF.Front(StubLlama(1), "m")
        clock = [1000.0]
        with mock.patch.object(LF.time, "time", lambda: clock[0]):
            rid = front.begin(None)
            for _ in range(40):                          # 40 tok/s for 1 s: not divided by the whole 4 s window
                front.token(rid)
                clock[0] += 0.025
            self.assertAlmostEqual(front.tok_s(), 40.0, delta=1.5)
            front.finish(rid, "stop", None, None)
            self.assertEqual(front.tok_s(), 0.0)


class PassThrough(Base):
    def test_stream_comes_back_byte_for_byte_and_is_counted(self):
        body = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "1+1"}],
                                                  "stream": True, "reasoning_effort": "none"}).read()
        self.assertEqual(body, sse_bytes())
        path, req = FakeLlama.seen[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": False})
        m = json.loads(urllib.request.urlopen(self.base + "/metrics").read())
        r = m["requests"][0]
        self.assertEqual((r["finish"], r["prompt_tokens"], r["output_tokens"], r["reused"]), ("stop", 12, 3, 5))
        self.assertEqual((r["decode_tok_s"], r["drafts_offered"], r["drafts_accepted"]), (30.0, 4, 2))
        self.assertEqual(m["totals"]["requests"], 1)
        self.assertEqual(m["engine"]["version"], "llama.cpp b1-abc")
        self.assertEqual(m["engine"]["max_context"], 4096)
        self.assertEqual(m["live"]["state"], "idle")

    def test_plain_json(self):
        obj = json.loads(self.post("/v1/chat/completions", {"messages": []}).read())
        self.assertEqual(obj["choices"][0]["message"]["content"], "ok")
        r = json.loads(urllib.request.urlopen(self.base + "/metrics").read())["requests"][0]
        self.assertEqual((r["finish"], r["output_tokens"]), ("length", 9))

    def test_models_and_health(self):
        self.assertEqual(json.loads(urllib.request.urlopen(self.base + "/v1/models").read())["object"], "list")
        h = json.loads(urllib.request.urlopen(self.base + "/health").read())
        self.assertEqual((h["model"], h["max_context"], h["backend"]), ("test-model", 4096, "llama.cpp"))

    def test_the_web_app_is_served(self):
        self.assertIn(b"<html", urllib.request.urlopen(self.base + "/").read().lower())
        self.assertEqual(urllib.request.urlopen(self.base + "/web/app.js").status, 200)
        self.assertEqual(self.code(urllib.request.Request(self.base + "/web/../setup.py")), 404)
        self.assertEqual(self.code(urllib.request.Request(self.base + "/config")), 404)   # the card hides


class Access(Base):
    def test_rebinding_host_is_refused(self):
        self.assertEqual(self.code(urllib.request.Request(self.base + "/health",
                                                          headers={"Host": "evil.example.com"})), 403)

    def test_foreign_page_is_refused(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", b"{}",
                                     {"Origin": "https://evil.example.com", "Content-Type": "application/json"})
        self.assertEqual(self.code(req), 403)
        self.assertEqual(FakeLlama.seen, [])

    def test_api_key(self):
        self.front.api_key = "k1"
        req = lambda key: urllib.request.Request(  # noqa: E731
            self.base + "/v1/chat/completions", json.dumps({"messages": []}).encode(),
            {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
        self.assertEqual(self.code(req(None)), 401)
        self.assertEqual(self.code(req("wrong")), 401)
        self.assertEqual(self.code(req("k1")), 200)

    def test_settings_only_from_own_page(self):
        req = urllib.request.Request(self.base + "/settings", json.dumps({"defaults": {"temperature": 0.3}}).encode(),
                                     {"Content-Type": "application/json", "Origin": "https://evil.example.com"})
        self.assertEqual(self.code(req), 403)
        self.assertEqual(self.front.shared, {})
        self.post("/settings", {"defaults": {"temperature": 0.3}})
        self.assertEqual(self.front.shared, {"temperature": 0.3})


if __name__ == "__main__":
    unittest.main()
