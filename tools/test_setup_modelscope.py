"""ModelScope as a download source (--source / STRATA_SOURCE): which source is used, the URL mapping, and that a file
from ModelScope is checked against the SHA-256 ModelScope publishes (a wrong one is deleted).  No network: a local
HTTP server plays ModelScope.

    python -m unittest tools.test_setup_modelscope
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
PATH = "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
HF_URL = setup.hf(REPO) + PATH


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class Base(unittest.TestCase):
    def setUp(self):
        for name in ("say", "ok", "warn"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for k in ("STRATA_SOURCE", "HF_ENDPOINT", "MODELSCOPE_ENDPOINT"):
            os.environ.pop(k, None)
        setup._sources.clear()
        setup._ms_files.clear()
        self.addCleanup(setup._sources.clear)


class Source(Base):
    def test_explicit(self):
        for value, want in (("modelscope", "modelscope"), ("ms", "modelscope"), ("huggingface", "huggingface"),
                            ("hf", "huggingface")):
            setup._sources.clear()
            os.environ["STRATA_SOURCE"] = value
            self.assertEqual(setup.model_source(), want, value)

    def test_auto(self):
        def only(host):
            return lambda url, timeout=5.0: host in url
        # this fork: ModelScope first
        for answers, want in ((only("huggingface.co"), "huggingface"), (only("modelscope.cn"), "modelscope"),
                              (lambda url, timeout=5.0: True, "modelscope"),
                              (lambda url, timeout=5.0: False, "huggingface")):
            setup._sources.clear()
            with mock.patch.object(setup, "reachable", side_effect=answers):
                self.assertEqual(setup.model_source(), want)

    def test_hf_endpoint_alone_keeps_modelscope(self):
        # HF_ENDPOINT=https://hf-mirror.com set for other tools: ModelScope still comes first, the mirror when it
        # does not answer (2026-10-22: the mirror crawled at 0.0 MB/s on llmserver where ModelScope gives 11-14)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        for answers, want in ((True, "modelscope"), (False, "huggingface")):
            setup._sources.clear()
            with mock.patch.object(setup, "reachable", return_value=answers):
                self.assertEqual(setup.model_source(), want)
        os.environ["STRATA_SOURCE"] = "huggingface"
        setup._sources.clear()
        with mock.patch.object(setup, "reachable", return_value=True) as r:
            self.assertEqual(setup.model_source(), "huggingface")
            r.assert_not_called()

    def test_mapping(self):
        self.assertEqual(setup.ms_file(HF_URL), (REPO, PATH))
        self.assertEqual(setup.ms_url(REPO, PATH),
                         "https://www.modelscope.cn/models/" + REPO + "/resolve/master/" + PATH)
        self.assertIsNone(setup.ms_file("https://huggingface.co/someone/other/resolve/main/x.gguf"))
        self.assertIsNone(setup.ms_file(setup.LLAMA_CPP_ZIP))


class Download(Base):
    """download() through a local server laid out as ModelScope: /models/<repo>/resolve/master/<path> and
    /api/v1/models/<repo>/repo/files (the query string is ignored by the test server)."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.www = Path(self.tmp.name) / "www"
        self.data = os.urandom(300_000)
        f = self.www / "models" / REPO / "resolve" / "master" / PATH
        f.parent.mkdir(parents=True)
        f.write_bytes(self.data)
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(self.www), **k))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        os.environ["MODELSCOPE_ENDPOINT"] = "http://127.0.0.1:%d" % self.server.server_address[1]
        os.environ["STRATA_SOURCE"] = "modelscope"
        self.dst = Path(self.tmp.name) / "out" / Path(PATH).name

    def api(self, sha):
        p = self.www / "api" / "v1" / "models" / REPO / "repo" / "files"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"Data": {"Files": [{"Path": PATH, "Size": len(self.data), "Sha256": sha}]}}))

    def test_right_hash_is_kept(self):
        sha = hashlib.sha256(self.data).hexdigest()
        self.api(sha)
        setup.download(HF_URL, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertIn("sha256 " + sha, self.dst.with_name(self.dst.name + ".done").read_text())

    def test_wrong_hash_is_deleted(self):
        self.api("0" * 64)
        with mock.patch.object(setup, "say"), self.assertRaises(SystemExit):
            setup.download(HF_URL, self.dst)
        self.assertFalse(self.dst.exists())
        self.assertFalse(self.dst.with_name(self.dst.name + ".done").exists())

    def test_no_hash_published(self):
        setup.download(HF_URL, self.dst)              # no API file: the download is kept, as from Hugging Face
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertTrue(self.dst.with_name(self.dst.name + ".done").exists())

    def test_falls_back_when_modelscope_is_silent(self):
        calls = []
        with mock.patch.object(setup, "reachable", return_value=False), \
                mock.patch.object(setup.urllib.request, "urlopen",
                                  side_effect=lambda req, timeout=0: calls.append(req.full_url) or (_ for _ in ()).throw(
                                      OSError("offline"))), \
                mock.patch.object(setup.time, "sleep"), self.assertRaises(SystemExit):
            setup.download(HF_URL, self.dst)
        self.assertTrue(calls and all(u == HF_URL for u in calls), calls[:2])


class NoLengthOnHead(Download):
    """ModelScope's file links answer HEAD without a Content-Length (measured 2026-10-07); the size is in the GET's
    Content-Range.  A transfer cut short must be resumed, not taken for the whole file."""

    def setUp(self):
        super().setUp()
        data, gets = self.data, []
        self.gets = gets

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.end_headers()

            def do_GET(self):
                start = int(re.match(r"bytes=(\d+)-", self.headers.get("Range", "bytes=0-")).group(1))
                gets.append(start)
                part = data[start:] if len(gets) > 1 else data[start:start + len(data) // 2]   # the first one is cut
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)

        self.server.shutdown()
        self.server.server_close()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        os.environ["MODELSCOPE_ENDPOINT"] = "http://127.0.0.1:%d" % self.server.server_address[1]

    def test_a_cut_transfer_is_resumed(self):                 # one stream (the parallel path probes a range first)
        with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}),                 mock.patch.object(setup, "reachable", return_value=True), mock.patch.object(setup, "ms_meta",
                                                                                          return_value=None),                 mock.patch.object(setup, "say"):
            setup.download(HF_URL, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertEqual(self.gets, [0, len(self.data) // 2])

    test_right_hash_is_kept = test_wrong_hash_is_deleted = test_no_hash_published = None
    test_falls_back_when_modelscope_is_silent = None


class RangeServer(http.server.BaseHTTPRequestHandler):
    """A file served with byte ranges (or without them, `ranges` False), every GET's range recorded; `cut` cuts the
    answer to the first n bytes once (a dropped connection)."""
    data, ranges, gets, cut = b"", True, [], 0

    def log_message(self, *a):
        pass

    def do_HEAD(self):                                    # as ModelScope: no length on HEAD
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        cls, data = type(self), type(self).data
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if not (m and cls.ranges):
            cls.gets.append(None)
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start, end = int(m.group(1)), int(m.group(2) or len(data) - 1)
        cls.gets.append((start, end))
        body = data[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.cut and len(body) > cls.cut:
            body, cls.cut = body[:cls.cut], 0
        self.wfile.write(body)


class Parallel(Base):
    """A big file over several connections, a chunk each (STRATA_DL_CONNECTIONS, default 4), resumable by chunk."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = os.urandom(1_000_003)
        handler = type("H", (RangeServer,), {"data": self.data, "gets": []})
        self.handler = handler
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = "http://127.0.0.1:%d/f.gguf" % self.server.server_address[1]
        self.dst = Path(self.tmp.name) / "f.gguf"
        self.part = self.dst.with_name("f.gguf.part")
        for name, value in (("DL_CHUNK", 100_000), ("DL_PARALLEL_MIN", 200_000)):
            p = mock.patch.object(setup, name, value)
            p.start()
            self.addCleanup(p.stop)
        for p in (mock.patch.object(setup, "model_source", lambda: "huggingface"),
                  mock.patch.object(setup.time, "sleep"), mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "4"})):
            p.start()
            self.addCleanup(p.stop)

    def chunk_gets(self):
        return sorted(g for g in self.handler.gets if g and g != (0, 0))

    def test_every_chunk_once(self):
        setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertEqual(self.chunk_gets(), [(i, min(i + 99_999, len(self.data) - 1)) for i in range(0, len(self.data), 100_000)])
        self.assertFalse(self.part.exists() or self.part.with_name("f.gguf.part.chunks").exists())

    def test_a_dropped_connection_is_resumed_within_its_chunk(self):
        self.handler.cut = 30_000
        setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)

    def test_resumes_with_the_chunks_missing(self):
        # a run stopped after chunks 0, 3 and 9: their bytes are on the disk, the state lists them
        buf = bytearray(len(self.data))
        for i in (0, 3, 9):
            buf[i * 100_000:(i + 1) * 100_000] = self.data[i * 100_000:(i + 1) * 100_000]
        self.part.write_bytes(bytes(buf[:1_000_000]))
        self.part.with_name("f.gguf.part.chunks").write_text(
            json.dumps({"total": len(self.data), "chunk": 100_000, "done": [0, 3, 9]}))
        self.assertEqual(setup.part_bytes(self.part), 300_000)
        setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertNotIn(300_000, [g[0] for g in self.chunk_gets()])
        self.assertEqual(len(self.chunk_gets()), 8)

    def test_a_one_stream_part_counts(self):
        self.part.write_bytes(self.data[:450_000])          # an older setup's .part: chunks 0-3 are whole
        setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertEqual([g[0] for g in self.chunk_gets()], [i * 100_000 for i in range(4, 11)])

    def test_no_ranges_one_stream(self):
        self.handler.ranges = False
        setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)

    def test_small_file_one_stream(self):
        with mock.patch.object(setup, "DL_PARALLEL_MIN", 2_000_000):
            setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertEqual([g for g in self.handler.gets if g != (0, 0)], [(0, len(self.data) - 1)])

    def test_one_connection_by_choice(self):
        with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
            setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertNotIn((0, 0), self.handler.gets)


if __name__ == "__main__":
    unittest.main()
