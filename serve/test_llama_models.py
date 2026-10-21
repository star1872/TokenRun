"""serve/llama_models.py (any GGUF model through llama.cpp) and its part of the model manager (serve/setup_web.py):
which official llama.cpp build a PC takes, whether a size runs here, the names and configs it writes, and the
manager's checks on what a page may ask for.  No network: the hub listing is replaced.

Run: python -m unittest serve.test_llama_models
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import llama_models as LM  # noqa: E402
from serve import setup_web as W  # noqa: E402


class Engine(unittest.TestCase):
    def test_kind_by_driver(self):
        self.assertEqual(LM.engine_kind([{"vendor": "nvidia", "driver": "610.47"}]), "cuda13")
        self.assertEqual(LM.engine_kind([{"vendor": "nvidia", "driver": "580.65"}]), "cuda13")
        self.assertEqual(LM.engine_kind([{"vendor": "nvidia", "driver": "552.22"}]), "cuda12")
        self.assertEqual(LM.engine_kind([{"vendor": "amd", "driver": "amdgpu"}]), "vulkan")
        self.assertEqual(LM.engine_kind([]), "cpu")

    def test_assets_are_the_published_names(self):
        # names as ggml-org publishes them (release b11457, checked 2026-10-07); the CUDA runtime comes apart
        with mock.patch.object(LM, "WIN", True):
            self.assertEqual(LM.engine_assets("cuda13", "b11457"),
                             ["llama-b11457-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"])
            self.assertEqual(LM.engine_assets("cpu", "b11457"), ["llama-b11457-bin-win-cpu-x64.zip"])
        with mock.patch.object(LM, "WIN", False):
            self.assertEqual(LM.engine_assets("cuda12", "b11457"),
                             ["llama-b11457-bin-ubuntu-cuda-12.8-x64.tar.gz",
                              "cudart-llama-b11457-bin-ubuntu-cuda-12.8-x64.tar.gz"])

    def test_modelscope_copy_first_in_mainland_china(self):
        with mock.patch.object(LM.S, "model_source", lambda: "modelscope"):
            urls = LM.asset_urls("x.zip", "b1")
        self.assertIn("modelscope", urls[0])
        self.assertTrue(urls[0].endswith("/llama.cpp/b1/x.zip"))
        self.assertTrue(urls[1].startswith("https://github.com/ggml-org/llama.cpp/releases/download/b1/"))
        with mock.patch.object(LM.S, "model_source", lambda: "huggingface"):
            self.assertIn("github.com", LM.asset_urls("x.zip", "b1")[0])


    def test_cuda_runtime_beside_the_server(self):
        # the Linux archives: llama-<b>/ with llama-server and libggml-cuda.so, cudart-...-x64/ with the runtime
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "llama-b1").mkdir()
            (root / "llama-b1" / "libggml-cuda.so").write_bytes(b"x")
            (root / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64").mkdir()
            for n in ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "README.md"):
                (root / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64" / n).write_bytes(b"x")
            moved = LM.gather_runtime(root, root / "llama-b1")
            self.assertEqual(sorted(moved), ["libcublas.so.13", "libcublasLt.so.13", "libcudart.so.13"])
            self.assertTrue((root / "llama-b1" / "libcudart.so.13").exists())
            self.assertTrue((root / "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64" / "README.md").exists())

    def test_disk_need_counts_only_what_is_left(self):
        # an install run again (or resumed) needs no room for the files it already has (llmserver, 2026-10-07:
        # 22 GB there, 12 GB free, and the install stopped at "need ~24 GB")
        v = {"files": [{"path": "a/m-00001.gguf", "size": 3000}, {"path": "a/m-00002.gguf", "size": 2000},
                       {"path": "a/m-00003.gguf", "size": 1000}]}
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            (dest / "m-00001.gguf").write_bytes(b"x" * 3000)          # finished
            (dest / "m-00002.gguf.part").write_bytes(b"x" * 500)       # half-way
            self.assertAlmostEqual(LM.left_gb(v, dest) * 1e9, 2500)

    def test_an_archive_put_there_by_hand_is_used(self):
        # GitHub's downloads time out from mainland China (llmserver, 2026-10-07): an archive copied into the engine
        # folder is unpacked without asking the network
        import io
        import tarfile
        import zipfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(LM, "ROOT", Path(d)), \
                mock.patch.object(LM.S, "reachable", lambda *a, **k: self.fail("asked the network")), \
                mock.patch.object(LM.S, "download", lambda *a, **k: self.fail("downloaded")), \
                mock.patch.object(LM.S, "ok", lambda *a: None), mock.patch.object(LM.S, "say", lambda *a: None):
            folder = LM.engine_dir("cpu", "b1")
            folder.mkdir(parents=True)
            (asset,) = LM.engine_assets("cpu", "b1")
            if asset.endswith(".zip"):
                with zipfile.ZipFile(folder / asset, "w") as z:
                    z.writestr(LM.SERVER_EXE, b"x")
            else:
                with tarfile.open(folder / asset, "w:gz") as t:
                    info = tarfile.TarInfo(f"llama-b1/{LM.SERVER_EXE}")
                    info.size = 1
                    t.addfile(info, io.BytesIO(b"x"))
            exe = LM.get_engine([], "b1")
            self.assertEqual(exe.name, LM.SERVER_EXE)
            self.assertFalse((folder / asset).exists())


class Fit(unittest.TestCase):
    def test_where_it_goes(self):
        self.assertEqual(LM.fit(10, 64, 16)["where"], "gpu")
        v = LM.fit(22, 64, 16)
        self.assertEqual((v["level"], v["where"]), ("ok", "gpu+ram"))
        self.assertEqual(LM.fit(60, 64, 16)["level"], "slow")            # needs most of the RAM
        v = LM.fit(120, 64, 16)
        self.assertEqual((v["level"], v["where"]), ("no", "ssd"))
        self.assertEqual(v["need_ram"], round(120 - (16 - LM.VRAM_LEFT_GB) + LM.RAM_LEFT_GB))

    def test_no_gpu(self):
        self.assertEqual(LM.fit(0.6, 12, 0), {"level": "slow", "where": "cpu"})
        self.assertEqual(LM.fit(10, 12, 0)["where"], "ssd")


def tiny_gguf(path: Path, tensors: list[tuple[str, int]]) -> None:
    """A GGUF with only a tensor table: (name, bytes) laid out one after another (no metadata, no real weights)."""
    import struct
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    offset = 0
    for name, size in tensors:
        n = name.encode()
        head += struct.pack("<Q", len(n)) + n + struct.pack("<IQIQ", 1, size, 0, offset)
        offset += size
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"\0" * offset)


class MoeCache(unittest.TestCase):
    def test_only_for_a_moe_model_bigger_than_the_vram(self):
        self.assertEqual(LM.moe_cache_mib("cuda13", 22.4, 21.0, 16), 8192)       # llmserver: half of 16 GB
        self.assertEqual(LM.moe_cache_mib("cuda12", 22.4, 21.0, 8), 4096)
        self.assertEqual(LM.moe_cache_mib("cuda13", 22.4, 3.0, 16), 3072)        # at most the experts' size
        self.assertEqual(LM.moe_cache_mib("cuda13", 10, 9, 16), 0)               # fits in VRAM
        self.assertEqual(LM.moe_cache_mib("cuda13", 30, 0, 16), 0)               # dense
        self.assertEqual(LM.moe_cache_mib("vulkan", 22.4, 21, 16), 0)            # measured on CUDA only
        self.assertEqual(LM.moe_cache_mib("cuda13", 22.4, 21, 0), 0)             # no GPU
        self.assertEqual(LM.run_args(32768, "cuda13", 22.4, 21, 16),
                         ["-c", "32768", "--fit", "on", "--moe-cache-mib", "8192"])
        self.assertEqual(LM.run_args(8192, "cuda13", 10, 9, 16), ["-c", "8192", "--fit", "on"])

    def test_expert_bytes_from_the_headers(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.gguf"
            tiny_gguf(f, [("token_embd.weight", 4096), ("blk.0.ffn_gate_exps.weight", 8192),
                          ("blk.0.ffn_down_exps.weight", 8192), ("blk.0.attn_q.weight", 1024)])
            self.assertEqual(LM.expert_gb([f]), 16384 / 1e9)
            self.assertEqual(LM.expert_gb([Path(d) / "missing.gguf"]), 0)

    def test_an_older_config_moves_up_once(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "M-Q4-00001-of-00002.gguf"
            tiny_gguf(m, [("blk.0.ffn_up_exps.weight", 1 << 20)])
            tiny_gguf(Path(d) / "M-Q4-00002-of-00002.gguf", [("blk.1.ffn_up_exps.weight", 1 << 20)])
            p = Path(d) / "llama-m.json"
            p.write_text(json.dumps({"backend": "llamacpp", "exe": "old/llama-server", "model": str(m), "gb": 22.4,
                                     "args": ["-c", "65536", "--fit", "on", "-t", "8"], "llama_build": "b11457"}))
            gpus = [{"name": "RTX 4060 Ti", "vram_gb": 16, "driver": "580.1", "vendor": "nvidia"}]
            with mock.patch.object(LM, "gpus_here", lambda: gpus), \
                    mock.patch.object(LM, "get_engine", lambda g: Path("new/llama-server")), \
                    mock.patch.object(LM, "expert_gb", lambda files: 21.0 if len(files) == 2 else 0):
                cfg = LM.refresh_config(p)
                self.assertEqual(cfg["args"], ["-c", "65536", "--fit", "on", "-t", "8", "--moe-cache-mib", "8192"])
                self.assertEqual((cfg["llama_build"], Path(cfg["exe"])), (LM.LLAMA_BUILD, Path("new/llama-server")))
                self.assertEqual(json.loads(p.read_text()), cfg)
                self.assertEqual(LM.refresh_config(p), cfg)                     # nothing more to do

    def test_a_failed_engine_download_keeps_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "llama-m.json"
            old = {"backend": "llamacpp", "exe": "old/llama-server", "model": str(Path(d) / "m.gguf"),
                   "args": ["-c", "32768", "--fit", "on"], "llama_build": "b11457"}
            p.write_text(json.dumps(old))
            def offline(g):
                raise SystemExit(1)
            with mock.patch.object(LM, "gpus_here", lambda: []), mock.patch.object(LM, "get_engine", offline):
                self.assertEqual(LM.refresh_config(p), old)
            self.assertEqual(json.loads(p.read_text()), old)


class Names(unittest.TestCase):
    def test_key(self):
        self.assertEqual(LM.model_key("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL"),
                         "qwen3.6-35b-a3b-ud-q4_k_xl")
        self.assertEqual(LM.model_key("unsloth/Qwen3.6-35B-A3B-GGUF", "BF16/Qwen3.6-35B-A3B-BF16"),
                         "qwen3.6-35b-a3b-bf16")
        self.assertEqual(LM.model_key("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0"), "qwen3-0.6b-q8_0")

    def test_variants_group_shards_and_skip_projectors(self):
        files = [LM.QS.Source(n, s) for n, s in (("BF16/M-BF16-00001-of-00002.gguf", 5e9), ("BF16/M-BF16-00002-of-00002.gguf", 5e9),
                                                 ("M-Q4_K_M.gguf", 3e9), ("mmproj-F16.gguf", 1e9), ("README.md", 1),
                                                 ("imatrix_unsloth.gguf", 1e7), ("mtp-M-Q4_0.gguf", 1e9))]
        with mock.patch.object(LM, "hub_files", lambda repo, source: files):
            vs = LM.variants("o/M-GGUF", "modelscope")
        self.assertEqual([(v["label"], len(v["files"]), v["gb"]) for v in vs], [("M-Q4_K_M", 1, 3.0), ("M-BF16", 2, 10.0)])

    def test_search_keeps_gguf_and_puts_known_publishers_first(self):
        # the shapes of ModelScope's /openapi/v1/models and Hugging Face's /api/models (checked 2026-10-20)
        ms = {"data": {"models": [{"id": "someone/Qwen3.8-27B-GGUF", "downloads": 900, "tags": ["library:gguf"]},
                                  {"id": "Qwen/Qwen3.8-27B-FP8", "downloads": 5000, "tags": []},
                                  {"id": "unsloth/Qwen3.8-27B-GGUF", "downloads": 400, "tags": ["library:gguf"]}]}}
        with mock.patch.object(LM.QS, "http", lambda url: json.dumps(ms).encode()):
            self.assertEqual([x["repo"] for x in LM.search("Qwen3.8-27B", "modelscope")],
                             ["unsloth/Qwen3.8-27B-GGUF", "someone/Qwen3.8-27B-GGUF"])
        hf = [{"id": "a/X-GGUF", "downloads": 10}, {"id": "b/X-GGUF", "downloads": 20}]
        with mock.patch.object(LM.QS, "http", lambda url: json.dumps(hf).encode()):
            self.assertEqual([x["repo"] for x in LM.search("X", "huggingface")], ["b/X-GGUF", "a/X-GGUF"])


class Installed(unittest.TestCase):
    def test_only_llama_configs(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(LM, "ROOT", Path(d)):
            (Path(d) / "llama-a.json").write_text(json.dumps({"backend": "llamacpp", "model_name": "a", "port": 8081,
                                                              "repo": "o/r", "variant": "v", "gb": 1.5}))
            (Path(d) / "llama-b.json").write_text("{not json")
            (Path(d) / "llama-c.json").write_text(json.dumps({"backend": "other"}))
            self.assertEqual(list(LM.installed()), ["a"])
            self.assertEqual(LM.installed()["a"]["port"], 8081)


class Manager(unittest.TestCase):
    def test_repo_and_variant_are_checked(self):
        for repo in ("../x", "a/b/c", "a b/c", ""):
            code, body = W.api("POST", "/manage/api/llama/install", json.dumps({"repo": repo, "variant": "v"}).encode())
            self.assertEqual(code, 400, repo)
        code, _ = W.api("POST", "/manage/api/llama/install", json.dumps({"repo": "o/r", "variant": "v\n--x"}).encode())
        self.assertEqual(code, 400)
        code, _ = W.api("GET", "/manage/api/llama/variants?repo=..%2Fx", None)
        self.assertEqual(code, 400)
        code, _ = W.api("GET", "/manage/api/llama/search?q=", None)
        self.assertEqual(code, 400)

    def test_install_runs_the_installer(self):
        with mock.patch.object(W, "Job", lambda kind, cmd, what: mock.Mock(cmd=cmd, what=what, code=None,
                                                                           state=lambda: {"kind": kind})):
            W.JOBS.clear()
            code, _ = W.api("POST", "/manage/api/llama/install",
                            json.dumps({"repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "variant": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
                                        "source": "modelscope", "context": 16384}).encode())
            job = W.JOBS.pop("install")
        self.assertEqual(code, 200)
        self.assertEqual(job.cmd[1:], [str(W.ROOT / "serve" / "llama_models.py"), "install",
                                       "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL", "--source",
                                       "modelscope", "--context", "16384"])

    def test_download_folder(self):
        self.assertIsNone(W.data_dir_arg(""))
        for bad in ("models", "-x", "/data/a\nb", "../x"):
            with self.assertRaises(ValueError, msg=bad):
                W.data_dir_arg(bad)
        with tempfile.TemporaryDirectory() as d:
            info = W.folder_info(str(Path(d) / "new" / "models"))
            self.assertEqual((info["exists"], info["writable"]), (False, True))
            self.assertGreater(info["free_gb"], 0)
            with mock.patch.object(W, "Job", lambda kind, cmd, what: mock.Mock(cmd=cmd, code=None, state=dict)):
                W.JOBS.clear()
                W.api("POST", "/manage/api/llama/install", json.dumps({"repo": "o/r", "variant": "v", "data_dir": d}).encode())
                llama_cmd = W.JOBS.pop("install").cmd
                W.api("POST", "/manage/api/install", json.dumps({"family": "qwen", "model": "Q2_0", "data_dir": d}).encode())
                strata_cmd = W.JOBS.pop("install").cmd
        self.assertEqual(llama_cmd[-2:], ["--data-dir", str(Path(d).resolve())])
        self.assertEqual(strata_cmd[-2:], ["--data-dir", str(Path(d).resolve())])

    def test_start_a_llama_model(self):
        with mock.patch.object(LM, "installed", lambda: {"m": {"config": "llama-m.json", "model_name": "m", "port": 8085}}), \
                mock.patch.object(W, "Job", lambda kind, cmd, what: (kind, cmd, what)):
            kind, cmd, what = W.start_model("llama/m")
        self.assertEqual(kind, "start")
        self.assertEqual(cmd[1:], [str(W.ROOT / "serve" / "llama_front.py"), "--config", str(W.ROOT / "llama-m.json"),
                                   "--port", "8085"])
        with self.assertRaises(ValueError):
            W.start_model("llama/none")


if __name__ == "__main__":
    unittest.main()
