"""Tests for setup.py's reproducible installs (#214): the engine of the checkout's own release first (the latest as
the fallback), Hugging Face files at pinned revisions (the current files when a revision is gone), the pinned
requirements file, and an existing install left as it is.  Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_pins
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def not_found(url):
    return urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class HuggingFacePins(unittest.TestCase):
    def test_every_model_url_is_pinned(self):
        for name, fam in setup.FAMILIES.items():
            for key in ("hf", "mmproj_hf"):
                with self.subTest(family=name, key=key):
                    self.assertRegex(fam[key], SHA)
                    self.assertNotIn("/resolve/main/", fam[key])
        self.assertRegex(setup.HF, SHA)

    def test_unpinned(self):
        url = setup.FAMILIES["swift"]["hf"] + "x.gguf"
        self.assertEqual(setup.hf_unpinned(url),
                         "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/x.gguf")
        self.assertEqual(setup.hf_unpinned("https://example.com/a/b"), "https://example.com/a/b")

    def test_a_gone_revision_downloads_the_current_file(self):
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            if "/resolve/main/" not in req.full_url:
                raise not_found(req.full_url)
            return Response(b"model bytes", status=200)

        # the pinned Hugging Face path itself (setup's --source huggingface; auto may pick ModelScope)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}):
            dst = Path(d) / "m.gguf"
            _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
        self.assertIn("not at the pinned revision any more", out)
        self.assertEqual([m for m, _ in seen], ["HEAD", "HEAD", "GET"])
        self.assertTrue(all("/resolve/main/" in u for _, u in seen[1:]))

    def test_mtp_fetch_is_pinned_and_falls_back(self):
        import mtp_fetch
        self.assertRegex(mtp_fetch.REPO, SHA)
        self.assertIn(mtp_fetch.REVISION, mtp_fetch.REPO)
        pinned = mtp_fetch.REPO

        def urlopen(req, timeout=None):
            raise not_found(req.full_url)

        try:
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mtp_fetch.resolve_repo(), "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/")
            self.assertIn("pinned revision", err.getvalue())
        finally:
            mtp_fetch.REPO = pinned

    def test_hf_endpoint(self):
        # #495: HF_ENDPOINT (a mirror) serves the same pinned revision; a trailing slash and blanks are dropped
        repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        with mock.patch.dict(setup.os.environ, {}, clear=False):
            setup.os.environ.pop("HF_ENDPOINT", None)
            self.assertEqual(setup.hf(repo), f"https://huggingface.co/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
            for value in ("https://hf-mirror.com", "https://hf-mirror.com/", " https://hf-mirror.com/ "):
                setup.os.environ["HF_ENDPOINT"] = value
                url = setup.hf(repo)
                self.assertEqual(url, f"https://hf-mirror.com/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
                self.assertRegex(url, SHA)
                self.assertEqual(setup.hf_unpinned(url + "x.gguf"), f"https://hf-mirror.com/{repo}/resolve/main/x.gguf")
            setup.os.environ["HF_ENDPOINT"] = ""
            self.assertEqual(setup.hf_endpoint(), "https://huggingface.co")

    def test_mtp_fetch_honours_hf_endpoint(self):
        import importlib
        import mtp_fetch
        try:
            with mock.patch.dict(mtp_fetch.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}):
                importlib.reload(mtp_fetch)
                self.assertTrue(mtp_fetch.REPO.startswith("https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/"))
                self.assertRegex(mtp_fetch.REPO, SHA)
                self.assertTrue(mtp_fetch.pinned())                 # the tensors' SHA-256 checks still apply
        finally:
            importlib.reload(mtp_fetch)
        self.assertTrue(mtp_fetch.REPO.startswith("https://huggingface.co/"))

    def test_step5_names_the_folder(self):
        # #495: where setup expects the model files, and how to give it files downloaded by hand
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["96GB-1x16GB"]
        with mock.patch.dict(setup.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
            code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "IQ3_XXS", "--no-start"])
        self.assertEqual(code, 0)
        text = out.replace("\\", "/")
        self.assertRegex(text, r"The model files go in .*/models/IQ3_XXS")
        self.assertIn("put them here with their original names (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                      text)
        self.assertIn("--gguf-dir", text)
        self.assertIn("Downloading from https://hf-mirror.com (HF_ENDPOINT)", text)


class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "engine").mkdir()
        self.patches = [mock.patch.object(setup, "ROOT", self.root),
                        mock.patch.object(setup, "source_version", lambda: "0.1.31"),
                        mock.patch.object(setup, "model_source", lambda: "huggingface")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def fake_download(self, got):
        def download(url, dst, what=None):
            got.append(url)
            with zipfile.ZipFile(dst, "w") as z:
                z.writestr("BUILD.json", json.dumps({"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [89]}))
                z.writestr(setup.EXE, b"engine")
        return download

    def run_get(self, published):
        heads, got = [], []

        def urlopen(req, timeout=None):
            heads.append(req.full_url)
            if not any(req.full_url.startswith(p) for p in published):
                raise not_found(req.full_url)
            return Response()

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup, "download", self.fake_download(got)):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        return eng, out, heads, got

    MS = "https://www.modelscope.cn/models/mymodel3861/Strata-oldhw-engine/resolve/master/oldhw-v0.1.31/"

    def test_bases(self):
        # the fork's own engine first (its ModelScope copy next), then the official release of the same version,
        # then the official latest
        gh = ["https://github.com/star1872/TokenRun/releases/download/oldhw-v0.1.31/",
              "https://github.com/Niko1221/Strata/releases/download/v0.1.31/", setup.UPSTREAM_PREBUILT_URL]
        self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL), gh[:1] + [self.MS] + gh[1:])
        # mainland China: the ModelScope copy first
        with mock.patch.object(setup, "model_source", lambda: "modelscope"):
            self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL), [self.MS] + gh)
        self.assertEqual(setup.prebuilt_bases("https://mirror.example/x"), ["https://mirror.example/x/"])

    def test_the_modelscope_copy_is_checked(self):
        def zip_sha():
            z = self.root / "engine" / setup.PREBUILT_ASSET
            self.fake_download([])(None, z)
            return setup.hashlib.sha256(z.read_bytes()).hexdigest()

        good = zip_sha()
        for sha, used in ((good, True), ("0" * 64, False)):
            with self.subTest(used=used):
                (self.root / "engine" / "BUILD.json").unlink(missing_ok=True)
                got = []
                with mock.patch.object(setup, "model_source", lambda: "modelscope"),                         mock.patch.object(setup, "ms_meta", lambda repo, path: (1, sha)),                         mock.patch.object(setup.urllib.request, "urlopen", lambda req, timeout=None: Response()),                         mock.patch.object(setup, "download", self.fake_download(got)):
                    eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
                self.assertEqual(got, [self.MS + setup.PREBUILT_ASSET])
                self.assertEqual(eng, self.root / "engine" if used else None)
                if not used:
                    self.assertIn("does not match its SHA-256", out)

    def test_the_fork_s_release_first(self):
        tag = "https://github.com/star1872/TokenRun/releases/download/oldhw-v0.1.31/"
        eng, out, heads, got = self.run_get([tag, setup.UPSTREAM_PREBUILT_URL])
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [tag + setup.PREBUILT_ASSET])
        self.assertEqual(len(heads), 1)

    def test_the_official_release_when_the_fork_has_none(self):
        tag = "https://github.com/Niko1221/Strata/releases/download/v0.1.31/"
        eng, out, heads, got = self.run_get([tag])
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [tag + setup.PREBUILT_ASSET])
        self.assertEqual(len(heads), 3)                # the fork's release and its ModelScope copy first

    def test_latest_when_it_is_not_published(self):
        eng, out, heads, got = self.run_get([setup.UPSTREAM_PREBUILT_URL])
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [setup.UPSTREAM_PREBUILT_URL + setup.PREBUILT_ASSET])
        self.assertIn("No ready-made engine for v0.1.31", out)

    def test_a_refused_archive_is_not_kept(self):
        """PR #324: a refused archive (too old, or no code for the GPU) kept its zip and .done mark, and every later
        run reused it ("already downloaded") instead of the published one."""
        for meta in ({"version": "0.1.0", "archs": [89]},
                     {"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [120]}):
            with self.subTest(meta=meta):
                def download(url, dst, what=None):
                    with zipfile.ZipFile(dst, "w") as z:
                        z.writestr("BUILD.json", json.dumps(meta))
                        z.writestr(setup.EXE, b"engine")
                    setup.mark(dst)

                with mock.patch.object(setup.urllib.request, "urlopen", lambda req, timeout=None: Response()), \
                        mock.patch.object(setup, "download", download):
                    eng, _ = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
                self.assertIsNone(eng)
                z = self.root / "engine" / setup.PREBUILT_ASSET
                self.assertFalse(z.exists())
                self.assertFalse(z.with_name(z.name + ".done").exists())

    def test_an_installed_engine_is_kept(self):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps(
            {"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [89]}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")

        def urlopen(req, timeout=None):
            raise AssertionError("asked the network for an installed engine")

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")


class Requirements(unittest.TestCase):
    def test_every_package_is_pinned(self):
        lines = setup.requirement_lines()
        names = {setup.req_name(x) for x in lines}
        for p in setup.PY_PACKAGES:
            self.assertIn(p, names)
        for x in lines:
            self.assertIn("==", x, x)
        self.assertEqual(setup.req_name('numpy==2.5.3; python_version >= "3.12"'), "numpy")
        self.assertEqual(setup.req_name("charset_normalizer==3"), "charset-normalizer")

    def pip(self, stamp, packages, installed=(), source="huggingface", fails=0):
        ran = []

        def run(cmd, check=True, **kw):                 # the first `fails` pip runs fail
            ran.append(cmd)
            code = 1 if len(ran) <= fails else 0
            if code and check:
                raise SystemExit(1)
            return SimpleNamespace(returncode=code)
        with tempfile.TemporaryDirectory() as d:
            if stamp is not None:
                (Path(d) / ".strata-pip.json").write_text(json.dumps(stamp))
            with mock.patch.object(setup.sys, "prefix", d), \
                    mock.patch.object(setup, "run", run), \
                    mock.patch.object(setup, "model_source", lambda: source), \
                    mock.patch.dict(setup.os.environ, {"PIP_INDEX_URL": ""}), \
                    mock.patch.object(setup, "_installed", lambda name: name in installed):
                quiet(setup.pip_install, packages, "the packages")
            after = json.loads((Path(d) / ".strata-pip.json").read_text()) if ran else stamp
        self.indexes = [cmd[cmd.index("--index-url") + 1] if "--index-url" in cmd else "pypi" for cmd in ran]
        return [c for cmd in ran[-1:] for c in cmd if "==" in c or c in setup.PY_PACKAGES], after

    def test_pypi_first_then_the_mirrors(self):
        lines = setup.requirement_lines()
        ran, stamp = self.pip(None, lines, fails=1)
        self.assertEqual(self.indexes, ["pypi", setup.PIP_MIRRORS[0]])
        self.assertEqual((ran, sorted(stamp)), (lines, sorted(lines)))

    def test_mirrors_first_in_china(self):                # ModelScope as the model source: pypi.org last
        lines = setup.requirement_lines()
        self.pip(None, lines, source="modelscope")
        self.assertEqual(self.indexes, setup.PIP_MIRRORS[:1])
        self.pip(None, lines, source="modelscope", fails=2)
        self.assertEqual(self.indexes, setup.PIP_MIRRORS + ["pypi"])

    def test_fresh_install_gets_every_pin(self):
        lines = setup.requirement_lines()
        ran, stamp = self.pip(None, lines)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertEqual(self.pip(stamp, lines)[0], [])                       # the second run: nothing

    def test_an_install_from_before_the_pins_is_left_alone(self):
        lines = setup.requirement_lines()
        legacy = sorted(setup.PY_PACKAGES) + ["nvidia-cublas==13.0.2.14"]
        deps = {"markupsafe", "certifi", "charset-normalizer", "idna", "urllib3", "colorama"}
        self.assertEqual(self.pip(legacy, lines, installed=deps)[0], [])
        missing = [p for p in legacy if p != "psutil"]                         # an older list without psutil
        ran, _ = self.pip(missing, lines, installed=deps)
        self.assertEqual(ran, ["psutil==7.2.2"])

    def test_a_changed_pin_is_installed(self):
        lines = setup.requirement_lines()
        old = [x.replace("tqdm==4.70.1", "tqdm==4.60.0") for x in lines]
        ran, _ = self.pip(old, lines)
        self.assertEqual(ran, ["tqdm==4.70.1"])


class Certificates(unittest.TestCase):
    """Windows: certifi's root certificates join the system's (a fresh cloud Windows failed github.com, 2026-10-11)."""

    def test_windows_sets_certifi(self):
        import certifi
        with mock.patch.object(setup.os, "name", "nt"), mock.patch.dict(setup.os.environ, {}, clear=True):
            setup._trust_certifi()
            self.assertEqual(setup.os.environ.get("SSL_CERT_FILE"), certifi.where())

    def test_own_setting_and_other_systems_kept(self):
        with mock.patch.object(setup.os, "name", "nt"), mock.patch.dict(setup.os.environ, {"SSL_CERT_FILE": "mine.pem"}):
            setup._trust_certifi()
            self.assertEqual(setup.os.environ["SSL_CERT_FILE"], "mine.pem")
        with mock.patch.object(setup.os, "name", "posix"), mock.patch.dict(setup.os.environ, {}, clear=True):
            setup._trust_certifi()
            self.assertNotIn("SSL_CERT_FILE", setup.os.environ)


if __name__ == "__main__":
    unittest.main()
