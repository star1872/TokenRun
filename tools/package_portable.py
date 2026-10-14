"""The ready-to-run TokenRun packages: Windows (TokenRun-windows-x64.zip, double-click TokenRun.exe) and Linux
(TokenRun-linux-x64.tar.gz, the `tokenrun` command; tools/portable/install.sh installs it in one command).

    python tools/package_portable.py [--out dist] [--python-archive FILE]

Built on the system it is for (Windows on Windows, Linux on Linux): the package's own Python installs the pinned
requirements, so the packages' wheels and entry points are the right ones.  Inside the package:

    TokenRun/
      TokenRun.exe  or  tokenrun     the start: the model manager page (app/serve/setup_web.py)
      python/                        a portable CPython (python-build-standalone) with requirements.txt installed
      app/                           this folder's files (git ls-files), the same code START-HERE.bat runs
      README.txt, LICENSE

So the PC needs no Python, pip or git: what the setup on a fresh PC downloads from python.org and PyPI (which stall
from mainland China, 2026-10-07) is already in the package.  The engines and the models are still downloaded by the
page, from ModelScope first.  setup.py's pip stamp (.strata-pip.json in the Python's prefix) lists the installed
requirements, so setup does not install them again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
# python-build-standalone (github.com/astral-sh/python-build-standalone), pinned with its published SHA-256
PBS_RELEASE = "20261003"
PBS_VERSION = "3.12.15"
PBS = {
    "windows": ("cpython-{v}+{r}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
                "6fba7f2ae506facf41d457ea8293c7497910a675c69a4e954875169410a50402"),
    "linux": ("cpython-{v}+{r}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
              "731af898886c5f821890dc901eca3c651cca8e51fa7308c159d12a1194aeac91"),
}
PBS_URL = "https://github.com/astral-sh/python-build-standalone/releases/download/{r}/{name}"
# the app's files a user does not need (benchmark data and the paper: about 3 MB)
LEAVE_OUT = ("bench/results/", "docs/paper/", "data/experimental-speed-projection/", ".github/")
CSC = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"

README = """TokenRun {version}

Windows: double-click TokenRun.exe. The model manager opens in your browser: pick a model, download it, start it,
then chat. Closing TokenRun's window stops the page.
Linux: run ./tokenrun (install.sh links it as `tokenrun`). On a server without a screen: tokenrun --host 0.0.0.0,
then open http://<this server>:8090/ from another PC.

Windows：双击 TokenRun.exe，浏览器会打开模型管理页：选模型、下载、启动，然后对话。关闭 TokenRun 窗口会停止该页面。
Linux：运行 ./tokenrun（install.sh 会把它装成 `tokenrun` 命令）。没有屏幕的服务器：tokenrun --host 0.0.0.0，
再从另一台电脑打开 http://<服务器地址>:8090/。

Models are downloaded into the Strata-data folder beside app/ unless you choose another folder on the page.
模型默认下载到 app 旁边的 Strata-data 文件夹，也可以在页面上改。

TokenRun is built on Strata by Niko1221 (https://github.com/Niko1221/Strata), MIT license: see LICENSE.
TokenRun: 星辰 (https://github.com/star1872/TokenRun). Questions: GitHub Issues. Business: cloud17778@outlook.com
TokenRun 作者：星辰（https://github.com/star1872/TokenRun）。使用问题请提 Issue；商务合作：cloud17778@outlook.com
The llama.cpp engine it downloads is ggml-org's (https://github.com/ggml-org/llama.cpp), MIT license.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def python_archive(system: str, given: str | None, cache: Path) -> Path:
    """The pinned portable Python: a given file (checked) or a download into the cache folder."""
    name, digest = PBS[system]
    name = name.format(v=PBS_VERSION, r=PBS_RELEASE)
    path = Path(given) if given else cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = PBS_URL.format(r=PBS_RELEASE, name=name)
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(path.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f)
        path.with_suffix(".part").replace(path)
    if sha256(path) != digest:
        sys.exit(f"{path}: not the pinned {name} (SHA-256 differs)")
    return path


def app_files() -> list[str]:
    """This folder's files as git sees them (tracked, and new ones not ignored), less LEAVE_OUT."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8").split("\0")
    return sorted(f for f in out if f and not f.startswith(LEAVE_OUT) and (ROOT / f).is_file())


def requirement_lines() -> list[str]:
    """requirements.txt less the build tools, as setup.py installs them (the stamp must list the same strings)."""
    sys.path.insert(0, str(ROOT))
    import setup                                       # its tables only: no side effects on import
    return setup.runtime_requirements()


def version() -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "dev"


def build(system: str, out: Path, given: str | None) -> Path:
    stage = out / "TokenRun"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    archive = python_archive(system, given, out / "cache")
    print(f"unpacking {archive.name}", flush=True)
    with tarfile.open(archive) as t:                   # its one top folder is python/
        t.extractall(stage, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(stage)
    py = stage / "python" / ("python.exe" if system == "windows" else "bin/python3")
    reqs = requirement_lines()
    req_file = out / "requirements-runtime.txt"
    req_file.write_text("".join(r + "\n" for r in reqs), encoding="utf-8")
    print("installing requirements.txt (less cmake, ninja) into the package's Python", flush=True)
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--disable-pip-version-check",
                    "--no-warn-script-location", "-r", str(req_file)], check=True)
    prefix = subprocess.run([str(py), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True,
                            check=True).stdout.strip()
    (Path(prefix) / ".strata-pip.json").write_text(json.dumps(sorted(reqs), indent=0))
    for d in (stage / "python").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)

    files = app_files()
    print(f"copying {len(files)} files into app/", flush=True)
    for f in files:
        dst = stage / "app" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
    (stage / "README.txt").write_text(README.format(version=version()), encoding="utf-8", newline="\r\n" if
                                      system == "windows" else "\n")

    if system == "windows":
        if not CSC.exists():
            sys.exit(f"{CSC} not found: the launcher is built with .NET Framework 4's C# compiler")
        subprocess.run([str(CSC), "/nologo", "/optimize+", "/codepage:65001", "/target:exe", f"/out:{stage / 'TokenRun.exe'}",
                        str(ROOT / "tools" / "portable" / "TokenRun.cs")], check=True)
        pkg = out / "TokenRun-windows-x64.zip"
        print(f"writing {pkg.name}", flush=True)
        with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, Path("TokenRun") / p.relative_to(stage))
    else:
        shutil.copy2(ROOT / "tools" / "portable" / "tokenrun", stage / "tokenrun")
        (stage / "tokenrun").chmod(0o755)
        pkg = out / "TokenRun-linux-x64.tar.gz"
        print(f"writing {pkg.name}", flush=True)
        with tarfile.open(pkg, "w:gz", compresslevel=6) as t:
            t.add(stage, arcname="TokenRun")
    print(f"{pkg}: {pkg.stat().st_size / 1e6:.0f} MB, SHA-256 {sha256(pkg)}", flush=True)
    return pkg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "dist"), help="where the package is written (default dist/)")
    ap.add_argument("--python-archive", help="the pinned python-build-standalone archive, if already downloaded")
    a = ap.parse_args(argv)
    system = "windows" if WIN else "linux" if sys.platform.startswith("linux") else None
    if system is None:
        sys.exit("packages are built on Windows or Linux")
    build(system, Path(a.out).resolve(), a.python_archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
