"""TokenRun's engine build on GitHub Actions (.github/workflows/oldhw-engine.yml): its steps outside CMake.  The
engine is the official Strata source at setup.py's ENGINE_VERSION (checked out into ENGINE_SRC); without ENGINE_SRC
this checkout's own source.

    python tools/ci_package.py engine-version
        setup.py's ENGINE_VERSION: the official Strata tag (v<it>) the engine job checks out and builds.
    python tools/ci_package.py llama
        llama.cpp at setup.py's pinned commit (LLAMA_CPP_COMMIT) into third_party/llama.cpp, as setup.py unpacks it
        (without tools/ui, whose deep paths pass Windows' 260-character limit).
    python tools/ci_package.py zip --exe build/strata.exe --vision build-vision/bin/strata-vision.exe --out X.zip
        the release archive setup.py downloads (get_prebuilt): the engine, the image encoder and BUILD.json, whose
        version, archs, ptx and cuda setup checks before it uses the engine.
    python tools/ci_package.py modelscope dist/*.zip
        a copy of the release archives on ModelScope (www.modelscope.cn), which mainland China reaches when GitHub's
        downloads do not: the model repository MODELSCOPE_REPO (default setup.py's MS_ENGINE_REPO), each archive at
        engine-v<version>/<name>, with the token MODELSCOPE_TOKEN (a write token from modelscope.cn/my/myaccesstoken).
        Without the token it says so and does nothing (forks without it still build).
    python tools/ci_package.py llama-mirror
        the official llama.cpp builds the llama.cpp backend installs (serve/llama_models.py LLAMA_BUILD, every
        archive engine_assets() names for Windows and Linux) copied from ggml-org's GitHub release to the same
        ModelScope repository as llama.cpp/<build>/<archive>; archives already there are skipped.
    python tools/ci_package.py tokenrun dist/TokenRun-windows-x64.zip dist/TokenRun-linux-x64.tar.gz install.sh
        TokenRun's ready-to-run packages (tools/package_portable.py) at the top of the ModelScope repository
        TOKENRUN_MS_REPO (default mymodel3861/TokenRun, created when missing), with its README.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pinned_commit() -> str:
    """setup.py's LLAMA_CPP_COMMIT, read from the file (importing setup.py is not needed for one string)."""
    m = re.search(r'^LLAMA_CPP_COMMIT = "([0-9a-f]{40})"', (ROOT / "setup.py").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("LLAMA_CPP_COMMIT not found in setup.py")
    return m.group(1)


def setup_value(name: str) -> str:
    """A string constant of setup.py (NAME = "..."), read from the file."""
    m = re.search(rf'^{name} = "([^"]+)"', (ROOT / "setup.py").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"{name} not found in setup.py")
    return m.group(1)


def engine_src() -> Path:
    return Path(os.environ.get("ENGINE_SRC") or ROOT)


def version() -> str:
    """The version of the engine source that is built (its CMakeLists.txt)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (engine_src() / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1)


def llama() -> None:
    commit = pinned_commit()
    dst = ROOT / "third_party" / "llama.cpp"
    url = f"https://github.com/ggml-org/llama.cpp/archive/{commit}.zip"
    print(f"llama.cpp {commit[:7]} <- {url}", flush=True)
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata-ci"}), timeout=600).read()
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp, [m for m in z.namelist() if "/tools/ui/" not in m])
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(next(tmp.iterdir())), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"-> {dst}")


def cuda_version() -> str:
    """nvcc's release (e.g. "13.0"), as BUILD.json's "cuda"."""
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    except OSError:                                     # no nvcc on the PATH: say so instead of guessing
        return "?"
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else "?"


def package(exe: Path, vision: Path | None, out: Path) -> None:
    archs_spec = os.environ.get("CUDA_ARCHS", "")
    archs = sorted({int(re.match(r"\d+", a).group()) for a in archs_spec.split(";") if re.match(r"\d+", a)})
    meta = {"version": version(), "source": "release", "archs": archs,
            # an architecture without "-real" carries its PTX too: newer cards JIT-compile it (setup's "ptx" check)
            "ptx": any(not a.endswith("-real") for a in archs_spec.split(";") if a),
            "cuda": cuda_version(), "vision": "gpu" if vision else "none", "portable": True,
            "fork": "tokenrun", "strata_ref": os.environ.get("STRATA_REF", ""),
            "commit": os.environ.get("GITHUB_SHA", "")[:12]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(exe, exe.name)
        if vision:
            z.write(vision, vision.name)
        z.writestr("BUILD.json", json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


MS_README = """# Strata engine (TokenRun, star1872/TokenRun)

A copy of the ready-made engine archives of https://github.com/star1872/TokenRun/releases (release engine-v<version>),
for PCs that reach ModelScope but not GitHub's downloads.  setup.py downloads them from here by itself; there is no
need to download them by hand.  Built by GitHub Actions from the official Strata source of that version
(STRATA_PORTABLE, CUDA 13.0, RTX 20/30/40/50); each archive's BUILD.json names the tag.  oldhw-v<version>/: the
engines of TokenRun before 2026-10-14 (Strata 0.1.39 with the old-PC changes), kept for older installs.

Strata: https://github.com/Niko1221/Strata (MIT)

llama.cpp/<build>/: the official llama.cpp builds (https://github.com/ggml-org/llama.cpp/releases, MIT; each archive
carries its license) for the fork's llama.cpp backend (serve/llama_models.py), copied unchanged.
"""


def modelscope(files: list[Path]) -> None:
    token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
    repo = os.environ.get("MODELSCOPE_REPO", "").strip() or setup_value("MS_ENGINE_REPO")
    if not token:
        print("MODELSCOPE_TOKEN is not set: no copy on ModelScope")
        return
    from modelscope_hub.api import HubApi            # pip install modelscope-hub (CI only; setup.py does not need it)
    api = HubApi(token=token)
    api.login(token)
    if not api.repo_exists(repo, "model"):
        api.create_repo(repo, "model", visibility="public", license="mit",
                        description="Ready-made Strata engine (TokenRun, star1872/TokenRun)")
        print(f"created {repo}")
    api.upload_file(repo, "model", MS_README.encode(), "README.md", commit_message="README", disable_tqdm=True)
    tag = f"engine-v{setup_value('ENGINE_VERSION')}"
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    for f in files:
        api.upload_file(repo, "model", str(f), f"{tag}/{f.name}", commit_message=f"{tag} {f.name} ({sha})",
                        disable_tqdm=True)
        print(f"-> {repo}/{tag}/{f.name} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


def ms_api():
    token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
    repo = os.environ.get("MODELSCOPE_REPO", "").strip() or setup_value("MS_ENGINE_REPO")
    if not token:
        print("MODELSCOPE_TOKEN is not set: no copy on ModelScope")
        return None, None
    from modelscope_hub.api import HubApi
    api = HubApi(token=token)
    api.login(token)
    return api, repo


def llama_assets() -> tuple[str, list[str]]:
    """The pinned llama.cpp build and every archive serve/llama_models.py may download, Windows and Linux."""
    import sys
    sys.path.insert(0, str(ROOT))
    from serve import llama_models as LM
    names = []
    for win in (True, False):
        LM.WIN = win
        for kind in ("cuda13", "cuda12", "vulkan", "cpu"):
            names += [n for n in LM.engine_assets(kind) if n not in names]
    return LM.LLAMA_BUILD, names


def llama_mirror() -> None:
    api, repo = ms_api()
    if api is None:
        return
    build, names = llama_assets()
    have = {f.path: f.size for f in api.list_repo_files(repo, "model", recursive=True)}
    api.upload_file(repo, "model", MS_README.encode(), "README.md", commit_message="README", disable_tqdm=True)
    for name in names:
        path = f"llama.cpp/{build}/{name}"
        if have.get(path):
            print(f"= {path} (already there)", flush=True)
            continue
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{build}/{name}"
        tmp = ROOT / "_mirror.tmp"
        req = urllib.request.Request(url, headers={"User-Agent": "strata-ci"})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 8 << 20)
        api.upload_file(repo, "model", str(tmp), path, commit_message=f"llama.cpp {build} {name}", disable_tqdm=True)
        print(f"-> {repo}/{path} ({tmp.stat().st_size / 1e6:.1f} MB)", flush=True)
        tmp.unlink()


TOKENRUN_MS_README = """---
license: mit
---
# TokenRun

在自己的电脑上运行大模型：中文网页、模型管理、Strata 和 llama.cpp 两种引擎。基于
[Niko1221/Strata](https://github.com/Niko1221/Strata)（MIT）。源码：https://github.com/star1872/TokenRun。

## Windows

下载 [TokenRun-windows-x64.zip](https://www.modelscope.cn/models/{repo}/resolve/master/TokenRun-windows-x64.zip)，
整个解压，双击 `TokenRun.exe`。浏览器会打开模型管理页：选模型、下载、启动，然后对话。
电脑上不需要装 Python 或 git。

## Linux（x86_64）

```bash
curl -fsSL https://www.modelscope.cn/models/{repo}/resolve/master/install.sh | sh
tokenrun
```

不用 sudo，装在 `~/.local/share/tokenrun`。没有屏幕的服务器：`tokenrun --host 0.0.0.0`，再从另一台电脑打开
`http://<服务器地址>:8090/`。再运行一次安装命令就是更新，已下载的模型和引擎会保留。

引擎和模型由页面从魔搭下载。本仓库的文件由 GitHub Actions 打包上传（{sha}）。

## 作者与联系方式

TokenRun 由 **星辰**（GitHub [star1872](https://github.com/star1872)）开发并开源，引擎和原始代码来自 [Niko1221/Strata](https://github.com/Niko1221/Strata)。

- 使用问题：请在 GitHub 提 [Issue](https://github.com/star1872/TokenRun/issues)。个人精力有限，不提供一对一的免费技术支持。
- 商务合作、定制开发：邮箱 cloud17778@outlook.com，或加微信（请注明来意）：

<img src="https://www.modelscope.cn/models/{repo}/resolve/master/wechat-qr.png" width="273" alt="星辰的微信二维码">
"""


def tokenrun(files: list[Path]) -> None:
    """The ready-to-run packages (tools/package_portable.py) and install.sh at the top of the TokenRun repository on
    ModelScope (TOKENRUN_MS_REPO), the links its README and install.sh give."""
    token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
    repo = os.environ.get("TOKENRUN_MS_REPO", "").strip() or "mymodel3861/TokenRun"
    if not token:
        print("MODELSCOPE_TOKEN is not set: no copy on ModelScope")
        return
    from modelscope_hub.api import HubApi
    api = HubApi(token=token)
    api.login(token)
    if not api.repo_exists(repo, "model"):
        api.create_repo(repo, "model", visibility="public", license="mit",
                        description="TokenRun: run LLMs on your own PC (Windows exe, Linux one command)")
        print(f"created {repo}")
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    readme = TOKENRUN_MS_README.replace("{repo}", repo).replace("{sha}", sha or "local")
    api.upload_file(repo, "model", readme.encode(), "README.md", commit_message="README", disable_tqdm=True)
    qr = ROOT / "docs" / "media" / "wechat-qr.png"                # the README's contact picture
    if qr.exists():
        api.upload_file(repo, "model", str(qr), qr.name, commit_message="contact picture", disable_tqdm=True)
    for f in files:
        api.upload_file(repo, "model", str(f), f.name, commit_message=f"{f.name} ({sha})", disable_tqdm=True)
        print(f"-> {repo}/{f.name} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["engine-version", "llama", "zip", "modelscope", "llama-mirror", "tokenrun"])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--exe", type=Path)
    ap.add_argument("--vision", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.step == "engine-version":                    # the official tag the engine job builds (v<this>)
        print(setup_value("ENGINE_VERSION"))
    elif a.step == "llama":
        llama()
    elif a.step == "llama-mirror":
        llama_mirror()
    elif a.step == "modelscope":
        modelscope(a.files)
    elif a.step == "tokenrun":
        tokenrun(a.files)
    else:
        if not (a.exe and a.out):
            ap.error("zip needs --exe and --out")
        package(a.exe, a.vision, a.out)


if __name__ == "__main__":
    main()
