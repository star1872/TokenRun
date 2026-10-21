"""serve/llama_models.py - any GGUF model through llama.cpp: the engine (an official llama.cpp build), the model's
files (ModelScope or Hugging Face) and the run config + start script that serve/llama_front.py runs.

    python serve/llama_models.py variants unsloth/Qwen3.6-35B-A3B-GGUF [--source ms|hf]
    python serve/llama_models.py install unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_XL [--context 32768] [--source ms|hf]
    python serve/llama_models.py start <key>            (a model installed here: run-llama-<key>.bat / .sh does this)

The model manager (serve/setup_web.py) calls the same functions.  The Strata engine and its models (setup.py) are not
touched: these models live in Strata-data/llama/, their configs are llama-<key>.json beside the strata-*.json ones.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup as S  # noqa: E402
import quantscope as QS  # noqa: E402

WIN = os.name == "nt"
# the llama.cpp build these models run on: pinned, so a setup made today starts the same engine next month
LLAMA_BUILD = "b11609"                                  # the first with --moe-cache-mib (#29887)
LLAMA_GH = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{asset}"
SERVER_EXE = "llama-server.exe" if WIN else "llama-server"
CUDA13_DRIVER = 580                                     # CUDA 13 needs an NVIDIA driver 580 or newer

# a starting list for the model manager; any other GGUF repository can be typed in.  All checked on ModelScope
# (2026-10-07) with these names; the sizes are their smallest-largest variants.
SUGGESTED = [
    ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6 35B-A3B", "MoE, 3B active: fast with experts in RAM; 10-32 GB"),
    ("unsloth/Qwen3.6-27B-GGUF", "Qwen3.6 27B", "dense: best on a GPU that holds it; 9-35 GB"),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3 Coder 30B-A3B", "code and tools, MoE; 8-36 GB"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "Gemma 4 26B-A4B", "Google, MoE; 10-23 GB"),
    ("unsloth/gpt-oss-20b-GGUF", "gpt-oss 20B", "OpenAI, MoE; 11-14 GB"),
    ("unsloth/gpt-oss-120b-GGUF", "gpt-oss 120B", "OpenAI, MoE, 64 GB+ PCs; 63-65 GB"),
    ("unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF", "Qwen3-Next 80B-A3B", "MoE, 64 GB PCs; 20-159 GB"),
    ("unsloth/GLM-4.5-Air-GGUF", "GLM-4.5-Air", "Zhipu, MoE, 64-96 GB PCs; 38-128 GB"),
    ("Qwen/Qwen3-8B-GGUF", "Qwen3 8B", "small and quick, for any GPU; 5-9 GB"),
]
RAM_LEFT_GB = 6                                         # the OS and the server beside the model's weights in RAM
VRAM_LEFT_GB = 1.5                                      # the KV cache and compute buffers beside its weights in VRAM
# --moe-cache-mib for a MoE model bigger than the VRAM: this share of the VRAM caches the experts --fit leaves in RAM.
# llmserver (RTX 4060 Ti 16 GB, PCIe 3.0 x8, DDR3), Qwen3.6-35B-A3B UD-Q4_K_XL (22.4 GB), b11609, 4 prompts, 2 rounds:
# --fit on 33.6 tok/s; with a cache of 2048 MiB 42.3, 4096 46.7, 8192 49.5 (half the VRAM); -cmoe with 11000 48.9.
MOE_CACHE_VRAM_SHARE = 0.5


# ------------------------------------------------------------------------------------------------ the engine
def engine_kind(gpus: list[dict]) -> str:
    """Which official llama.cpp build this PC takes: cuda13, cuda12 (an older NVIDIA driver), vulkan (AMD, Intel)
    or cpu."""
    nv = [g for g in gpus if g.get("vendor", "nvidia") == "nvidia" and str(g.get("driver", "0")).split(".")[0].isdigit()]
    if nv:
        return "cuda13" if int(str(nv[0]["driver"]).split(".")[0]) >= CUDA13_DRIVER else "cuda12"
    return "vulkan" if gpus else "cpu"


def engine_assets(kind: str, build: str = LLAMA_BUILD) -> list[str]:
    """The release files of that build, as ggml-org publishes them (the CUDA runtime is a file of its own)."""
    if WIN:
        return {"cuda13": [f"llama-{build}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"],
                "cuda12": [f"llama-{build}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
                "vulkan": [f"llama-{build}-bin-win-vulkan-x64.zip"],
                "cpu": [f"llama-{build}-bin-win-cpu-x64.zip"]}[kind]
    return {"cuda13": [f"llama-{build}-bin-ubuntu-cuda-13.4-x64.tar.gz",
                       f"cudart-llama-{build}-bin-ubuntu-cuda-13.4-x64.tar.gz"],
            "cuda12": [f"llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                       f"cudart-llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz"],
            "vulkan": [f"llama-{build}-bin-ubuntu-vulkan-x64.tar.gz"],
            "cpu": [f"llama-{build}-bin-ubuntu-x64.tar.gz"]}[kind]


def engine_dir(kind: str, build: str = LLAMA_BUILD) -> Path:
    return ROOT / "engine-llama" / f"{build}-{kind}"


def find_server(folder: Path) -> Path | None:
    hits = sorted(folder.rglob(SERVER_EXE), key=lambda p: len(p.parts))
    return hits[0] if hits else None


RUNTIME_LIBS = ("libcudart", "libcublas", "cudart64", "cublas64", "cublasLt64")


def gather_runtime(folder: Path, dest: Path) -> list[str]:
    """The CUDA runtime (its own archive) beside llama-server: the Linux build's libggml-cuda.so looks for
    libcudart / libcublas only in its own folder (RUNPATH $ORIGIN), and the runtime archive unpacks into a folder of
    its own - without this llama.cpp quietly runs on the CPU (found 2026-10-07 on llmserver)."""
    moved = []
    for f in list(folder.rglob("*")):
        if f.is_file() and f.parent != dest and f.name.startswith(RUNTIME_LIBS):
            target = dest / f.name
            if target.exists():
                target.unlink()
            f.replace(target)
            moved.append(f.name)
    return moved


MSVC_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def vc_runtime(dest: Path) -> list[str]:
    """Windows: Microsoft's C++ runtime beside llama-server.  ggml-org's Windows builds import it from the system, and
    a PC without the Visual C++ Redistributable stops with "VCRUNTIME140.dll was not found" (a user's PC, 2026-10-20).
    The msvc-runtime wheel (in the Windows package's Python; else installed here) has all three, one version, in
    Scripts/; a DLL beside the exe is the one Windows loads."""
    def source(name: str) -> Path | None:
        for base in dict.fromkeys((Path(sys.prefix), Path(sys.base_prefix))):
            for p in (base / "Scripts" / name, base / name):
                if p.is_file():
                    return p
        return None
    if not WIN or all((dest / n).exists() for n in MSVC_DLLS):
        return []
    if any(source(n) is None or source(n).parent.name != "Scripts" for n in MSVC_DLLS):
        S.pip_install(S.MSVC_WHEELS, "Microsoft's C++ runtime for llama.cpp")
    copied = []
    for n in MSVC_DLLS:
        src = source(n)
        if src is None:
            S.fail(f"{n} is missing: llama.cpp cannot start without Microsoft's C++ runtime",
                   "install the Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) and run it again")
        if not (dest / n).exists():
            shutil.copy2(src, dest / n)
            copied.append(n)
    return copied


def asset_urls(asset: str, build: str = LLAMA_BUILD) -> list[str]:
    """Where a release file comes from: the copy on ModelScope first when ModelScope is the source (mainland China
    reaches it when GitHub's downloads do not), then GitHub."""
    gh = LLAMA_GH.format(build=build, asset=asset)
    ms = S.ms_url(S.MS_ENGINE_REPO, f"llama.cpp/{build}/{asset}")
    return [ms, gh] if S.model_source() == "modelscope" else [gh, ms]


def get_engine(gpus: list[dict], build: str = LLAMA_BUILD) -> Path:
    """llama-server of the pinned build for this PC, downloaded and unpacked once (engine-llama/<build>-<kind>/)."""
    kind = engine_kind(gpus)
    folder = engine_dir(kind, build)
    have = find_server(folder) if folder.exists() else None
    if have and (folder / ".done").exists():
        vc_runtime(have.parent)                        # an engine installed before the runtime was put beside it
        S.ok(f"llama.cpp {build} ({kind}) already installed")
        return have
    folder.mkdir(parents=True, exist_ok=True)
    for asset in engine_assets(kind, build):
        z = folder / asset
        if z.exists() and not S.done(z):               # put there by hand (a download in progress is <name>.part)
            S.ok(f"{asset}: using the copy in {folder}")
        else:
            for i, url in enumerate(asset_urls(asset, build)):
                if S.reachable(url, timeout=30):
                    break
                if i == 1:
                    S.fail(f"cannot download {asset} (neither GitHub nor ModelScope answers)",
                           f"check the internet connection and run it again - or download it on another PC from "
                           f"{LLAMA_GH.format(build=build, asset=asset)} and put it in {folder}")
            S.download(url, z, asset)
        S.say(f"  Unpacking {asset} ...")
        if asset.endswith(".zip"):
            with zipfile.ZipFile(z) as f:
                f.extractall(folder)
        else:
            with tarfile.open(z) as f:
                f.extractall(folder, filter="data") if hasattr(tarfile, "data_filter") else f.extractall(folder)
        z.unlink(missing_ok=True)
        z.with_name(z.name + ".done").unlink(missing_ok=True)
    exe = find_server(folder)
    if exe is None:
        S.fail(f"the llama.cpp archive has no {SERVER_EXE}", "run it again; if it repeats, report it")
    gather_runtime(folder, exe.parent)
    vc_runtime(exe.parent)
    if not WIN:
        for p in exe.parent.iterdir():
            if p.is_file() and (p.name.startswith("llama-") or p.suffix == ""):
                p.chmod(0o755)
    (folder / ".done").write_text(build, encoding="utf-8")
    S.ok(f"llama.cpp {build} ({kind}): {exe}")
    return exe


def expert_gb(files: list) -> float:
    """The routed experts' bytes in a GGUF (its *_exps tensors, from the headers only), in GB; 0 for a dense model."""
    total = 0
    for f in files:
        f = Path(f)
        try:
            _, tensors = QS.read_gguf(QS.Source(f.name, f.stat().st_size, path=str(f)))
        except (OSError, ValueError):
            continue
        total += sum(t.get("bytes", 0) for t in tensors if "_exps" in t["name"])
    return total / 1e9


def moe_cache_mib(kind: str, model_gb: float, experts_gb: float, vram_gb: float) -> int:
    """--moe-cache-mib for this model on this PC: half the VRAM (at most the experts' size), for a MoE model that does
    not fit in VRAM, on the CUDA builds (the cache needs a GPU backend; measured on CUDA only).  0: no cache."""
    if kind not in ("cuda13", "cuda12") or experts_gb <= 0 or model_gb <= vram_gb - VRAM_LEFT_GB:
        return 0
    mib = int(min(vram_gb * MOE_CACHE_VRAM_SHARE, experts_gb) * 1024) // 256 * 256
    return mib if mib >= 1024 else 0


def run_args(context: int, kind: str, model_gb: float, experts_gb: float, vram_gb: float) -> list[str]:
    args = ["-c", str(context), "--fit", "on"]
    mib = moe_cache_mib(kind, model_gb, experts_gb, vram_gb)
    return args + ["--moe-cache-mib", str(mib)] if mib else args


def gpus_here() -> list[dict]:
    try:
        return [{**g, "vendor": "nvidia"} for g in S.gpus()]
    except Exception:
        return []


def refresh_config(path: Path) -> dict:
    """A config written for an older llama.cpp build: the pinned build's engine (downloaded once) and the expert cache
    it can use, so a model installed before an update gets them without installing it again.  The context and any
    flags added by hand stay; when anything fails the config is left as it was."""
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("llama_build") == LLAMA_BUILD:
        if WIN and cfg.get("exe"):                     # installed before the C++ runtime was put beside llama-server
            try:
                vc_runtime(Path(cfg["exe"]).parent)
            except (Exception, SystemExit) as e:
                S.warn(f"Microsoft's C++ runtime: {e}")
        return cfg
    try:
        gpus = gpus_here()
        exe = get_engine(gpus)
        vram = max((g["vram_gb"] for g in gpus), default=0.0)
        model = Path(cfg["model"])
        m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", model.name)   # every shard of a split model
        files = sorted(model.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf")) if m else [model]
        args = [a for a in cfg.get("args", [])]
        if "--moe-cache-mib" not in args:
            mib = moe_cache_mib(engine_kind(gpus), float(cfg.get("gb") or 0), expert_gb(files), vram)
            if mib:
                args += ["--moe-cache-mib", str(mib)]
        cfg.update(exe=str(exe), args=args, llama_build=LLAMA_BUILD)
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        S.ok(f"{path.name}: llama.cpp {LLAMA_BUILD}" + (f", expert cache {args[args.index('--moe-cache-mib') + 1]} MiB"
                                                        if "--moe-cache-mib" in args else ""))
    except (Exception, SystemExit) as e:                   # S.fail exits; keep the old engine then
        S.warn(f"{path.name}: staying on llama.cpp {cfg.get('llama_build')} ({e})")
    return cfg


# ------------------------------------------------------------------------------------------------ the model
def hub_files(repo: str, source: str) -> list:
    """The repository's files (quantscope.Source: name, size, url)."""
    return QS.list_hub("ms" if source == "modelscope" else "hf", repo)


def variants(repo: str, source: str) -> list[dict]:
    """The GGUF variants of a repository: name, GB, its files (shards together; projector, imatrix and MTP files apart)."""
    groups: dict[str, list] = {}
    for f in hub_files(repo, source):
        base = f.name.rsplit("/", 1)[-1].lower()     # not models of their own: projectors, imatrix data, MTP draft heads
        if f.name.endswith(".gguf") and "mmproj" not in base and "imatrix" not in base and not base.startswith("mtp-"):
            groups.setdefault(QS.variant_of(f.name), []).append(f)
    out = []
    for name, fs in groups.items():
        fs.sort(key=lambda f: f.name)
        out.append({"name": name, "label": name.rsplit("/", 1)[-1], "gb": round(sum(f.size for f in fs) / 1e9, 2),
                    "files": [{"path": f.name, "size": f.size} for f in fs]})
    return sorted(out, key=lambda v: v["gb"])


# publishers whose GGUF files follow llama.cpp closely: listed first in a search, before more downloaded fine-tunes
KNOWN_PUBLISHERS = ("unsloth", "ggml-org", "Qwen", "lmstudio-community", "bartowski", "google", "openai", "zai-org")


def search(query: str, source: str, limit: int = 20) -> list[dict]:
    """GGUF repositories whose name matches `query` (a model name such as "Qwen3.8-27B"): repo, downloads, known."""
    from urllib.parse import quote
    q = quote(query)
    if source == "modelscope":
        data = json.loads(QS.http(f"{S.ms_endpoint()}/openapi/v1/models?search={q}&page_size=50&sort=downloads"))
        found = [(m["id"], int(m.get("downloads") or 0)) for m in data["data"]["models"]
                 if "library:gguf" in m.get("tags", []) or "gguf" in m["id"].lower()]
    else:
        base = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        data = json.loads(QS.http(f"{base}/api/models?search={q}&filter=gguf&sort=downloads&limit=100"))
        found = [(m["id"], int(m.get("downloads") or 0)) for m in data]
    out = [{"repo": r, "downloads": d, "known": r.split("/", 1)[0] in KNOWN_PUBLISHERS} for r, d in found]
    out.sort(key=lambda x: (not x["known"], -x["downloads"]))
    return out[:limit]


def fit(gb: float, ram_gb: float, vram_gb: float) -> dict:
    """Whether a model of `gb` runs here: llama-server's --fit puts what fits in VRAM there and the rest in RAM.
    "ok": it fits in VRAM (fast) or VRAM + RAM; "slow": it needs most of the RAM (little left for anything else);
    "no": it does not fit at all (llama.cpp would read it from the SSD through mmap - very slow)."""
    gpu = max(0.0, vram_gb - VRAM_LEFT_GB)
    if gpu <= 0:                                        # no GPU llama.cpp can use: the CPU from RAM
        return ({"level": "slow", "where": "cpu"} if gb <= ram_gb - RAM_LEFT_GB else
                {"level": "no", "where": "ssd", "need_ram": round(gb + RAM_LEFT_GB)})
    if gb <= gpu:
        return {"level": "ok", "where": "gpu"}
    if gb <= gpu + ram_gb - RAM_LEFT_GB:
        return {"level": "ok" if gb <= gpu + 0.7 * ram_gb else "slow", "where": "gpu+ram"}
    return {"level": "no", "where": "ssd", "need_ram": round(gb - gpu + RAM_LEFT_GB)}


def model_key(repo: str, variant: str) -> str:
    base = repo.split("/")[-1]
    base = re.sub(r"(?i)-gguf$", "", base)
    v = variant.rsplit("/", 1)[-1]
    v = v[len(base) + 1:] if v.lower().startswith(base.lower() + "-") else v
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{base}-{v}").strip("-").lower()


def models_root(data_dir: str | None = None) -> Path:
    """Where the llama.cpp models go: <data folder>/llama - the folder given (--data-dir, the page's download
    folder), else the one setup remembers, else Strata-data beside this folder (as setup.py's own models)."""
    settings = S.load_settings()
    data = (Path(data_dir).expanduser().resolve() if data_dir else
            Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-data")
    return data / "llama"


def remember_data_dir(data_dir: str) -> None:
    """The folder chosen becomes the default (setup.py keeps the same setting).  Nothing already there is moved."""
    settings = S.load_settings()
    S.save_settings({**settings, "data_dir": str(Path(data_dir).expanduser().resolve())})


def left_gb(variant: dict, dest: Path) -> float:
    """What the variant still has to download into dest, in GB: each file's size less what is already there (the
    finished file, or the .part of one in progress) - an install run again needs no room for what it has."""
    left = 0
    for f in variant["files"]:
        dst = dest / Path(f["path"]).name
        part = dst.with_name(dst.name + ".part")
        have = dst.stat().st_size if dst.exists() else S.part_bytes(part)
        left += max(0, f["size"] - have)
    return left / 1e9


def download_model(repo: str, variant: dict, source: str, dest: Path) -> list[Path]:
    """Every shard of the variant into dest (resumable; a ModelScope file is checked against its published
    SHA-256)."""
    left = left_gb(variant, dest)
    need = left + 2
    free = S.free_gb(dest.parent if dest.parent.exists() else ROOT)
    if left > 0 and free < need:
        S.fail(f"not enough free disk space for {variant['label']}: need ~{need:.0f} GB, {free:.0f} GB free",
               "free some space, or choose another folder with setup's --data-dir")
    out = []
    for f in variant["files"]:
        path = f["path"]
        dst = dest / Path(path).name
        url = S.ms_url(repo, path) if source == "modelscope" else f"{S.hf_endpoint()}/{repo}/resolve/main/{path}"
        S.download(url, dst, Path(path).name)
        if source == "modelscope":
            meta = S.ms_meta(repo, path)
            if meta and meta[1]:
                S.verify_sha256(dst, meta[0], meta[1])
        out.append(dst)
    return out


def install(repo: str, variant_name: str, source: str = "auto", context: int | None = None,
            data_dir: str | None = None) -> Path:
    """The engine, the model and its run config: llama-<key>.json and run-llama-<key>.bat/.sh in the Strata folder."""
    if source == "auto":
        source = S.model_source()
    S.step(1, "this PC")
    gpus = gpus_here()
    vram = max((g["vram_gb"] for g in gpus), default=0.0)
    ram = S.ram_gb()
    S.ok(f"{'GPU: ' + gpus[0]['name'] + f', {vram:.0f} GB VRAM' if gpus else 'no NVIDIA GPU'}; RAM {ram:.0f} GB")
    S.step(2, f"{repo} - {variant_name}")
    vs = [v for v in variants(repo, source) if v["name"] == variant_name or v["label"] == variant_name]
    if not vs:
        S.fail(f"{repo} has no GGUF variant {variant_name!r}", "list them with: python serve/llama_models.py "
               f"variants {repo}")
    v = vs[0]
    verdict = fit(v["gb"], ram, vram)
    S.ok(f"{v['label']}: {v['gb']:.1f} GB in {len(v['files'])} file(s) - "
         + {"gpu": "fits in VRAM", "gpu+ram": "split between VRAM and RAM", "cpu": "runs on the CPU from RAM (no GPU)",
            "ssd": "does NOT fit in VRAM + RAM"}[
             verdict["where"]])
    if verdict["level"] == "no":
        S.warn(f"it needs about {verdict['need_ram']} GB of RAM beside the GPU; it would run from the SSD, very slowly")
    S.step(3, "the llama.cpp engine")
    # llama_front.py shares serve/server.py's page and checks, which need jinja2 & co.: on a PC where no Strata model
    # was installed first they are not in .venv yet (found 2026-10-07 on a fresh Windows PC: No module named 'jinja2')
    S.pip_install(S.runtime_requirements(), "numpy, jinja2, regex, pyyaml, tqdm, requests, pillow, psutil")
    exe = get_engine(gpus)
    S.step(4, f"downloading {v['label']} from {'ModelScope' if source == 'modelscope' else 'Hugging Face'}")
    key = model_key(repo, v["name"])
    dest = models_root(data_dir) / key
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        S.fail(f"cannot write to {dest} ({e})", "choose another folder")
    S.say(f"  The model files go in {dest}")
    if data_dir:
        remember_data_dir(data_dir)
    files = download_model(repo, v, source, dest)
    S.step(5, "the run config and start script")
    port = 8080
    args = run_args(context or 32768, engine_kind(gpus), v["gb"], expert_gb(files), vram)
    if "--moe-cache-mib" in args:
        S.ok(f"expert cache in VRAM: {args[args.index('--moe-cache-mib') + 1]} MiB (the model does not fit in VRAM)")
    log = ROOT / f"llama-{key}.log"
    cfg = {"backend": "llamacpp", "exe": str(exe), "model": str(files[0]), "args": args,
           "model_name": key, "port": port, "log": str(log), "repo": repo, "variant": v["name"], "source": source,
           "gb": v["gb"], "llama_build": LLAMA_BUILD}
    path = ROOT / f"llama-{key}.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    py = sys.executable
    if WIN:
        script = ROOT / f"run-llama-{key}.bat"
        script.write_text(f'@echo off\r\ntitle TokenRun {key}\r\ncd /d "{ROOT}"\r\n"{py}" "{ROOT / "serve" / "llama_front.py"}" '
                          f'--config "{path}" --open\r\nif errorlevel 1 pause\r\n', encoding="utf-8")
    else:
        script = ROOT / f"run-llama-{key}.sh"
        script.write_text(f'#!/bin/sh\ncd "{ROOT}"\nexec "{py}" "{ROOT / "serve" / "llama_front.py"}" --config "{path}"'
                          f' "$@"\n', encoding="utf-8")
        script.chmod(0o755)
    S.ok(f"run config: {path.name}; start script: {script.name}")
    S.say("All set.")
    return path


def installed() -> dict:
    """key -> its run config, for the llama.cpp models set up in this folder."""
    out = {}
    for p in sorted(ROOT.glob("llama-*.json")):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and cfg.get("backend") == "llamacpp":
            out[p.stem[len("llama-"):]] = {"config": p.name, "model_name": cfg.get("model_name", p.stem),
                                           "port": cfg.get("port") or 8080, "repo": cfg.get("repo"),
                                           "variant": cfg.get("variant"), "gb": cfg.get("gb")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variants", help="the GGUF variants of a repository")
    v.add_argument("repo")
    v.add_argument("--source", choices=["auto", "modelscope", "huggingface"], default="auto")
    i = sub.add_parser("install", help="download the engine and the model, write the run config")
    i.add_argument("repo")
    i.add_argument("variant")
    i.add_argument("--source", choices=["auto", "modelscope", "huggingface"], default="auto")
    i.add_argument("--context", type=int)
    i.add_argument("--data-dir", help="the data folder: the model goes in <it>/llama/<model> (default: setup's data "
                                      "folder, else Strata-data beside this folder)")
    s = sub.add_parser("start", help="start an installed model (serve/llama_front.py)")
    s.add_argument("key")
    s.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "variants":
        src = S.model_source() if a.source == "auto" else a.source
        for x in variants(a.repo, src):
            print(f"{x['gb']:8.1f} GB  {x['name']}  ({len(x['files'])} file{'s' if len(x['files']) > 1 else ''})")
        return 0
    if a.cmd == "install":
        install(a.repo, a.variant, a.source, a.context, a.data_dir)
        return 0
    cfg = installed().get(a.key)
    if not cfg:
        ap.error(f"{a.key} is not installed here; installed: {', '.join(installed()) or 'none'}")
    return subprocess.call([sys.executable, str(ROOT / "serve" / "llama_front.py"), "--config",
                            str(ROOT / cfg["config"]), *(["--open"] if a.open else [])])


if __name__ == "__main__":
    sys.exit(main())
