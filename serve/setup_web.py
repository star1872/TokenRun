"""The model manager: a local web page for what setup.py asks in the terminal.

    START-HERE.bat --web   (Windows)   /   ./setup.sh --web   (Linux)   ->   http://127.0.0.1:8090/

It shows this PC (GPU, RAM, disk), every model and size setup installs with whether it will run here and why (setup's
own rules: the experts must fit the RAM, a supported GPU, room on the disk), reads a GGUF's headers to tell its real
quantization (tools/strata_inspect.py), downloads and prepares a model by running setup.py itself (from ModelScope or
Hugging Face), and starts an installed one.  Nothing is installed by this file: every install is setup.py's.

Only this PC can reach it (127.0.0.1), and a request from another web page is refused (the Origin and Host checks).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import setup as S  # noqa: E402  (setup.py: its tables and checks, no side effects on import)
import quantscope as QS  # noqa: E402  (tools/quantscope.py: a bits-per-weight number as "Q3 class")

WIN = os.name == "nt"
SMALL_VRAM_GB = 12        # README: "It needs 12 GB of VRAM or more"; smaller cards start (0.1.38: 6 GB) but run slower
MIN_VRAM_GB = 6


# ------------------------------------------------------------------------------------------------ this PC
def detect() -> dict:
    """GPU, RAM, CPU and the model folder's free space, as setup sees them (no install, no file moved)."""
    nv = []
    try:
        nv = S.gpus()
    except Exception:                                   # no nvidia-smi
        pass
    cards = [{"index": g["index"], "name": g["name"], "vram_gb": round(g["vram_gb"], 1), "vendor": "nvidia",
              "problem": S.gpu_problem(g)} for g in nv]
    if not any(c["problem"] is None for c in cards):
        try:
            amd = S.amd_gpus_windows() if WIN else S.amd_gpus()
        except Exception:
            amd = []
        cards += [{"index": g.get("index", i), "name": g.get("name", "AMD GPU"), "vram_gb": round(g.get("vram_gb", 0), 1),
                   "vendor": "amd", "problem": S.amd_problem(g)} for i, g in enumerate(amd)]
    try:
        cpu, avx2, avx512 = S.cpu_info()
    except Exception:
        cpu, avx2, avx512 = "unknown CPU", False, False
    settings = S.load_settings()
    data = Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-data"
    models_dir = data / "models"
    probe = next((p for p in (models_dir, data, ROOT.parent) if p.exists()), ROOT)
    return {"ram_gb": round(S.ram_gb(), 1), "cpu": cpu, "avx2": avx2, "avx512": avx512, "gpus": cards,
            "os": "Windows" if WIN else "Linux", "data_dir": str(data), "models_dir": str(models_dir),
            "free_gb": round(shutil.disk_usage(probe).free / 1e9, 1)}


# ------------------------------------------------------------------------------------------------ will it run here
def fit(family: str, model: str, ram: float, vram: float, gpu_ok: bool, amd: bool, free: float | None,
        avx512: bool = False) -> dict:
    """Whether a size runs on a PC with this RAM and GPU, by setup's rules: "ok", "slow" (it runs, but slower: a
    small card, the low-RAM mode, experts read from the SSD) or "no" (it will not start).  Each reason is a code and
    its numbers; the page words them."""
    m = S.MODELS[model]
    arena, reasons, level = m["arena_gb"], [], "ok"

    def worse(lv, code, **kw):
        nonlocal level
        reasons.append({"code": code, "level": lv, **kw})
        if ["ok", "slow", "no"].index(lv) > ["ok", "slow", "no"].index(level):
            level = lv

    if not gpu_ok:
        worse("no", "no_gpu")
    elif vram < MIN_VRAM_GB - 0.5:
        worse("no", "vram_tiny", vram=vram, need=MIN_VRAM_GB)
    elif vram < SMALL_VRAM_GB - 0.5:
        worse("slow", "vram_small", vram=vram, want=SMALL_VRAM_GB)
    if amd and m.get("nvidia_only"):
        worse("no", "nvidia_only")
    headroom = S.LOW_RAM_HEADROOM_GB
    if m.get("budget"):                                 # Unsloth: a RAM budget of experts, the rest read from the SSD
        if ram >= arena + headroom:
            pass
        elif ram >= m["ram_gb"]:
            worse("slow", "ram_budget", ram=ram, experts=arena, need=round(arena + headroom))
        else:
            worse("no", "ram_short", ram=ram, need=m["ram_gb"])
    elif ram >= arena + headroom:
        pass
    elif gpu_ok and S.low_ram_resident(model, ram, vram):
        # the low-RAM mode's resident variant: what the card does not hold is copied into RAM once (no SSD reads)
        reasons.append({"code": "low_ram_resident", "level": "info", "ram": ram, "experts": arena})
    elif gpu_ok and S.low_ram_fits(model, ram, vram):
        worse("slow", "low_ram", ram=ram, experts=arena, need=round(arena + headroom))
    else:
        worse("no", "ram_short", ram=ram, need=round(arena + headroom))
    if free is not None:
        need = S.disk_need_gb(family, model, avx512, low_ram=not m.get("budget") and S.low_ram_needed(model, ram))
        if free < need:                                # another folder fixes it: a warning, not the verdict
            reasons.append({"code": "disk", "level": "warn", "free": free, "need": round(need)})
    if m.get("experimental"):
        reasons.append({"code": "experimental", "level": "info"})
    if not reasons:
        reasons.append({"code": "fits", "level": "ok", "ram": ram, "experts": arena})
    return {"level": level, "reasons": reasons}


def catalog(hw: dict, ram: float | None = None, vram: float | None = None) -> list[dict]:
    """Every family x size setup installs, with its size, the experts' real bits and the verdict for this PC (or for
    the RAM / VRAM given: "what if" for another PC)."""
    usable = [g for g in hw["gpus"] if g["problem"] is None]
    gpu_ok = bool(usable) or vram is not None
    vram = vram if vram is not None else max((g["vram_gb"] for g in usable), default=0.0)
    ram = ram if ram is not None else hw["ram_gb"]
    amd = bool(usable) and all(g["vendor"] == "amd" for g in usable)
    have = installed()
    out = []
    for fam, d in S.FAMILIES.items():
        for model, m in S.MODELS.items():
            if fam not in m.get("families", ("qwen", "swift")):
                continue
            bits = S.expert_bits(fam, model)
            out.append({"family": fam, "family_title": d["title"], "by": d["by"], "model": model,
                        "about": m["about"], "download_gb": m["download_gb"], "ram_gb": m["ram_gb"],
                        "experts_gb": m["arena_gb"], "expert_bits": bits,
                        "class": QS.q_class(bits) if bits else None,
                        "experimental": bool(m.get("experimental")), "installed": f"{fam}/{model}" in have,
                        "fit": fit(fam, model, ram, vram, gpu_ok, amd, hw["free_gb"], bool(hw.get("avx512")))})
    return out


def installed() -> dict:
    """family/model -> its run config, for the models set up in this Strata folder."""
    out = {}
    for p in sorted(ROOT.glob("strata-*.json")):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
            if not (isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list)):
                continue
            ch = S.choices_from_config(p)
        except (OSError, ValueError, KeyError):
            continue
        if ch.get("model"):
            out[f"{ch['family']}/{ch['model']}"] = {"config": p.name, "model_name": cfg.get("model_name", p.stem),
                                                    "port": cfg.get("port") or 8080, "context": ch.get("context")}
    return out


# ------------------------------------------------------------------------------------------------ inspect a GGUF
def inspect(source: str, variant: str | None) -> dict:
    """tools/strata_inspect.py as data: the variants of a repository or folder, or one variant's storage per weight
    group and the verdict (known / layout / no / untested)."""
    import strata_inspect as SI
    import quantscope as qs
    from collections import defaultdict
    files = [f for f in qs.resolve(source) if f.name.endswith(".gguf") and "mmproj" not in f.name.lower()]
    variants = defaultdict(list)
    for f in files:
        variants[qs.variant_of(f.name)].append(f)
    if not variants:
        return {"error": "no_gguf"}
    import fnmatch
    chosen = {k: v for k, v in variants.items() if not variant or variant in k or fnmatch.fnmatch(k, variant)}
    if not chosen:
        return {"error": "no_variant", "variants": sorted(variants)}
    if len(chosen) > 1:
        return {"variants": [{"name": k, "gb": round(sum(f.size for f in v) / 1e9, 2), "files": len(v)}
                             for k, v in sorted(chosen.items())]}
    name, fs = next(iter(chosen.items()))
    arch, tensors = SI.read_headers(sorted(fs, key=lambda f: f.name))
    groups = defaultdict(lambda: {"elements": 0, "bytes": 0, "types": defaultdict(int)})
    for t in tensors:
        g = groups[qs.category(t["name"])]
        g["elements"] += t["elements"]
        g["bytes"] += t["bytes"]
        g["types"][t["type"]] += t["bytes"]
    rows = [{"group": k, "gb": round(g["bytes"] / 1e9, 2), "bpw": round(8 * g["bytes"] / g["elements"], 2) if g["elements"] else None,
             "types": [[ty, round(100 * b / g["bytes"])] for ty, b in sorted(g["types"].items(), key=lambda kv: -kv[1])]}
            for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])]
    fp = SI.fingerprint(arch, tensors)
    kind, sentence = SI.verdict(fp, SI.load_fingerprints())
    known = re.search(r"--family (\w+) --model (\S+)", sentence)
    return {"name": name, "arch": arch, "gb": round(fp["bytes"] / 1e9, 2), "bpw": fp["bpw"],
            "experts_bpw": fp["experts_bpw"], "class": qs.q_class(fp["experts_bpw"]) if fp["experts_bpw"] else None,
            "groups": rows, "verdict": kind, "sentence": sentence,
            "known": {"family": known.group(1), "model": known.group(2)} if known else None}


# ------------------------------------------------------------------------------------------------ setup.py jobs
class Job:
    """One setup.py (install) or server (start) process, its output kept for the page."""

    def __init__(self, kind: str, cmd: list[str], what: dict):
        self.kind, self.what, self.lines, self.progress = kind, what, [], ""
        self.started, self.code = time.time(), None
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if WIN else 0
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                     start_new_session=not WIN)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(1)
            buf += chunk
            # a line ends with \n (or Windows' \r\n); a progress line rewrites itself with a lone \r: only its newest
            # state is kept
            while True:
                m = re.search(rb"\r\n|\n|\r(?=[^\n])|\r$", buf)
                if not m or (m.group() == b"\r" and m.end() == len(buf) and chunk):
                    break                               # a \r at the end: wait for the next byte (\r\n or not)
                piece, sep, buf = buf[:m.start()], m.group(), buf[m.end():]
                text = piece.decode("utf-8", "replace").rstrip()
                if sep == b"\r":
                    if text:
                        self.progress = text
                elif text:
                    self.lines.append(text)
                    self.progress = ""
                    del self.lines[:-400]
            if not chunk:
                break
        if buf.strip():
            self.lines.append(buf.decode("utf-8", "replace").rstrip())
        self.code = self.proc.wait()

    def state(self) -> dict:
        out = {"kind": self.kind, "what": self.what, "running": self.code is None, "code": self.code,
               "seconds": round(time.time() - self.started), "lines": self.lines[-60:], "progress": self.progress}
        if self.kind == "start" and self.code is None:  # the model answers: its page can be opened
            out["ready"] = self._ready()
        return out

    def _ready(self) -> bool:
        if getattr(self, "_ok", False):
            return True
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{self.what['port']}/health", timeout=1) as r:
                self._ok = r.status == 200
        except OSError:
            self._ok = False
        return self._ok

    def stop(self):
        if self.code is not None:
            return
        try:
            if WIN:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except OSError:
            pass


JOBS: dict[str, Job] = {}
LISTEN_HOST = "127.0.0.1"    # main --host: a model started from the page listens there too (its chat page opens)
LOCK = threading.Lock()


def data_dir_arg(path) -> str | None:
    """A data folder the page names (where the model files go): an absolute path, made if missing.  None: the
    default (setup's remembered data folder, else Strata-data beside this folder).  ValueError for anything else."""
    if path is None or not str(path).strip():
        return None
    p = str(path).strip()
    if any(c in p for c in "\r\n\0") or p.startswith("-") or not os.path.isabs(os.path.expanduser(p)):
        raise ValueError("the folder must be a full path, e.g. /data/models or D:\\Models")
    return str(Path(p).expanduser().resolve())


def folder_info(path) -> dict:
    """GET folder?path=: whether the models can go there and how much space it has."""
    p = data_dir_arg(path) or hardware()["data_dir"]
    probe = Path(p)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = round(shutil.disk_usage(probe).free / 1e9, 1)
    except OSError:
        free = None
    writable = os.access(probe, os.W_OK)
    return {"path": p, "exists": Path(p).is_dir(), "writable": writable, "free_gb": free}


def install(family: str, model: str, source: str, context: int | None, data_dir: str | None = None) -> Job:
    """setup.py --setup --no-start --yes with this family, size and download source: the same install the terminal
    makes with these answers (an explicit --model is the consent to a risk setup would otherwise stop at)."""
    if family not in S.FAMILIES or model not in S.MODELS or family not in S.MODELS[model].get("families", ("qwen", "swift")):
        raise ValueError("unknown family or size")
    if source not in S.SOURCES:
        raise ValueError("unknown download source")
    cmd = [sys.executable, str(ROOT / "setup.py"), "--setup", "--no-start", "--yes", "--no-browser",
           "--family", family, "--model", model, "--source", source]
    if context:
        if context not in S.CONTEXTS:
            raise ValueError("unknown context")
        cmd += ["--context", str(context)]
    data = data_dir_arg(data_dir)
    if data:
        cmd += ["--data-dir", data]
    return Job("install", cmd, {"family": family, "model": model, "source": source})


# owner/name as ModelScope and Hugging Face name repositories: no path tricks (".."), nothing to read as an option
REPO_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def llama():
    """serve/llama_models.py: any GGUF model through llama.cpp (imported when first asked)."""
    from serve import llama_models
    return llama_models


def llama_variants(repo: str, source: str, data_dir: str | None = None) -> dict:
    """A repository's GGUF variants, each with whether it runs on this PC (llama-server places what fits in VRAM there
    and the rest in RAM)."""
    if not REPO_NAME.match(repo):
        raise ValueError("a repository is owner/name")
    if source not in S.SOURCES:
        raise ValueError("unknown download source")
    hw = hardware()
    vram = max((g["vram_gb"] for g in hw["gpus"] if g["problem"] is None and g["vendor"] == "nvidia"), default=0.0)
    src = S.model_source() if source == "auto" else source
    vs = llama().variants(repo, src)
    free = folder_info(data_dir)["free_gb"] if data_dir else hw["free_gb"]
    root = llama().models_root(data_dir)
    for v in vs:
        v["fit"] = llama().fit(v["gb"], hw["ram_gb"], vram)
        v["left_gb"] = round(llama().left_gb(v, root / llama().model_key(repo, v["name"])), 2)
        v["files"] = len(v["files"])
        v["disk"] = free is not None and v["left_gb"] > 0 and free < v["left_gb"] + 2
    return {"repo": repo, "source": src, "variants": vs, "free_gb": free}


def llama_install(repo: str, variant: str, source: str, context: int | None, data_dir: str | None = None) -> Job:
    if not REPO_NAME.match(repo) or not variant or any(c in variant for c in "\r\n"):
        raise ValueError("repository or variant")
    if source not in S.SOURCES:
        raise ValueError("unknown download source")
    cmd = [sys.executable, str(ROOT / "serve" / "llama_models.py"), "install", repo, variant, "--source", source]
    if context:
        cmd += ["--context", str(int(context))]
    data = data_dir_arg(data_dir)
    if data:
        cmd += ["--data-dir", data]
    return Job("install", cmd, {"family": "llama.cpp", "model": variant.rsplit("/", 1)[-1], "repo": repo})


def host_args() -> list[str]:
    return [] if LISTEN_HOST in ("127.0.0.1", "localhost") else ["--host", LISTEN_HOST]


def start_model(key: str) -> Job:
    """The model's server, as its run-<model> script starts it (serve/server.py with its run config; a llama.cpp
    model, "llama/<key>": serve/llama_front.py with its llama-<key>.json)."""
    if key.startswith("llama/"):
        cfg = llama().installed().get(key[len("llama/"):])
        if not cfg:
            raise ValueError("not installed")
        cmd = [sys.executable, str(ROOT / "serve" / "llama_front.py"), "--config", str(ROOT / cfg["config"]),
               "--port", str(cfg["port"]), *host_args()]
        return Job("start", cmd, {"key": key, "model_name": cfg["model_name"], "port": cfg["port"]})
    cfg = installed().get(key)
    if not cfg:
        raise ValueError("not installed")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(ROOT / cfg["config"]),
           "--port", str(cfg["port"]), *host_args()]
    return Job("start", cmd, {"key": key, "model_name": cfg["model_name"], "port": cfg["port"]})


def running_ports() -> list[int]:
    """The ports where a Strata model answers on this PC (the installed models' ports and the default 8080): its chat
    page can be opened from here, however it was started."""
    import urllib.request
    ports = sorted({8080, *(c["port"] for c in installed().values()), *(c["port"] for c in llama().installed().values())})
    out = []
    for p in ports:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=0.5) as r:
                if r.status == 200 and json.loads(r.read() or b"{}").get("service") == "strata":
                    out.append(p)
        except (OSError, ValueError):
            pass
    return out


# ------------------------------------------------------------------------------------------------ the API
API = "/manage/api/"     # the same paths here and in serve/server.py (the chat page's "Models" tab)


def api(method: str, url: str, raw: bytes | None, embedded: bool = False) -> tuple[int, dict]:
    """One model-manager request: (HTTP status, JSON).  The caller has checked who may ask (this server: the Host and
    the Origin; serve/server.py: its API key, Host check and own-page rule).  embedded: inside a running model's server."""
    path, _, query = url.partition("?")
    what = path[len(API):]
    q = dict(x.split("=", 1) for x in query.split("&") if "=" in x)
    try:
        if method == "GET":
            if what == "context":
                return 200, {"embedded": embedded}
            if what == "hardware":
                return 200, hardware("refresh" in q)
            if what == "models":
                num = lambda k: float(q[k]) if q.get(k, "").replace(".", "", 1).isdigit() else None   # noqa: E731
                hw = hardware()
                if q.get("dir"):
                    from urllib.parse import unquote
                    hw = {**hw, "free_gb": folder_info(unquote(q["dir"]))["free_gb"]}
                return 200, {"models": catalog(hw, num("ram"), num("vram")), "installed": installed()}
            if what == "sources":
                ms, hf = S.reachable(S.ms_endpoint()), S.reachable(S.HF_PROBE)
                return 200, {"modelscope": ms, "huggingface": hf, "auto": "modelscope" if ms else "huggingface" if hf else None}
            if what == "running":
                return 200, {"ports": running_ports()}
            if what == "llama/models":                   # the llama.cpp section: suggestions and what is installed
                return 200, {"suggested": [{"repo": r, "title": t, "about": a} for r, t, a in llama().SUGGESTED],
                             "installed": llama().installed(), "build": llama().LLAMA_BUILD}
            if what == "llama/variants":
                from urllib.parse import unquote
                return 200, llama_variants(unquote(q.get("repo", "")).strip(), unquote(q.get("source", "auto")),
                                           unquote(q.get("dir", "")) or None)
            if what == "llama/search":                   # a model name instead of owner/name: its GGUF repositories
                from urllib.parse import unquote
                query, source = unquote(q.get("q", "")).strip(), unquote(q.get("source", "auto"))
                if not query or len(query) > 100 or source not in S.SOURCES:
                    raise ValueError("a model name to search for")
                src = S.model_source() if source == "auto" else source
                return 200, {"query": query, "source": src, "repos": llama().search(query, src)}
            if what == "folder":
                from urllib.parse import unquote
                return 200, folder_info(unquote(q.get("path", "")) or None)
            if what == "job":
                with LOCK:
                    return 200, {k: j.state() for k, j in JOBS.items()}
            return 404, {"error": "not found"}
        body = json.loads(raw or b"{}")
        if not isinstance(body, dict):
            raise ValueError("a JSON object")
        if what == "inspect":
            src = str(body.get("source") or "").strip()
            if not src:
                raise ValueError("source")
            return 200, inspect(src, (body.get("variant") or "").strip() or None)
        with LOCK:
            if what in ("install", "start", "llama/install") and any(j.code is None for j in JOBS.values()):
                return 409, {"error": "busy"}
            if what == "llama/install":
                JOBS["install"] = llama_install(str(body.get("repo") or "").strip(), str(body.get("variant") or ""),
                                                str(body.get("source") or "auto"), body.get("context"),
                                                body.get("data_dir"))
                return 200, JOBS["install"].state()
            if what == "install":
                JOBS["install"] = install(str(body.get("family")), str(body.get("model")),
                                          str(body.get("source") or "auto"), body.get("context"),
                                          body.get("data_dir"))
                return 200, JOBS["install"].state()
            if what == "start":
                if running_ports():                     # one model at a time: the GPU holds one
                    return 409, {"error": "running"}
                JOBS["start"] = start_model(str(body.get("key")))
                return 200, JOBS["start"].state()
            if what == "stop":
                job = JOBS.get(str(body.get("job")))
                if job:
                    job.stop()
                return 200, {"stopped": bool(job)}
        return 404, {"error": "not found"}
    except ValueError as e:
        return 400, {"error": str(e)}
    except Exception as e:                              # a source that cannot be read, a network error
        return 500, {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------ HTTP
FILES = {"setup.html": "text/html; charset=utf-8", "setup.js": "text/javascript; charset=utf-8",
         "i18n.js": "text/javascript; charset=utf-8", "tokens.css": "text/css; charset=utf-8",
         "components.css": "text/css; charset=utf-8", "app.css": "text/css; charset=utf-8", "sprite.svg": "image/svg+xml"}
HW: dict = {}
HW_LOCK = threading.Lock()


def hardware(refresh=False) -> dict:
    """This PC, detected once (PowerShell and nvidia-smi take seconds on Windows) - at the start, in the background,
    and again on "Check again".  Requests that come meanwhile wait for the same detection instead of starting another."""
    with HW_LOCK:
        if refresh or not HW:
            HW.clear()
            HW.update(detect())
        return dict(HW)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # the browser closed the connection before the answer (a reload, a closed tab): normal, nothing to report
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def handler(port: int, extra_hosts=()):
    hosts = {"127.0.0.1", "localhost", *extra_hosts}  # any port: an SSH tunnel may forward another one
    name = lambda h: h.rsplit(":", 1)[0] if h.count(":") == 1 else h   # noqa: E731  ("127.0.0.1:8090" -> "127.0.0.1")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _ours(self) -> bool:
            """This PC's own page only: the Host is this server (no DNS rebinding) and a POST comes from this
            origin with a JSON body (a form on another site can send neither)."""
            if name(self.headers.get("Host", "")) not in hosts:
                self._json(403, {"error": "host"})
                return False
            if self.command == "POST":
                origin = self.headers.get("Origin")
                if origin and name(origin.split("://", 1)[-1]) not in hosts:
                    self._json(403, {"error": "origin"})
                    return False
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    self._json(415, {"error": "json"})
                    return False
            return True

        def do_GET(self):
            if not self._ours():
                return
            path = self.path.split("?")[0]
            if path in ("/", "/manage"):
                return self._send(200, (WEB / "setup.html").read_bytes(), FILES["setup.html"])
            if path.startswith("/web/") and path[5:] in FILES:
                return self._send(200, (WEB / path[5:]).read_bytes(), FILES[path[5:]])
            if path.startswith("/fonts/") and re.fullmatch(r"[\w.-]+\.woff2", path[7:]) and (WEB / "fonts" / path[7:]).is_file():
                return self._send(200, (WEB / "fonts" / path[7:]).read_bytes(), "font/woff2")
            if path.startswith(API):
                return self._json(*api("GET", self.path, None))
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._ours():
                return
            if not self.path.startswith(API):
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            self._json(*api("POST", self.path, self.rfile.read(min(n, 65536))))

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on; 0.0.0.0 (or this PC's LAN address) lets other PCs of the network "
                         "open the page - and install and start models here: only on a network you trust")
    a = ap.parse_args(argv)
    extra = ()
    if a.host not in ("127.0.0.1", "localhost"):
        from serve.server import host_names_for, lan_addresses
        extra = host_names_for(a.host)                 # this PC's name and LAN addresses
        print(f"[manager] listening on {a.host}: other PCs of the network can open it, and install and start "
              f"models here", flush=True)
        lan = [a.host] if a.host not in ("0.0.0.0", "::") else sorted(lan_addresses())
        for ip in lan:
            print(f"From another PC: http://{ip}:{a.port}/", flush=True)
    global LISTEN_HOST
    LISTEN_HOST = a.host
    url = f"http://127.0.0.1:{a.port}/"
    # no screen (a Linux server over SSH): webbrowser would start a text browser in this terminal, if any
    browser = not a.no_browser and (WIN or sys.platform == "darwin" or bool(os.environ.get("DISPLAY")
                                                                            or os.environ.get("WAYLAND_DISPLAY")))
    try:                                               # TokenRun.exe double-clicked again: the page that runs
        with urllib.request.urlopen(url + "manage/api/context", timeout=3) as r:
            running = json.loads(r.read()).get("embedded") is False
    except Exception:                                  # (Windows lets a second server bind the same port)
        running = None
    if running:
        print(f"TokenRun model manager is already running: {url}", flush=True)
        if browser:
            webbrowser.open(url)
        return 0
    if running is False:
        print(f"Port {a.port} is used by another program: start the model manager on another one, e.g. --port "
              f"{a.port + 1}", flush=True)
        return 1
    srv = Server((a.host, a.port), handler(a.port, extra))
    threading.Thread(target=hardware, daemon=True).start()   # ready before the page asks
    print(f"TokenRun model manager: {url}  (Ctrl+C to stop)", flush=True)
    if browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    elif not a.no_browser and a.host in ("127.0.0.1", "localhost"):
        print("No screen here: open it from another PC with --host 0.0.0.0 (then http://<this PC>:"
              f"{a.port}/), or through an SSH tunnel: ssh -L {a.port}:127.0.0.1:{a.port} <this PC>", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in JOBS.values():
            j.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
