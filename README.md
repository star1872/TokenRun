<h1 align="center">TokenRun</h1>

**English** · [简体中文](README.zh-CN.md)

<p align="center"><b>Run large AI models on your own PC, including older ones</b><br>
Web app for chat and model management · Strata and llama.cpp engines · Windows and Linux · free and open source (MIT)</p>

TokenRun is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221. Strata runs
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 125-billion-parameter model, on a gaming PC.
TokenRun adds:

- **Faster on older PCs, at the same accuracy.** On a Haswell Xeon + DDR3 + PCIe 3.0 PC: Strata's engine with
  TokenRun's settings answers 6.5% faster than TokenRun's earlier engine, which was itself about 15% faster than
  Strata 0.1.39; models run through llama.cpp that do not fit in VRAM answer up to 47% faster with its expert cache
  (measurements below).
- **A web app.** Model manager (checks your PC, says which models fit, downloads, starts and stops them) and a chat
  page (several chats, system prompts, regenerate, edit and resend, math). English and Chinese.
- **A second engine: llama.cpp.** Any GGUF model, not only Qwen3.8-Flash-Next, in the same web app.
- **Ready-to-run packages.** Windows: unzip and double-click. Linux: one command. No Python or git needed.
- **Downloads that work from mainland China.** Models, engines and packages also come from ModelScope, checked by
  SHA-256; Hugging Face is the fallback.

## Download

| | GitHub | ModelScope (faster in mainland China) |
|---|---|---|
| **Windows x64** | [TokenRun-windows-x64.zip](https://github.com/star1872/TokenRun/releases/download/tokenrun-latest/TokenRun-windows-x64.zip) | [TokenRun-windows-x64.zip](https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/TokenRun-windows-x64.zip) |
| **Linux x86_64** | [TokenRun-linux-x64.tar.gz](https://github.com/star1872/TokenRun/releases/download/tokenrun-latest/TokenRun-linux-x64.tar.gz) | `curl -fsSL https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/install.sh \| sh` |

**Windows:** unzip the whole folder and double-click `TokenRun.exe`. The model manager opens in your browser: pick a
model, download it, start it, then chat.

**Linux:** run the command above, then `tokenrun`. It installs into `~/.local/share/tokenrun` without sudo. On a
server without a screen: `tokenrun --host 0.0.0.0`, then open `http://<server>:8090/` from another PC. Running the
install command again updates TokenRun; downloaded models are kept.

The Strata engine needs what Strata needs: a graphics card with 12 GB of VRAM or more and 32 GB of RAM or more
(64 GB runs every model size); see [Strata's README](README.strata.md#what-you-need). TokenRun's engine changes were
measured on NVIDIA cards only. The llama.cpp engine runs smaller models on smaller PCs, and on the CPU alone.

## Speed on an older PC

Test PC: Xeon E5-2673 v3 (12 cores, AVX2 only), 3×32 GB DDR3-1866, RTX 4060 Ti 16 GB on PCIe 3.0 x8, SATA SSD,
Ubuntu 24.04. Four prompts (Chinese essay, code edit, English-to-Chinese translation, technical explanation), greedy,
512 tokens each, thinking off; the arms were run in turns, page cache dropped before each start. Numbers from
different days are not compared with each other: this PC's speed depends on where its RAM pages land (only 64 of its
96 GB are dual-channel).

### Strata engine (Qwen3.8-Flash-Next)

TokenRun runs the official Strata engine 0.1.44.1 (built from Niko1221/Strata's tag) with these settings
(`serve/server.py`, `engine_env`): the route tail skip off (`STRATA_ROUTE_TAIL_SKIP=0`; on by default in Strata since
0.1.42, it leaves out a missed expert ranked 7th or lower, i.e. it computes fewer experts than the model routes), the
CPU share of short prompts off, the Q2_0 AVX2 kernel on, and on Linux the adaptive tier's copies overlapped with the
next decode window (`STRATA_ADAPT_OVERLAP=1`).

2026-10-14, IQ3_S, 3 rounds each, median of the 4-prompt means:

| Engine | Decode |
| --- | ---: |
| TokenRun's earlier engine (Strata 0.1.39 + the old-PC changes below) | 40.4 tok/s |
| Strata 0.1.44.1, the same settings without the overlap | 40.9 tok/s (+1.2%, within noise) |
| **Strata 0.1.44.1 with TokenRun's settings** | **43.0 tok/s (+6.5%)** |

All 3 rounds of the last row were faster than every round of the first; prompt reading is the same (about 440 tok/s
on 3,454 tokens). Accuracy: the overlap gives exactly the same log-probabilities as without it, and Strata 0.1.44.1
differs from the earlier engine on a 3,166-token text (`STRATA_LOGPOS`) as much as changing only the GPU cache size
does (mean |Δ log p| per token 0.248 against 0.253). Not measured: Q2_0, Unsloth's files, Windows (Strata measured
the overlap 2.6% slower on Windows with an RTX 5070, so TokenRun leaves it off there).

2026-10-05, the old-PC changes against Strata 0.1.39 (compiled from source, CUDA 12.8, 64K context), median of 3 rounds:

| Model (official GGUF) | Strata 0.1.39 | TokenRun (then) | Change |
| --- | ---: | ---: | ---: |
| **Q2_0** (experts 2.25 bit) | 48.1 tok/s | **55.7 tok/s** | +15.6% |
| **IQ3_S** (experts 3.33 bit) | 30.5 tok/s | **35.0 tok/s** | +14.8% |

All 16 prompt/model pairs were faster. Unsloth's files gained less: UD-IQ4_XS +5.2%, UD-Q4_K_XL +10% (2 rounds).
Accuracy: +0.013 mean NLL against Strata 0.1.39, less than changing only the GPU cache size (+0.012 to +0.016). The
three changes, all in Strata itself since 0.1.40 (opt-in there, on in TokenRun):

1. **AVX2 kernel for Q2_0**, upstream PR [#706](https://github.com/Niko1221/Strata/pull/706) by @merbanan
   (`STRATA_Q2_BITPLANE=1`): +4.3%.
2. **Expert swaps no longer stall decoding** on slow PCIe, upstream PR [#764](https://github.com/Niko1221/Strata/pull/764)
   by @merbanan (`STRATA_ADAPT_LAG`; on Linux the overlap above does the same job): +7%.
3. **A more active GPU expert cache** (`--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`), measured here;
   [#907](https://github.com/Niko1221/Strata/pull/907) lets Strata's `--calibrate` test it on your PC: +7.9%.

Also from TokenRun in Strata: ModelScope downloads (#908), `--inspect` (#911), the PCIe link check (#912) and the
measurement report (#913). Strata's own numbers (RTX 5070 + Ryzen 5 7600 + DDR5: Q2_0 94, IQ3_S 53 tok/s) are from
other hardware. The full 2026-10-05 measurement: [Niko1221/Strata#906](https://github.com/Niko1221/Strata/issues/906)
and (in Chinese) [README.oldhw.zh-CN.md](README.oldhw.zh-CN.md).

### llama.cpp engine (any GGUF model)

Pick "llama.cpp" in the model manager to download and run any GGUF model with ggml-org's official llama.cpp builds
(TokenRun pins b11609). When a MoE model does not fit in VRAM, TokenRun turns on llama.cpp's expert cache
(`--moe-cache-mib`, [llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887)) with half the VRAM: the
experts used most are computed on the GPU instead of the CPU; every expert the model routes is still computed.

2026-10-14, Qwen3.6-35B-A3B UD-Q4_K_XL (22.4 GB), `-c 32768 --fit on`, 2 rounds of the 4-prompt mean:

| Setting | Decode |
| --- | ---: |
| b11457 (TokenRun's earlier build) | 34.5 / 32.8 tok/s |
| b11609 | 32.8 / 34.3 tok/s |
| b11609, expert cache 2048 MiB | 42.5 / 42.0 tok/s |
| b11609, expert cache 4096 MiB | 46.6 / 46.7 tok/s |
| **b11609, expert cache 8192 MiB** | **49.4 / 49.6 tok/s (+47%)** |

TokenRun itself (a config written for b11457, moved up at start; its rule gave 7936 MiB on this card): 49.0 / 49.1
tok/s. Prompt reading is about 4% slower
with the cache (346 → 330 tok/s on 3,453 tokens). Not measured: other models and GPUs, Windows. Qwen3.8-Flash-Next
itself is no match for llama.cpp on this PC (8.0 tok/s, 83.6 GB with its PLE table): use the Strata engine for it.

## From source

```bash
git clone https://github.com/star1872/TokenRun.git
cd TokenRun
./setup.sh            # Linux; START-HERE.bat on Windows
```

`setup` downloads TokenRun's build of the official Strata engine (GitHub Actions, release `engine-v<version>`).
`--build` compiles this repository's own engine source instead, which is still Strata 0.1.39 with the old-PC changes.
Strata's install guide: [docs/INSTALL.md](docs/INSTALL.md).

## Credits and license

TokenRun is made by **星辰** ([star1872](https://github.com/star1872)). The engine and the original code are
[Strata](https://github.com/Niko1221/Strata) by Niko1221, MIT license; both copyright lines are in [LICENSE](LICENSE).
The llama.cpp builds TokenRun downloads are [ggml-org's](https://github.com/ggml-org/llama.cpp), MIT license.

- **Questions and bugs:** [GitHub Issues](https://github.com/star1872/TokenRun/issues), answered in public so others
  can find them. There is no one-to-one free support.
- **Business and custom work:** cloud17778@outlook.com
