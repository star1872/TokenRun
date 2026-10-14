<h1 align="center">TokenRun</h1>

**English** · [简体中文](README.zh-CN.md)

<p align="center"><b>Run large AI models on your own PC, including older ones</b><br>
Web app for chat and model management · Strata and llama.cpp engines · Windows and Linux · free and open source (MIT)</p>

TokenRun is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221. Strata runs
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 125-billion-parameter model, on a gaming PC.
TokenRun adds:

- **A faster engine on older PCs.** About 15% faster replies on a Haswell Xeon + DDR3 + PCIe 3.0 PC, with the same
  model files and the same accuracy (measurements below).
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
Ubuntu 24.04. Strata 0.1.39 compiled from source (CUDA 12.8), 64K context. Four prompts (Chinese essay, code edit,
English-to-Chinese translation, technical explanation), greedy, 512 tokens each, thinking off. Official Strata and
TokenRun were run in turns, 3 rounds each, page cache dropped before each start. Median of the 3 rounds:

| Model (official GGUF) | Strata 0.1.39 | TokenRun | Change |
| --- | ---: | ---: | ---: |
| **Q2_0** (experts 2.25 bit) | 48.1 tok/s | **55.7 tok/s** | +15.6% |
| **IQ3_S** (experts 3.33 bit) | 30.5 tok/s | **35.0 tok/s** | +14.8% |

All 16 prompt/model pairs were faster; code editing gained the most (Q2_0 46.5 → 59.6 tok/s). Unsloth's 4-bit-class
files gained less on the same PC: UD-IQ4_XS 19.2 → 20.2 tok/s (+5.2%), UD-Q4_K_XL 15.5 → 17.0 tok/s (+10%; 2 rounds).

**Not faster:** reading the prompt (prefill) is unchanged, 715 tok/s for Q2_0 on 3,454 tokens on this PC. Newer PCs
were not compared. Strata's own numbers (RTX 5070 + Ryzen 5 7600 + DDR5: Q2_0 94, IQ3_S 53 tok/s) are from different
hardware and cannot be compared with the table above.

**Accuracy:** same quantization, same number of active experts. Comparing per-token log-probabilities on a
3,166-token text with Strata's `STRATA_LOGPOS`, TokenRun differs from official Strata by +0.013 mean NLL, less than
changing only the GPU cache size does (+0.012 to +0.016), which Strata already allows.

The full measurement, data and the attempts that did not help: [Niko1221/Strata#906](https://github.com/Niko1221/Strata/issues/906)
and (in Chinese) [README.oldhw.zh-CN.md](README.oldhw.zh-CN.md).

### Where the speed comes from

1. **AVX2 kernel for Q2_0** from upstream PR [#706](https://github.com/Niko1221/Strata/pull/706) by @merbanan,
   ported and measured here: +4.3%.
2. **Expert swaps no longer stall decoding** on slow PCIe, upstream PR [#764](https://github.com/Niko1221/Strata/pull/764)
   by @merbanan, ported to 0.1.39: +7%.
3. **A more active GPU expert cache** (`--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`), measured here; upstream
   PR [#907](https://github.com/Niko1221/Strata/pull/907) lets `--calibrate` test it on your PC: +7.9%. The CPU computes
   about 30% fewer experts on a PC with slow RAM.

Other changes sent upstream: ModelScope downloads (#908), `--inspect` to read a GGUF's real bit width (#911), PCIe
link check (#912).

## llama.cpp engine

Pick "llama.cpp" in the model manager to download and run any GGUF model with ggml-org's official llama.cpp builds.
Measured on the test PC above: Qwen3.6-35B-A3B UD-Q4_K_XL at 40.85 tok/s writing, 384 tok/s reading a 3,453-token
prompt. That is llama.cpp's own speed; TokenRun does not change the llama.cpp engine.

## From source

```bash
git clone https://github.com/star1872/TokenRun.git
cd TokenRun
./setup.sh            # Linux; START-HERE.bat on Windows
```

`setup` downloads TokenRun's ready-made Strata engine (built by GitHub Actions, release `oldhw-v<version>`), or
compiles it with `--build`. Strata's install guide: [docs/INSTALL.md](docs/INSTALL.md).

## Credits and license

TokenRun is made by **星辰** ([star1872](https://github.com/star1872)). The engine and the original code are
[Strata](https://github.com/Niko1221/Strata) by Niko1221, MIT license; both copyright lines are in [LICENSE](LICENSE).
The llama.cpp builds TokenRun downloads are [ggml-org's](https://github.com/ggml-org/llama.cpp), MIT license.

- **Questions and bugs:** [GitHub Issues](https://github.com/star1872/TokenRun/issues), answered in public so others
  can find them. There is no one-to-one free support.
- **Business and custom work:** cloud17778@outlook.com
