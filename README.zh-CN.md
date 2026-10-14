<h1 align="center">TokenRun</h1>

[English](README.md) · **简体中文**

<p align="center"><b>在自己的电脑上运行大模型，老电脑也能跑</b><br>
对话和模型管理网页 · Strata 和 llama.cpp 两种引擎 · Windows 和 Linux · 免费开源（MIT）</p>

TokenRun 基于 Niko1221 的 [Strata](https://github.com/Niko1221/Strata)。Strata 能在游戏电脑上运行
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（1250 亿参数）。TokenRun 在它之上增加了：

- **老电脑上更快，精度不变。** 在 Haswell 至强 + DDR3 + PCIe 3.0 的电脑上：Strata 引擎加上 TokenRun 的设置，
  比 TokenRun 之前的引擎快 6.5%（之前的引擎又比 Strata 0.1.39 快约 15%）；装不进显存的模型用 llama.cpp 跑，
  打开专家缓存后最多快 47%（测量见下文）。
- **网页。** 模型管理页（检测本机硬件、判断哪个模型能跑、下载、启动、停止）和对话页（多个会话、系统提示词、
  重新生成、修改后重发、数学公式）。中文和英文界面。
- **第二种引擎 llama.cpp。** 任何 GGUF 模型都能在同一个网页里用，不只是 Qwen3.8-Flash-Next。
- **直接能用的安装包。** Windows 解压后双击；Linux 一条命令。不用装 Python 或 git。
- **国内能顺利下载。** 模型、引擎和安装包都有魔搭（ModelScope）下载源，按 SHA-256 校验；魔搭不通时退回 Hugging Face。

## 下载

| | 魔搭（国内推荐） | GitHub |
|---|---|---|
| **Windows x64** | [TokenRun-windows-x64.zip](https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/TokenRun-windows-x64.zip) | [TokenRun-windows-x64.zip](https://github.com/star1872/TokenRun/releases/download/tokenrun-latest/TokenRun-windows-x64.zip) |
| **Linux x86_64** | `curl -fsSL https://www.modelscope.cn/models/mymodel3861/TokenRun/resolve/master/install.sh \| sh` | [TokenRun-linux-x64.tar.gz](https://github.com/star1872/TokenRun/releases/download/tokenrun-latest/TokenRun-linux-x64.tar.gz) |

**Windows：** 整个解压，双击 `TokenRun.exe`。浏览器会打开模型管理页：选模型、下载、启动，然后对话。

**Linux：** 运行上面的命令，然后运行 `tokenrun`。不用 sudo，装在 `~/.local/share/tokenrun`。没有屏幕的服务器：
`tokenrun --host 0.0.0.0`，再从另一台电脑打开 `http://<服务器地址>:8090/`。再运行一次安装命令就是更新，
已下载的模型会保留。

Strata 引擎的硬件要求和 Strata 相同：显卡 12 GB 显存以上，内存 32 GB 以上（64 GB 能跑所有版本），
详见 [Strata 的说明](README.strata.zh-CN.md)。TokenRun 的引擎改动只在 NVIDIA 显卡上测过。
llama.cpp 引擎可以在配置更低的电脑上跑小一些的模型，也可以只用 CPU。

## 老电脑上的速度

测试机：Xeon E5-2673 v3（12 核，只有 AVX2）、3×32 GB DDR3-1866、RTX 4060 Ti 16 GB（PCIe 3.0 x8）、SATA SSD、
Ubuntu 24.04。4 个提示词（中文作文、改代码、英译中、技术解释），贪心，每次 512 token，不思考；各组交替测试，
每次启动前释放页缓存。不同日期的数字不互相比较：这台机器 96 GB 内存里只有 64 GB 是双通道，内存页落在哪里会影响速度。

### Strata 引擎（Qwen3.8-Flash-Next）

TokenRun 用的是官方 Strata 0.1.44.1 引擎（用 Niko1221/Strata 的源码标签编译），加上这些设置（`serve/server.py` 的
`engine_env`）：关闭 route tail skip（`STRATA_ROUTE_TAIL_SKIP=0`；Strata 从 0.1.42 起默认打开，它会跳过排名第 7 以后、
又不在显存里的专家，等于少算了模型选中的专家）、关闭短提示词的 CPU 分担、打开 Q2_0 的 AVX2 内核，
Linux 上让自适应缓存的拷贝和下一个解码窗口同时进行（`STRATA_ADAPT_OVERLAP=1`）。

2026-10-14，IQ3_S，每组 3 轮，4 个提示词平均值的中位数：

| 引擎 | 解码 |
| --- | ---: |
| TokenRun 之前的引擎（Strata 0.1.39 + 下面的老电脑改动） | 40.4 tok/s |
| Strata 0.1.44.1，设置相同但不开 overlap | 40.9 tok/s（+1.2%，在波动范围内） |
| **Strata 0.1.44.1 + TokenRun 的设置** | **43.0 tok/s（+6.5%）** |

最后一行的 3 轮都比第一行的每一轮快；读提示词速度相同（3,454 token 约 440 tok/s）。精度：开不开 overlap，
对数概率完全相同；Strata 0.1.44.1 和之前的引擎在一段 3,166 token 文本上（`STRATA_LOGPOS`）的差别，和只改显存缓存
大小造成的差别一样大（每 token 对数概率平均差 0.248 对 0.253）。没测：Q2_0、Unsloth 的文件、Windows
（Strata 在 Windows 的 RTX 5070 上测到 overlap 慢 2.6%，所以 TokenRun 在 Windows 上不开）。

2026-10-05，老电脑改动和 Strata 0.1.39 的对比（源码编译，CUDA 12.8，64K 上下文），3 轮中位数：

| 模型（官方 GGUF） | Strata 0.1.39 | TokenRun（当时） | 变化 |
| --- | ---: | ---: | ---: |
| **Q2_0**（专家 2.25 bit） | 48.1 tok/s | **55.7 tok/s** | +15.6% |
| **IQ3_S**（专家 3.33 bit） | 30.5 tok/s | **35.0 tok/s** | +14.8% |

16 组（提示词 × 模型）全部更快。Unsloth 的文件提升较小：UD-IQ4_XS +5.2%，UD-Q4_K_XL +10%（2 轮）。
精度：平均 NLL 比 Strata 0.1.39 高 0.013，小于只改显存缓存大小造成的差别（+0.012 ～ +0.016）。
这三项改动从 0.1.40 起都已进入 Strata 本身（在 Strata 里默认关，TokenRun 默认开）：

1. **Q2_0 的 AVX2 计算内核**：上游 PR [#706](https://github.com/Niko1221/Strata/pull/706)（作者 @merbanan，
   `STRATA_Q2_BITPLANE=1`）：+4.3%。
2. **专家换入显存不再卡住解码**（慢 PCIe 上）：上游 PR [#764](https://github.com/Niko1221/Strata/pull/764)
   （作者 @merbanan，`STRATA_ADAPT_LAG`；Linux 上由上面的 overlap 起同样作用）：+7%。
3. **显存专家缓存换得更积极**（`--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`）：我们测出来的；
   [#907](https://github.com/Niko1221/Strata/pull/907) 让 Strata 的 `--calibrate` 在你的电脑上实测这一项：+7.9%。

同样来自 TokenRun、已进入 Strata 的：魔搭下载（#908）、`--inspect`（#911）、PCIe 链路检查（#912）和测试报告（#913）。
Strata 官方数字（RTX 5070 + Ryzen 5 7600 + DDR5：Q2_0 94、IQ3_S 53 tok/s）是在别的硬件上测的。2026-10-05 的完整测量：
[README.oldhw.zh-CN.md](README.oldhw.zh-CN.md) 和上游 issue [Niko1221/Strata#906](https://github.com/Niko1221/Strata/issues/906)。

### llama.cpp 引擎（任何 GGUF 模型）

在模型管理页选「llama.cpp」，就能用 ggml-org 官方编译的 llama.cpp（TokenRun 固定用 b11609）下载并运行任何 GGUF 模型。
MoE 模型装不进显存时，TokenRun 会打开 llama.cpp 的专家缓存（`--moe-cache-mib`，
[llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887)），用一半显存：最常用的专家改在 GPU 上算，
不在 CPU 上算；模型选中的每个专家照样都算。

2026-10-14，Qwen3.6-35B-A3B UD-Q4_K_XL（22.4 GB），`-c 32768 --fit on`，4 个提示词平均值，2 轮：

| 设置 | 解码 |
| --- | ---: |
| b11457（TokenRun 之前的版本） | 34.5 / 32.8 tok/s |
| b11609 | 32.8 / 34.3 tok/s |
| b11609，专家缓存 2048 MiB | 42.5 / 42.0 tok/s |
| b11609，专家缓存 4096 MiB | 46.6 / 46.7 tok/s |
| **b11609，专家缓存 8192 MiB** | **49.4 / 49.6 tok/s（+47%）** |

TokenRun 本身（一份为 b11457 写的旧配置，启动时自动升级；它的规则在这张卡上给出 7936 MiB）：49.0 / 49.1 tok/s。
开缓存后读提示词慢约 4%（3,453 token：346 → 330 tok/s）。没测：其他模型和显卡、Windows。
Qwen3.8-Flash-Next 本身不适合用 llama.cpp 跑（这台机器上 8.0 tok/s，连同 PLE 表 83.6 GB）：请用 Strata 引擎。

## 从源码安装

```bash
git clone https://github.com/star1872/TokenRun.git
cd TokenRun
./setup.sh            # Linux；Windows 用 START-HERE.bat
```

`setup` 会下载 TokenRun 编译的官方 Strata 引擎（GitHub Actions 编译，release `engine-v<版本>`）。
加 `--build` 则编译本仓库自带的引擎源码，它仍是 Strata 0.1.39 加老电脑改动。
Strata 的安装说明：[docs/INSTALL.md](docs/INSTALL.md)。

## 作者与许可证

TokenRun 由 **星辰**（GitHub [star1872](https://github.com/star1872)）开发并开源。引擎和原始代码来自 Niko1221 的
[Strata](https://github.com/Niko1221/Strata)，MIT 许可证，两行版权声明都在 [LICENSE](LICENSE) 里。
TokenRun 下载的 llama.cpp 来自 [ggml-org](https://github.com/ggml-org/llama.cpp)，MIT 许可证。

- **使用问题和 bug**：请提 [GitHub Issue](https://github.com/star1872/TokenRun/issues)，公开回答，其他人也能看到。
  个人精力有限，不提供一对一的免费技术支持。
- **商务合作、定制开发**：邮箱 cloud17778@outlook.com，或加微信（请注明来意）：

<img src="docs/media/wechat-qr.png" width="273" alt="星辰的微信二维码">
