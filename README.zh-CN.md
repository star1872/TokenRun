<h1 align="center">TokenRun</h1>

[English](README.md) · **简体中文**

<p align="center"><b>在自己的电脑上运行大模型，老电脑也能跑</b><br>
对话和模型管理网页 · Strata 和 llama.cpp 两种引擎 · Windows 和 Linux · 免费开源（MIT）</p>

TokenRun 基于 Niko1221 的 [Strata](https://github.com/Niko1221/Strata)。Strata 能在游戏电脑上运行
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（1250 亿参数）。TokenRun 在它之上增加了：

- **老电脑上更快的引擎。** 在 Haswell 至强 + DDR3 + PCIe 3.0 的电脑上，回答速度快约 15%；模型文件相同，精度不变
  （测量见下文）。
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
Ubuntu 24.04。Strata 0.1.39 源码编译（CUDA 12.8），64K 上下文。4 个提示词（中文作文、改代码、英译中、技术解释），
贪心，每次 512 token，不思考。官方 Strata 和 TokenRun 交替测试，各 3 轮，每次启动前释放页缓存。3 轮取中位数：

| 模型（官方 GGUF） | Strata 0.1.39 | TokenRun | 变化 |
| --- | ---: | ---: | ---: |
| **Q2_0**（专家 2.25 bit） | 48.1 tok/s | **55.7 tok/s** | +15.6% |
| **IQ3_S**（专家 3.33 bit） | 30.5 tok/s | **35.0 tok/s** | +14.8% |

16 组（提示词 × 模型）全部更快，改代码提升最大（Q2_0 46.5 → 59.6 tok/s）。Unsloth 的 4 bit 级别版本在同一台机器上
提升较小：UD-IQ4_XS 19.2 → 20.2 tok/s（+5.2%），UD-Q4_K_XL 15.5 → 17.0 tok/s（+10%，测了 2 轮）。

**没有变快的：** 读取提示词（预填充）的速度不变，这台机器上 Q2_0 读 3,454 token 是 715 tok/s。较新的电脑没有做对比。
Strata 官方数字（RTX 5070 + Ryzen 5 7600 + DDR5：Q2_0 94、IQ3_S 53 tok/s）是在不同硬件上测的，不能和上表直接比较。

**精度：** 量化相同，激活的专家数相同。用 Strata 自带的 `STRATA_LOGPOS` 对同一段 3,166 token 文本逐 token 比较对数概率，
TokenRun 与官方 Strata 的差别（平均 NLL +0.013）小于只改显存缓存大小造成的差别（+0.012 ～ +0.016），
而改缓存大小是原版本来就允许的操作。

完整的测量过程、数据和没有效果的尝试：[README.oldhw.zh-CN.md](README.oldhw.zh-CN.md) 和上游 issue
[Niko1221/Strata#906](https://github.com/Niko1221/Strata/issues/906)。

### 速度从哪里来

1. **Q2_0 的 AVX2 计算内核**：来自上游 PR [#706](https://github.com/Niko1221/Strata/pull/706)（作者 @merbanan），
   移植过来并实测：+4.3%。
2. **专家换入显存不再卡住解码**（慢 PCIe 上）：来自上游 PR [#764](https://github.com/Niko1221/Strata/pull/764)
   （作者 @merbanan），移植到 0.1.39：+7%。
3. **显存专家缓存换得更积极**（`--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`）：我们自己测出来的；
   上游 PR [#907](https://github.com/Niko1221/Strata/pull/907) 让 `--calibrate` 在你的电脑上实测这一项：+7.9%。
   内存慢的电脑上，CPU 少算约 30% 的专家。

其他已提交上游的改动：魔搭下载（#908）、`--inspect` 查 GGUF 的真实 bit 数（#911）、PCIe 链路检查（#912）。

## llama.cpp 引擎

在模型管理页选「llama.cpp」，就能用 ggml-org 官方编译的 llama.cpp 下载并运行任何 GGUF 模型。
在上面的测试机上实测：Qwen3.6-35B-A3B UD-Q4_K_XL 生成 40.85 tok/s，读 3,453 token 提示词 384 tok/s。
这是 llama.cpp 本身的速度，TokenRun 没有改 llama.cpp 引擎。

## 从源码安装

```bash
git clone https://github.com/star1872/TokenRun.git
cd TokenRun
./setup.sh            # Linux；Windows 用 START-HERE.bat
```

`setup` 会下载 TokenRun 预编译的 Strata 引擎（GitHub Actions 编译，release `oldhw-v<版本>`），加 `--build` 则自己编译。
Strata 的安装说明：[docs/INSTALL.md](docs/INSTALL.md)。

## 作者与许可证

TokenRun 由 **星辰**（GitHub [star1872](https://github.com/star1872)）开发并开源。引擎和原始代码来自 Niko1221 的
[Strata](https://github.com/Niko1221/Strata)，MIT 许可证，两行版权声明都在 [LICENSE](LICENSE) 里。
TokenRun 下载的 llama.cpp 来自 [ggml-org](https://github.com/ggml-org/llama.cpp)，MIT 许可证。

- **使用问题和 bug**：请提 [GitHub Issue](https://github.com/star1872/TokenRun/issues)，公开回答，其他人也能看到。
  个人精力有限，不提供一对一的免费技术支持。
- **商务合作、定制开发**：邮箱 cloud17778@outlook.com，或加微信（请注明来意）：

<img src="docs/media/wechat-qr.png" width="273" alt="星辰的微信二维码">
