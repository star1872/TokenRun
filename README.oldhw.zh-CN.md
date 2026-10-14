# TokenRun

作者：**星辰**（[star1872](https://github.com/star1872)）· 联系方式见文末。

TokenRun 基于 [Niko1221/Strata](https://github.com/Niko1221/Strata)（MIT 许可证，版权归 Niko1221，原文见 `LICENSE`）。
引擎仍是 Strata；TokenRun 是在它之上的中文网页、模型管理和 llama.cpp 后端。

## 老平台分支（oldhw）

这是 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的一个分支，针对国内常见的老平台：
X99 主板 + 二手至强（只有 AVX2）+ DDR3/DDR4 + RTX 30/40 系显卡 + PCIe 3.0。
模型文件、量化、精度都和 Strata 官方完全相同，只改了运行方式。所有改动都已经以 PR 的形式提交给上游
（见文末），这里是合在一起、可以直接用的版本。

## 实测结果

测试机：Xeon E5-2673 v3（12 核，AVX2）、3×32 GB DDR3-1866、RTX 4060 Ti 16 GB（PCIe 3.0 x8）、SATA SSD、Ubuntu 24.04。
Strata 0.1.39 源码编译（CUDA 12.8），setup 对这台机器的默认配置（64K 上下文、int8 KV）。
4 个提示词（中文作文、改代码、英译中、技术解释），贪心，每次 512 token，不思考；每个配置 3 轮，每轮 2 遍，
官方路径和本分支交替测试，每次启动前释放页缓存。

| 模型（官方 GGUF） | Strata 0.1.39 | 本分支 | 提升 |
| --- | ---: | ---: | ---: |
| **Q2_0**（专家 2.25 bit） | 48.1 tok/s（48.08 / 48.33 / 48.14） | **55.7 tok/s**（55.40 / 55.80 / 55.66） | +15.6% |
| **IQ3_S**（专家 3.33 bit，作者测试中与原模型持平） | 30.5 tok/s（30.52 / 30.51 / 30.51） | **35.0 tok/s**（34.79 / 35.05 / 35.04） | +14.8% |

（中位数；括号里是 3 轮的值。）改代码类任务提升最大（Q2_0 46.5 → 59.6）。提示词读取速度不变
（3,454 token：Q2_0 715 tok/s，IQ3_S 440 tok/s，受 PCIe 3.0 x8 限制）。

精度：用 Strata 自带的 `STRATA_LOGPOS` 对同一段 3,166 token 文本逐 token 比较对数概率，本分支与官方路径的差别
（平均 NLL +0.013）小于只改显存缓存大小造成的差别（+0.012 ～ +0.016），即在 Strata 自身的数值波动范围内。
不改量化、不减少激活专家数。

官方数字（RTX 5070 + Ryzen 5 7600 + DDR5：Q2_0 94、IQ3_S 53 tok/s）是在不同硬件上测的，不能直接比较。

### Unsloth 的 4 bit 级别版本

同一台测试机，同样 4 个提示词、512 token；官方路径和本分支交替测 2 轮，取平均。测试机内存 94 GB，
`--resident-budget-gib` 设到全部专家都在显存 + 内存里、不从 SSD 读（UD-IQ4_XS 55，UD-Q4_K_XL 70）。

| 模型（Unsloth GGUF） | 官方路径 | 本分支 | 提升 |
| --- | ---: | ---: | ---: |
| **UD-IQ4_XS**（专家 3.94 bit） | 19.2 tok/s（19.32 / 19.05） | **20.2 tok/s**（20.18 / 20.24） | +5.2% |
| **UD-Q4_K_XL**（专家 5.10 bit） | 15.5 tok/s（15.68 / 15.30） | **17.0 tok/s**（17.04 / 17.05） | +10% |

UD-Q4_K_XL 各提示词（官方路径 → 本分支）：作文 15.3 → 19.0，改代码 16.6 → 16.55（没有提升），翻译 13.0 → 13.75，
解释 17.1 → 18.9。UD-Q4_K_XL 时显存放 2690 个专家（7.84 GiB），其余 63.89 GiB 锁定在内存。
内存更少的电脑会有部分专家从 SSD 读，速度会更低，没有测过。

### 各版本的速度和困惑度（本分支）

困惑度：同一段 3,166 token 文本，`STRATA_LOGPOS`，不计结束标记，越低越好。只测了一段文本、一次，只作参考。

| 模型 | 专家 bit | 速度 | 困惑度 |
| --- | ---: | ---: | ---: |
| Q2_0 | 2.25 | 55.7 tok/s | 15.4 |
| IQ3_S | 3.33 | 35.0 tok/s | 12.84 |
| UD-IQ4_XS | 3.94 | 20.2 tok/s | 12.55 |
| UD-Q4_K_XL | 5.10 | 17.0 tok/s | 12.12 |

这台机器上 CPU 从内存读专家是瓶颈，所以专家越大越慢：UD-Q4_K_XL 的速度约是 IQ3_S 的一半，困惑度低约 6%。

## 改了什么

1. **AVX2 的 Q2_0 内核**（上游 PR #706，作者 @merbanan）：老至强上原内核在每个专家有 2 个以上 token 时受计算限制，
   新内核在任何 token 数下都能跑满内存带宽。+4.3%。
2. **专家换入显存不再卡住解码**（上游 PR #764，作者 @merbanan，本分支移植到 0.1.39）：慢 PCIe 上窗口开头要等拷贝落地，
   改为晚一个窗口生效。+7%。
3. **显存专家缓存换得更积极**（本分支，上游 PR #907 让 `--calibrate` 自动测这一项）：
   `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`。慢内存机器上 CPU 少算约 30% 的专家。+7.9%。
4. **默认从 ModelScope 下载**（上游 PR #908）：国内直接下载全部模型文件，每个文件按 ModelScope 公布的 SHA-256 校验；
   ModelScope 不通时退回 Hugging Face。`--source huggingface` 可强制用 Hugging Face。
5. **`--inspect`**（上游 PR #911，基于 [quantscope](https://github.com/star1872/quantscope)）：只读文件头（几 MB），
   查出一个 GGUF 的真实 bit 数，并判断 Strata 能不能跑；安装菜单显示各版本专家的真实 bit 数。
6. **PCIe 检查**（上游 PR #912）：第 1 步显示显卡的 PCIe 代数和宽度，插错槽少了 lane 时警告。

## 硬件建议（都有实测依据）

- **内存插对称。** 测试机 3 根 32 GB：两根在一个内存控制器、一根在另一个，只有前 64 GB 是双通道交错，
  其余 32 GB 是单通道。同一个专家内核，从双通道区读 26.3 GB/s，从单通道区读 16.6 GB/s。页缓存占满快区时，
  专家内存会落进慢区，解码最多慢 20 ～ 30%（估算）。补一根同规格的条（本机：32 GB DDR3 LRDIMM 插到空槽）即可全部交错；
  查看自己的插法：`grep . /sys/devices/system/edac/mc/mc*/dimm*/{size,dimm_label}`。
- **显卡插 x16 槽。** setup 第 1 步会显示 `PCIe: 3.0 x8` 之类，少于显卡本身的 lane 数时会警告。
- **内存够 64 GB 用 IQ3_S，追求速度用 Q2_0。** 量化作者的测试：Q2_0 约为原模型的 95.7%（写代码约 93%），
  IQ3_S 约 100%（数学、科学问答持平，写代码 99.3%）。这些是作者测的，不是在 Strata 上测的。
- **内存 96 GB 左右、更看重精度，可以用 Unsloth UD-Q4_K_XL**（`--family unsloth --model UD-Q4_K_XL`，下载 111 GB）：
  测试机上 17.0 tok/s，困惑度 12.12（IQ3_S 35.0 tok/s、12.84），见上面的表。

## 安装（Linux，从源码编译）

本分支改了引擎，setup 下载的官方预编译引擎里没有这些改动，所以要编译：

```bash
git clone https://github.com/star1872/TokenRun.git
cd TokenRun
./setup.sh --build --model IQ3_S        # 或 --model Q2_0；--source 默认 auto（ModelScope 优先）
```

需要：NVIDIA 驱动、CUDA 工具包、gcc（setup 会用 pip 装 cmake/ninja）。RTX 20 ～ 40 系默认要 CUDA 13；只有 CUDA 12 时加
`--cuda 12`（测试机就是 CUDA 12.8）。llama.cpp 的源码包从 GitHub 下载。

**说明：这套 `./setup.sh --build` 流程在本分支上还没有完整跑过一遍**——上面的测试是手动编译引擎、手动准备模型做的
（步骤与 setup 相同：`tools/iq_pack.py` 生成包，`tools/mtp_fetch.py` 等生成草稿层）。遇到问题请开 issue。
启动后在配置文件（`strata-iq3_s.json`）的 `"args"` 里加上：

```
"--adapt-every", "1", "--adapt-swaps", "160", "--adapt-decay", "0.97"
```

或者运行 `./setup.sh --calibrate`，它会在你的机器上实测这三个参数要不要改（以及原有的 PCIe 份额、草稿门槛、线程数）。

Windows：本分支没在 Windows 上测过。

## 踩过的坑

- **Ubuntu 的 systemd-oomd 会杀掉刚启动的引擎**：页缓存占满内存时，加载 30 ～ 47 GB 专家需要边回收边申请，
  内存压力超过阈值约 20 秒就会被杀。启动前释放页缓存可避免（需要权限的话：`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`）。
- **huggingface.co 主页能打开不代表能下载**：国内主页可达，但文件跳转的 CDN 连不上。本分支探测的是真实文件地址。
- **调 calibrate 原有参数在这台机器上没有收益**：它选的 `--pcie-frac 0.20 --spec-min-p 0.70` 在上面 4 个提示词上是
  49.29 vs 49.59 tok/s；引擎启动时自己探测 PCIe 选的 0.17 已经合适。草稿数 `--spec` 默认 4 也是最好的（2/3/5 都更慢）。

完整的测量过程、数据和失败的尝试：上游 issue [#906](https://github.com/Niko1221/Strata/issues/906)。

## 上游 PR

#706（Q2_0 内核）、#764（交换延迟生效）、#907（calibrate 测自适应缓存）、#908（ModelScope）、#911（`--inspect`）、
#912（PCIe 检查）、#913（测试报告）。截至 2026-10-06，#906～#913 还没有维护者回复。

## 作者与联系方式

TokenRun（本分支的网页、模型管理、llama.cpp 后端、安装包和老平台优化）由 **星辰**（GitHub
[star1872](https://github.com/star1872)）开发并开源；引擎和原始代码来自 [Niko1221/Strata](https://github.com/Niko1221/Strata)。

- **使用问题**：请在 GitHub 提 [Issue](https://github.com/star1872/TokenRun/issues)，公开回答，其他人也能看到。个人精力有限，不提供一对一的免费技术支持。
- **商务合作、定制开发**：邮箱 cloud17778@outlook.com，或加微信（请注明来意）：

<img src="docs/media/wechat-qr.png" width="273" alt="星辰的微信二维码">
