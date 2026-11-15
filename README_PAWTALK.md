# PawTalk v12：程序、双数据集与最新设计说明

本分支基于 `GUOLIKey/Dogtalk` 的 `b186c8599ea5a3123c3e4ea9837cfa1cdd61d1bd`，保留原始 `README.md`，不修改 `main`。程序来自 `PawTalk_Web_Intent_v12_Full.zip`，不更改原版运行逻辑。

## 目录

| 目录 | 内容 |
| --- | --- |
| `PawTalk_Web_Clean/` | v12 完整源码、网页、Windows 启动与准备脚本、测试、固定 DEV 名单 |
| `datasets/Barkopedia_Dog_Act_Env/` | 官方训练、验证、测试音频 ZIP，训练／验证标签和来源校验清单 |
| `datasets/EmotionalCanines/` | 1,400 条实际 WAV、四份 arousal/valence 标签 CSV、原作者说明与许可证 |
| `docs/math_and_workflow/` | 最新中文 Word、LaTeX 公式、英文 PNG/SVG 流程图与拟议融合配置 |

## 下载含完整数据的版本

音频及大型 ZIP 用 Git LFS 保存；只下载源码 ZIP 或只看到 LFS 指针，不代表取得了数据。安装 Git LFS 后运行：

```bash
git lfs install
git clone --branch pawtalk-v12-two-datasets-20261115 https://github.com/GUOLIKey/Dogtalk.git
cd Dogtalk
git lfs pull
python datasets/verify_datasets.py
```

## 实际已实现与设计状态

已实现 v12：录音＋描述／问题输入；场景八类与情绪三类唤醒度＋三类效价输出；描述充分可零追问；信息不足逐次追问，最多三问；仍不明确则给最多三种解释。

**60∶40 声音／观察融合、综合支持分 60、领先差 10 的规则仍是设计方案，未写入本版运行逻辑，也不是已验证的正确概率。** 这些材料在 `docs/`，不要将其配置复制到程序中并声称功能已部署。模型常驻和两路并行等加速建议也尚未实施。

## 运行与本机资源

对于已经跑通的 Windows 项目：关闭旧程序，把 `PawTalk_Web_Clean/` 内的文件复制进原项目的同名目录并替换同名文件，保留原目录中的模型、配置、数据与环境，再运行 `00_START_APP.bat`。新建一次录音分析使用新版流程。

**这是原 v12 源码及公开数据的归档，不是带全部模型的即开即用镜像。** 本分支不包含：你本机的 `models/live_svm.joblib`、`models/live_meta.json`、CLAP 缓存、情绪运行环境／权重、既有特征缓存、原训练 `scripts/`、DEV 音频索引、DeepSeek 密钥和个人会话。`setup_live.py` 与 `prepare_audio.py` 依赖原训练项目的脚本及特征；仅取得原始数据无法立即替代这些资源。`08_ENABLE_EMOTION.bat` 可按原流程取得固定版本情绪权重。

原始数据独立归档在 `datasets/`，不会自动重训、覆盖固定划分或建立网页索引。数据结构与校验见 [datasets/README.md](datasets/README.md)。

## 验证（2026-11-15）

- v12 全部 88 个程序包文件保持原始字节一致。
- `python -m unittest discover -q`：110 项通过。
- `node --test test_recorder.cjs test_emotion_ui.cjs test_ui_v9.cjs`：通过。
- Barkopedia 官方大型文件 SHA-256 核对；ZIP CRC 校验；EmotionalCanines WAV 与上游 LFS 对象 SHA-256 核对；两套数据标签和音频清单核对。

程序测试使用模拟模型／API；没有执行真实 DeepSeek 请求、加载本机模型或重新测得意图准确率。

## 资源许可

每个数据集保留独立来源与条件，不因放入本仓库而改变。Barkopedia 官方数据卡标记 MIT；EmotionalCanines 原作者许可为 CC-BY-NC-SA-4.0（署名、非商业、相同方式共享）。本分支不额外授予外部数据或模型的许可。详见各数据集的 `README.md`、`LICENSE`（如上游提供）与 `SOURCE_MANIFEST.json`。
