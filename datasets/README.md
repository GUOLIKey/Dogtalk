# 两套公开数据的原始快照

本目录包含实际数据；大文件通过 Git LFS 保存。原始文件不重新标注、不重新划分。`SOURCE_MANIFEST.json` 记录固定上游版本、大小与 SHA-256。

| 数据集 | 固定版本 | 音频与标签 |
| --- | --- | --- |
| [Barkopedia Dog Activity & Environment](https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env) | `60a721a7f9477634971f6b3df29dc2a5e56994d8` | 训练 12,480，验证 500，测试 3,120；训练／验证标签；测试标签上游未提供 |
| [EmotionalCanines](https://github.com/tmdang1101/EmotionalCanines) | `6b73c5fce55fc132e97db6a76899f8594a955f59` | husky／shiba 各训练 600、测试 100，共 1,400；四份 CSV 的 `arousal`、`valence` 为情绪标签 |

EmotionalCanines 的品种名称不是情绪标签。DogBark 是程序使用的第三方情绪模型；不应仅凭两者主题相同就断言其权重在本数据集上训练或验证。

## 完整性检查

在仓库根目录运行 `python datasets/verify_datasets.py`。它检查实际字节、大小、哈希和 WAV／ZIP 文件头，LFS 指针会直接报错；不执行模型推理。

## 解压位置

Barkopedia 保持官方 ZIP 原样。原项目固定 DEV 名单使用：

```text
data/extracted/train/split1/split1/audio_XXXXX.wav
data/extracted/train/split2/split2/audio_XXXXX.wav
```

若需要恢复原项目训练数据，应把 `split1.zip` 解压到原项目 `data/extracted/train/split1/`、`split2.zip` 解压到 `data/extracted/train/split2/`；ZIP 内已包含另一层 `split1/` 或 `split2/`。恢复前保留现有目录和锁定划分。验证集、测试集不要加入训练特征或开发调参。

EmotionalCanines 保留 `husky/train`、`husky/test`、`shiba/train`、`shiba/test` 结构；CSV 用 `audio_id` 对应录音。公开数据的官方划分与 PawTalk 的 9,984 条 DEV／锁定留出划分并非同一概念。

## 来源与许可

Barkopedia 原始数据卡标注 MIT，快照保留其 README；上游该版本未单独提供 LICENSE 文件。EmotionalCanines 保留原作者 LICENSE（CC-BY-NC-SA-4.0）和论文引用。不要用仓库整体许可替代外部数据的原有条件。
