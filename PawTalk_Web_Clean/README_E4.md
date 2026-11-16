# E4 情绪分支更新

本版本将在线情绪推理切换到 **E4：整段录音的 153 维声学统计特征 → StandardScaler → 两个 RBF SVM**。
两个输出分别为 arousal（Low/Medium/High）和 valence（Negative/Neutral/Positive）。
场景分支保持原样；本次没有重新训练或更换场景模型、场景缓存、六类标签及 24 条规则。

## Windows 启动

1. 关闭旧 PawTalk 运行窗口。使用 `feature/six-scene-v1` 分支的完整新版程序。
2. 在原 `PawTalk_Web_Clean` 目录更新程序文件，保留原有 `.venv`、`models`、`data`、`state` 和本地 DeepSeek 配置。先备份原程序目录，不要删除模型和数据。
3. 双击 `08_ENABLE_EMOTION.bat`。它校验包内 E4 权重，创建独立 `.venv_emotion_e4`、安装锁定依赖并执行模型自检。不下载训练集，不重跑实验，不需要 TensorFlow 或 GPU。
4. 出现 `READY：E4情绪模型已启用` 后，双击 `00_START_APP.bat`，使用本次窗口给出的地址。
5. 页面、启动器和后端统一显示 `PawTalk Web Six Scenes v1 + Emotion E4`。上传新录音后，情绪来源应为 `E4 rich153 + RBF SVM`，模型版本为 `emotion-e4-rich153-svm-v1`。

`08` 沿用项目原先的 `..\.venv\Scripts\python.exe` 启动入口，需要原主环境仍存在。
若仅全新下载本仓库而没有本地场景权重/DEV 索引，它们仍需按 `README_SIX_SCENES.md` 准备。
本更新不把仓库缺少的场景权重伪装为已准备。DeepSeek 密钥仍使用本地配置，不上传到 GitHub。

## 分数和可用性

- E4 的 `decision_function` 是未校准决策分数，可以小于 0 或大于 1，**不是概率或准确率**。三个分数不要求和为 1。
- 按模型真实 `classes_` 对齐类别；保留 `predict` 首选及完整分数、排序和来源。`predict` 首选可能不同于决策分数最大项。
- E4 用整段统计特征，没有分段一致性投票，因此 `agreement=null`，不会填造 100% 一致率。
- `available=true` 表示推理成功；`usable=false` 表示**尚无经验证的自动四象限判定门槛**。这是两种不同状态，不代表模型没有运行。不能沿用旧 DogBark 的 0～1 分数阈值。
- DeepSeek 仍会收到两轴全部分数、首选、可用性、模型版本和不确定性，作为待核查线索；未通过可靠性门槛时不自动匹配四象限规则。
- 旧 DogBark 权重、旧环境和历史记录保留原含义。已有历史记录不会自动改写为 E4 结果。需要 E4 结果时重新上传录音。

## 模型来源和验证

部署权重由固定上游版本的 1,200 条官方 TRAIN 录音重新拟合，指纹分组数为 1,198。
`emotion_e4_assets/metadata.json` 保存实际选参、库版本、训练来源、特征代码哈希和开发评估指标。
`emotion_e4_spec.json` 和程序完整性清单共同校验发布权重。

用户原 E4 开发 OOF：arousal Macro-F1=0.507514、Accuracy=0.506667；
valence Macro-F1=0.429871、Accuracy=0.436667。这些不是整套 App 的意图准确率，也不是全量拟合后的独立测试成绩。
具体复现及测试结果见 `E4_VALIDATION.md`；训练命令见 `../experiments/emotion_e4/README.md`。

缺少犬只/源录音 ID，无法仅靠音频指纹证明跨犬只泛化。模型置信度门槛、实际家庭录音效果和最终意图效果尚需独立验证。
问答仅更新本次意图判断，不修改声音模型权重。

## 失败排查

`08` 失败：查看 `EMOTION_SETUP_ERROR.txt`。上传后情绪失败：查看 `EMOTION_ERROR.txt`。
缺失或损坏权重会明确失败，不回退为随机输出、演示结果或旧 DogBark 结果。
提交排查日志时不要包含 DeepSeek 密钥。
