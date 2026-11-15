# PawTalk Web Six Scenes v1：实施与验证

基于 **GUOLIKey/Dogtalk** 的 `pawtalk-v12-two-datasets-20261115`，基线提交 `f6391046896e246f5c61c157ab124ad95ed654df`。实际程序目录在仓库根目录下 `PawTalk_Web_Clean/`，不是旧仓库 Dog_talk。

**状态：六分类训练、评估及应用代码已实现；真实 Barkopedia 六分类训练尚未完成。** 本次没有新训练权重，不用旧八分类、演示或随机值代替。已实际完成历史 OOF 预测映射统计和功能验证；测试通过不代表预测有效。

## 1. 业务价值与唯一标签定义

先判断是否与玩耍有关，游戏对象通过用户观察补充。这是重新定义预测目标，不证明原任务准确率提高，也不保证新训练取得更高成绩。唯一活动配置是 `scene_labels.json`，版本 `scene-six-v1`。

| ID | 新场景 | 对应原始标签 |
|---|---|---|
| S01 | 对声音警觉 | alerting to sounds |
| S02 | 讨食 | begging for food |
| S03 | 玩耍相关情境 | playing with human / playing with other animals / playing with toy |
| S04 | 休息 | rest |
| S05 | 寻求关注 | seeking attention |
| S06 | 洗澡 | taking shower |

原始 DEV 清单、八类模型、标签和历史结果不覆盖；另存 `actual_original`、`target_six`。旧 A1—A8 不获得新含义，带版本的旧配置与32条规则归档到 `legacy_config_v02/`；旧准备脚本在 `legacy_training_eight/`，仅作源码归档，不是运行入口。原 I01—I06 意图库不因场景合并而删除。

## 2. 数据链路和主要文件

上传/录音 → 六类场景与原情绪模型独立分析 → 完整6+3+3分数、来源及不确定性 → DeepSeek 综合候选 → 用户观察核查 → 有依据的判断或不确定。

保留 v12 的描述优先、用户问题识别、每轮一个问题和最多三问。候选通常三个、最多三个，证据不足时可以更少。用户观察可更新判断、纠正此前回答和恢复候选，不能改写原声音输出；不知道不等于否定，不强迫唯一结果，不重训练声音模型。

| 文件/目录 | 实际职责 |
|---|---|
| scene_labels.json / scene_schema.py | 唯一标签定义、映射、配置摘要、实际 classes_ 解析 |
| scene_training.py | 历史审计、原特征导出/重新提取、同折八类与六类训练及评估 |
| scene_cache.py | 独立 OOF 缓存、版本/模型/划分身份和六类分数校验 |
| live_audio.py / engine.py | 原 CLAP512+MFCC30 特征和真正六分类推理 |
| server.py / analysis_contract.py | 上传双路分析、结构校验、元数据、历史与导出 |
| dialogue.py / intent_system_v4.txt | 完整分数输入、独立意图、观察与引用校验；复用 llm_client.py |
| intent_rules.py / intent_link_rules.json / intent_rule_reverse_index.json | 24个有效入口和反向索引 |
| emotion_quadrants.json | 原情绪象限独立定义；两轴模型和处理保持不变 |
| static/app.js / static/index.html | 六类表格、版本与来源、双路结果、候选及问答 |
| legacy_scene_engine.py / legacy_activity.py | 旧八类场景评估保持原语义 |
| evaluate_intents.py | 独立意图评估，不把场景标签当意图真值 |

模型版本 `mfcc-clap-svm-six-v1`，缓存 `scene-six-oof-v1`，新会话 `intent-session-six-v1`。模型新路径 `models/scene_six_v1/{live_svm.joblib,live_meta.json}`，缓存新路径 `data/scene_six_v1/audio_index.json`。旧文件不会作为六类使用。

输出保留标签→分数映射、原始 predict 首选、决策分数排名、score_kind、classes_、模型/标签版本和来源。实际数组按 classes_ 对齐，显示按配置排序，首选不强行改成分数最大项。SVM decision_function 是决策分数，不是概率或正确率；不相加旧玩耍分数，不用 Softmax 冒充校准。

DeepSeek 收到完整六类场景与 arousal、valence 各三类的分数、可用性和不确定性；不听原音频。提示词白名单排除 actual/gold/真实标签以及按标签编造的背景。真实用户观察可输入；模拟背景不能伪装观察。缺模型、依赖、坏缓存或 API 失败时明确报错。

旧八类会话只读保留原标签体系/版本和过程，不静默迁移到新问答；正常导出不泄漏真实标签。参考标签在持久化记录中仍保留。场景评估与意图评估独立，没有意图真值时不能计算意图准确率。

## 3. 24条场景与情绪规则

S01—S06 × E1—E4 共24个唯一入口。**每个有效组合唯一匹配一条规则，不代表唯一确定真实意图。** 原意图类别保持不变。

| 象限 | 情绪组合 | 合并后的 S03 规则 |
|---|---|---|
| E1 | Positive + High | 优先核查开始/继续游戏，不能直接确认 |
| E2 | Negative + High | 核查接触、回避、退开、主动返回，不直接认定停止或远离 |
| E3 | Positive + Low | 核查轻度互动或回应，不推断游戏对象 |
| E4 | Negative + Low | 保留不确定，询问具体接触/回避观察 |

S03 单独重写，不平均旧三类规则；模型不知道对象是人、动物还是玩具。其他场景保留原核查含义。Medium、Neutral、未知、低区分度和 unusable 状态正常回退，不强迫归入四象限。情绪完整两轴输出始终保留。

## 4. 已实际获得的结果

来源为 `dev_manifest.csv` 中已存 A_pred OOF 预测，DEV 共9,984条，主评估子集时长≥1秒，共4,954条。本次历史统计不重跑音频模型；只映射原最终预测标签，不处理或相加原 SVM 分数。

| 方案 | 评估目标 | 主子集 Accuracy | 主子集 Macro-F1 |
|---|---|---:|---:|
| 原八分类已存 OOF | 八分类 | 0.35244 | 0.35699 |
| 原预测标签映射六类 | 六分类 | 0.47013 | 0.42426 |
| 真正重新训练六分类 | 六分类 | 未运行 | 未运行 |

映射六类三折 Accuracy 均值±样本标准差为 **0.47023±0.00936**，Macro-F1 为 **0.42405±0.00779**。上表是合并 OOF 样本计算的指标，和折均值不同。全部9,984条映射六类 Accuracy=0.44501、Macro-F1=0.38730。

| 主子集类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
| S01 | 0.36269 | 0.42013 | 0.38930 | 745 |
| S02 | 0.37743 | 0.46190 | 0.41542 | 210 |
| S03 | 0.60508 | 0.58061 | 0.59259 | 2053 |
| S04 | 0.39932 | 0.34211 | 0.36850 | 684 |
| S05 | 0.39938 | 0.34771 | 0.37176 | 742 |
| S06 | 0.37184 | 0.45192 | 0.40799 | 520 |

完整逐折指标、两套子集、混淆矩阵 CSV/PNG、比较表在 `results/scene_six_v1/historical_audit/`。逐样本映射表可以用 audit 命令重建。

这是不同任务的历史描述统计，不能用六类 F1 减八类 F1 宣称原任务准确率提升。历史清单缺少可审计的音频指纹、犬只和源录音身份，不能证明旧划分没有泄漏；这不是修正划分后重新训练的公平比较，更不是最终意图准确率。

## 5. 同折重新训练与评估协议

固定标签再评估。原方案特征为 CLAP512（前10秒、原始向量）+ MFCC30（整段），训练折内分别 StandardScaler，特征块系数 sqrt(0.5/512) 和 sqrt(0.5/30)。RBF SVC 的 C=1、gamma=2、class_weight=balanced。八类和六类使用同一工厂、相同样本与划分、相同超参数，不额外搜索；balanced 根据各自目标频次计算，不是意图融合权重。

训练前核查文件 SHA256、解码 PCM 指纹，以及清单提供的 dog_id、source_recording_id，合并传递关联分组。若原三折跨组，则两个模型都在同一 StratifiedGroupKFold 三折（seed=20261112）重新训练；否则保留原折。缺失身份信息会写入审计，精确指纹不能检出全部重采样副本或重叠切片。

训练要求原锁定留出清单，只核对路径和已提供身份/指纹；不解码、预测或调参使用留出音频。没有留出指纹时，不宣称完成跨集合音频内容检查。

输出三方案比较、逐类指标、混淆矩阵、逐折结果和均值/样本标准差、分组审计、同折 OOF、各折八类/六类模型、全 DEV 最终六类模型和元数据。原八类文件不覆盖，新实验目录必须为空，合成测试模型禁止安装。

## 6. Windows 准备与启动

新增双击入口：09_TRAIN_SIX_AND_SHOW_F1.bat（检查、训练、安装并显示结果）、10_VIEW_SCENE_RESULTS.bat（查看最近完成的结果）、11_TRAIN_FROM_AUDIO.bat（明确重新提取音频特征的备用路线）。逐步操作见 START_HERE_SIX_F1.md。下方命令行仍可使用。

先备份原程序目录、关闭旧运行窗口。将新版 PawTalk_Web_Clean **里面的内容**覆盖到原同名文件夹里面，不能再嵌套一层；保留旧模型、数据、密钥和历史。本版需要新的真实六类模型，旧版“无需重训”的说明不再适用。

以下在原 `PawTalk_Web_Clean` 目录的 PowerShell 执行。沿用原来能跑模型的 `..\.venv`，不要直接升级现有推理环境；路径不同则替换 Python 路径。

```powershell
# 资源检查，不调用 API
& "..\.venv\Scripts\python.exe" scene_preflight.py --project-root .. --features data/scene_features_v1.npz --holdout-manifest ../results/optimization_step1/splits/LOCKED_holdout_DO_NOT_TUNE.csv --output results/scene_six_v1/preflight.json

# 可立即复现历史统计，不需要权重和音频
& "..\.venv\Scripts\python.exe" scene_training.py audit --output results/scene_six_v1/historical_audit
& "..\.venv\Scripts\python.exe" plot_scene_results.py --results results/scene_six_v1/historical_audit

# 特征路线 A：有原 scripts、缓存和 models/live_meta.json 时优先使用
& "..\.venv\Scripts\python.exe" scene_training.py export-existing --project-root .. --output data/scene_features_v1.npz

# 特征路线 B：无法用原缓存时，改从实际 DEV 音频提取；与 A 二选一
# 需要 torch、transformers、librosa、soundfile 和冻结 CLAP 权重
& "..\.venv\Scripts\python.exe" scene_training.py extract --project-root .. --output data/scene_features_v1.npz

# 同折跑两个模型，输出目录必须为空；安装新六类模型和缓存
& "..\.venv\Scripts\python.exe" scene_training.py train --project-root .. --features data/scene_features_v1.npz --holdout-manifest ../results/optimization_step1/splits/LOCKED_holdout_DO_NOT_TUNE.csv --output results/scene_six_v1/trained_run_01 --install

# 准备/核对实时编码器；不会把旧八类权重转为六类
& "..\.venv\Scripts\python.exe" setup_live.py
& "..\.venv\Scripts\python.exe" start_app.py --check
.\00_START_APP.bat
```

A 路线使用原 `scripts/step4b1_denoise_abc.py`、`scripts/step3_svm_tuning.py`，核对路径顺序、真值和旧预测，并在小样本上核对原训练工厂与新工厂预测/分数一致性；不一致则停止。B 路线保存实际提取库版本，不声称与原缓存逐位一致。CLAP 可用 CUDA 加速，sklearn SVC 使用 CPU。即使有特征缓存，仍需要实际 DEV 文件检查指纹。

清单路径以原实验项目根目录为准，例如 `data/extracted/train/split1/split1/audio_02407.wav`。Git LFS 小指针不等于实际音频；需要下载并解压，保持清单相对路径。两个原始数据集不等于原实验缓存或锁定划分。不要重切留出集或改名绕过检查。

01 准备脚本现在检查新六类缓存，07 只准备/核对已训练六类的实时环境。情绪准备入口和 DeepSeek 本地私有配置保留，不上传密钥。浏览器与终端应显示 **PawTalk Web Six Scenes v1**。

## 7. 验证与缺失项

```bash
python -m unittest discover -p 'test_*.py'
python -m unittest discover -s tests -p 'test_*.py'
node test_ui_v9.cjs
node test_recorder.cjs
node --check static/app.js
python start_app.py --check
# 可选真实浏览器交互；ML/API 是明确的测试替身，需要可启动的 Chromium
python test_browser_six.py
```

已通过110项原项目 Python 测试、22项新增测试、前端 DOM 测试、8项录音交互测试。包括映射、乱序 classes_、新旧缓存隔离、24入口、情绪回退、6+3+3提示词、标签不泄漏、未知回答、候选状态、历史和导出。

5项训练测试使用48条合成特征/音频真实拟合 SVM，验证同折训练、分组修复、输出和安装保护；不是 Barkopedia 性能验证。真实浏览器测试在当前环境因 Chromium socket 权限错误于启动阶段退出，**未完成**；DOM 测试不能替代它。

正确仓库检出时：9,984条 DEV 路径无实际音频，数据集有1,404个 LFS 指针；缺原 CLAP/MFCC 特征包、锁定留出清单、原训练工厂及完整缓存、六类权重、models/emotion_runtime.json、DeepSeek 配置。测试环境无 torch/transformers，也无可用 GPU。详见 resource_status.json 和 training_blocked.log。

因此真实六类训练、两模型真实联合推理和真实 DeepSeek 调用尚未验证，不能报告其指标。没有对锁定留出集做解码或评分。

测试依赖在 requirements_scene_test.txt，仅用于隔离功能验证环境。文件完整性机制保留；维护者改源码后运行 build_integrity.py 更新清单，不用自动重建掩盖部署文件混用。旧 README_INTENT_V12/V11、VALIDATION 和 docs/math_and_workflow 中八类/32规则内容是历史档案，本版以本文件和当前配置为准。
