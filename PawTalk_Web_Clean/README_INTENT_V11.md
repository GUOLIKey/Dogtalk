# PawTalk Web Intent v11 — 实现交接 v2 / 规则 v0.2

依据 `DogTalk_AI_完整开发提示词+(1).md` 的完整替代版实现。远端参考 HEAD 核对为 `4f14eab7ddad4a58bd501b479cee63b5479843d7`，在已完成上传修复与来源格式修复的本地代码上继续改造。未推送或覆盖远端。

## 启动与升级

已有 v10 能完成上传、两路推理的用户：关闭旧运行窗口，解压完整压缩包；运行包根目录 `INSTALL_UPDATE.bat`，选择你原来的 `PawTalk_Web_Clean` 文件夹。更新器先校验新程序、备份被替换的程序文件，再复制新版代码。不会复制、删除或覆盖 `data`、`models`、`state`、`.venv_emotion`、`DEEPSEEK_CONFIG.ini`。不用重做 01、07、08，不用重训，也不用重新填写密钥。

更新完成后，在原文件夹运行 `00_START_APP.bat`，使用本次窗口打印的地址。窗口和网页应显示 **PawTalk Web Intent v11**。先运行 `06_CHECK_INSTALL.bat` 可以检查源码一致性（不调用 API）。若手工安装，将包内 `PawTalk_Web_Clean` 的程序文件合并到原同名文件夹并替换同名程序；不要删除整个旧文件夹。

首次部署：本目录与原工程 `.venv`、`scripts`、`data` 同级。00 使用父目录 `.venv\Scripts\python.exe`。04 在本机配置 DeepSeek；密钥只保存在本机。只有资源缺失时才运行 01/07/08；07 涉及外部训练脚本，已有兼容模型时无需运行。

## 新流程与数据

上传/浏览器录音 → 场景完整 8 类 + 情绪完整 3+3 类 → 后端派生情绪摘要与规则 → DeepSeek 最多三个假设 → 单问核查 → 主要/并存/未决结论。

- `intent_rules.py` + `model_business_ids.json`：显式英文标签到 A1–A8，固定 E1–E4 转换；不按数组位置猜业务 ID。
- `intent_link_rules.json`：32 条唯一入口，全部 `observation_required=true`；原样保留 null/空池。
- `intent_rule_reverse_index.json`：由正向池自动生成，启动时核对一致性。
- `question_templates.json`：Q01–Q12 中性观察问题；`intent_catalog.json` 六类目的假设 v0.2。
- `emotion_summary`：quadrant / central / uncertain / unavailable；保留 central_axes、reason_codes、mapping_version、usable_version。两轴需 usable 且无并列；不计算联合概率。
- `rule_context`：anchor_scene 来自原始 predict；原始首选放第一，再按原始分数取其后两类辅助入口。保存主/辅助角色、次序、池、回退。DeepSeek 仍收到全部分数。
- `dialogue.py` / `intent_system_v3.txt`：初始独立生成与逐轮核查；原值引用、规则引用、观察来源与更正版本校验。允许 0 个候选先问中性背景；空池不是程序错误。`subtype` 未确认字段 null，只有有效支持观察可填写。
- `server.py`：新会话 `intent-session-v3`，保存原模型、不可变摘要、规则版本、候选快照、候选轮次、有效回答、修订与结果。旧记录只读保留，不迁移成新版意图。
- `question_budget`：limit=8 / used=已成功提交并展示的问题数 / remaining。明确重新生成不重置；重复请求、失败请求不重复计数，更正后相同待答问题不重复计数，新问题计数。耗尽后允许根据已有观察重评，但不再提问。
- `analysis_contract.py` / `live_audio.py`：保留并验证实际 classes_ 顺序；历史缓存没记录顺序时明确 unknown，不捏造。SVM 分数仍为 decision_function，不是概率。
- `static/app.js` / `index.html`：分别显示执行、原分数、派生摘要/规则、候选问答、最终判断；保留长按录音、后台上传状态与恢复。
- `evaluate_intents.py`：仅新版独立意图参考的配对 A/B/C；主要与集合指标、并存结论覆盖、跨轮问题数。旧场景统计不纳入新版意图记录。

情绪 usable 门槛保持兼容：max≥0.60、margin≥0.15、分段 agreement≥0.60。这是显示规则，不是已校准准确率。当前情绪模型每 2 秒段只使用开头约 0.6 秒 MFCC 特征，未在本项目验证情绪准确率；没有独立非狗叫检测器。

## 接口契约与限制

真实运行契约在 `dialogue.py` 的 generate_intents / next_turn 内随请求发送；JSON Schema 另见 `intent_output_schemas.json`。输入含 phase、catalog、完整模型输出、摘要、规则、所有有效观察/本轮历史、候选与共享预算；不传音频、gold/actual 或模拟背景。A 不传模型及模型派生信息；B 不传情绪或象限检索规则；C 使用完整双路。实验方案仍在两路成功后选择，未伪装单路缺失为完整流程。

本实现字段适配：初始 `intents` 对应 candidates，`supporting_signals` 对应 model_signals；更新 `assessments` 对应 active_candidates；`selected_intent_id` 对应 primary_intent_id。证据带 answer_id、quote、role、answer_revision 和简短依据。问答结果为 concluded / coexisting_supported / multiple_possible / outside_candidates / insufficient_evidence / user_stopped。

来源字段只证明存在于当前输入，不证明解释正确。保守词面门槛遇到否定、假设、过去/现在混合、未观察或猜测时可能保留 possible，需要进一步澄清；没有声称完整验证语义。I04/I06 仅相同回避证据不足以选主要目标，双方独立获支持而无主次时可并存。问答不训练任何声音模型。

结构校验失败最多再请求一次修正；每次底层 JSON 请求本身最多两次，所以极端情况下一个阶段最多四次 API 请求。认证/网络错误不作该结构修正重试；失败保留错误并允许用户重试，不填默认候选。不要把多次手动点击重试理解为免费调用。

## 本环境资源缺失与验证边界

未附并且本环境没有：

- `models/live_svm.joblib`、`models/live_meta.json`（已有 SVM 与元信息）；CLAP `laion/clap-htsat-unfused` 的本地权重缓存。
- `models/emotion/final_arousal_model.keras`、`models/emotion/final_valence_model.keras`，`models/emotion_runtime.json` 与所指的隔离 TensorFlow Python 环境。
- `data/audio_index.json` 与原 DEV 音频及特征缓存（仅缓存回放需要）。
- 本机 `DEEPSEEK_CONFIG.ini` 或相应环境密钥；不包含密钥，不要求公开发送。
- 父目录 `scripts/step3_svm_tuning.py`、`scripts/step4b1_denoise_abc.py` 等原工程训练/准备资源，首次建模型时才需要核对。

因此本次未实际加载上述模型、未真实调用 DeepSeek，未测 Windows 本机推理时延，也不能报告真实意图准确率提高。已完成的 HTTP、数据库、规则、前端 DOM、音频解码和接口测试范围见 `VALIDATION_V11.md`。
