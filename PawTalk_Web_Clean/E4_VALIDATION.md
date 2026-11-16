# E4 发布验证（2026-11-16）

## 实际训练与指标

在 Python 3.12.14、numpy 2.1.3、scikit-learn 1.6.1、librosa 0.11.0 下执行了
`experiments/emotion_e4/train_final.py`。下载/校验成功：1,200 条官方 TRAIN 音频和两份标签 CSV；
失败 0。所有文件通过固定版本清单的大小/SHA256 校验。

复用既有 E4 的样本清单、原始外层折和每折内层选择的参数，重新提取/复用版本化特征并拟合外层训练折，
两轴 Macro-F1 与用户报告完全一致（脚本允许的误差为 1e-10）。随后在完整 TRAIN 上进行分组 CV 选参，
两个维度均选择 StandardScaler + RBF SVC，C=1、gamma=0.01、class_weight=balanced，再以全部 TRAIN 拟合。

| 维度 | 开发 OOF Macro-F1 | 开发 OOF Accuracy | 全 TRAIN 内部选参 CV Macro-F1 |
|---|---:|---:|---:|
| Arousal | 0.5075139602 | 0.5066666667 | 0.5074525347 |
| Valence | 0.4298706348 | 0.4366666667 | 0.4371017782 |

最后一列用于全量拟合时选参，**不是新的独立评估成绩**。上述数字也不是最终意图准确率。
本次没有训练或评分官方 TEST。E4 已经根据开发比较被选中，不能将它的 OOF 成绩再称为未经选模的终验。
1,198 个指纹组只控制精确重复；缺少犬只/源录音 ID 的局限仍存在。

## 已执行的功能验证

| 检查 | 结果与边界 |
|---|---|
| Python 主目录回归 | 120 项通过，无跳过；包含真实打包 E4 权重加载、合成 WAV 推理、损坏权重拒绝、完整 HTTP 联调 |
| 六场景专用回归 | 29 项通过；其中训练测试为合成小样本，不是新场景成绩 |
| JavaScript DOM 检查 | 通过；完整 6+3+3 分数、E4 负值/大于 1 分数、原始 predict 首选、空 agreement、版本与页面状态 |
| 录音生命周期 | 8 项通过；释放、取消、权限竞态/拒绝、格式失败、时长/字节上限、重复点击 |
| E4 独立环境准备 | 实际创建新环境、安装依赖、执行权重自检并写入本地运行配置；Linux 环境通过 |
| 真实录音推理 | 在新独立环境中运行 husky_train_00001.wav 和 shiba_train_00001.wav，两轴均返回真实三类决策分数；两条均为训练样本，仅作可运行性检查 |
| 安装器与完整性 | 全量清单安装至临时目录通过；E4 权重复制成功，原程序备份成功，场景模型/历史/密钥配置的测试占位文件保持字节不变；启动完整性检查通过 |
| 真实 HTTP 联调 | 上传合成 WAV → 真实 E4 子进程 → 三候选 → 三次不知道 → 不确定结果 → 导出；场景和 DeepSeek 为模拟接口 |
| 首次及后续 LLM 输入 | 模拟调用检查完整六类场景分数、两轴完整三类 E4 分数与模型版本，不改写原始分析 |
| 历史兼容 | 原 DogBark 的分数解释、历史记录及权重保留，不静默转换成 E4 |
| 冻结检查 | engine.py、live_audio.py、场景标签/训练/缓存、24 条规则及情绪象限配置与父提交无差异 |

可重复执行：

```powershell
python -m unittest discover -p "test_*.py"
python -m unittest discover -s tests -p "test_*.py"
node test_ui_v9.cjs
node test_recorder.cjs
python start_app.py --check
```

Python 测试需安装 `emotion_e4_requirements.txt`；运行应用主入口仍按 README 的原主环境进行。
`test_browser_six.py --browser-executable <Chromium路径> --real-emotion` 提供浏览器联调入口。

## 尚未验证或不能推断的事项

- 当前执行环境的 Chromium 在启动时因 `socket() failed: Operation not permitted` 退出，因此本次**没有完成真实浏览器视觉检查**。DOM 和 HTTP 检查不能替代视觉验收。
- 本次未持有用户的 DeepSeek API 密钥，没有发起真实 DeepSeek 请求。模拟接口通过不代表真实 LLM 的回复质量已验证。
- 没有在用户 Windows 机器上运行 BAT；独立环境准备和推理是在 Linux Python 3.12 上执行，Windows 需按 README 再运行 08 和 00。
- 没有可用的本地场景部署权重，端到端联调中的场景分支采用明确模拟结果；场景源码与规则未变更，不声称重新验证了用户本地场景模型效果。
- 未进行新家庭录音的独立泛化评估、置信度校准或四象限可靠性门槛验证。E4 的 `usable=false` 仅禁止自动确定情绪象限，完整模型输出仍交给 DeepSeek 作为待验证线索。
- 功能测试通过不能证明预测有效；训练样本和合成音频的推理检查也不能作为泛化准确率。
