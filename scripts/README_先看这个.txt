PawTalk STEP 1 — 固定实验划分 + 两类行为的权重优化
=======================================================

适用项目：D:\PawTalk_Barkopedia_数据核查与模型准备包

为什么先做这个？
此前 CLAP + RBF SVM 使用全部 12,480 条训练音频做了探索实验，验证集约 29.6%。
我们现在重点尝试提高 seeking attention（寻求关注）与 begging for food（讨食）的 F1。
但不能直接在以前反复看过的 500 条验证音频上不停改权重，因此本轮先锁定一份保留测试集，
再只在剩余开发训练集内部用三折交叉验证比较权重策略。

【操作】
1. 解压本 ZIP。里面有一个 BAT 和 scripts 文件夹。
2. 把 BAT 复制到旧项目 D:\PawTalk_Barkopedia_数据核查与模型准备包\ 的根目录。
3. 把 scripts 文件夹里的 step1_split_and_weights.py 复制到旧项目的 scripts\ 中。
   （选择“合并文件夹”，不要覆盖原来的脚本，不要替换 data 或 results。）
4. 双击 11_STEP1_固定划分_类别权重优化.bat。
5. 等待显示 [PASS]，将 results\optimization_step1\STEP1_结果摘要.md 上传给 ChatGPT。

脚本做了什么？
- 从原始训练集 12,480 条做固定分层划分（80% 开发 / 20% 锁定测试），随机种子 20261112。
- 仅在开发集上利用 CLAP SQLite 缓存向量（不再重新运行大型 CLAP 推理）。
- 进行 3 折交叉验证，比较：
  * balanced 参考策略（当前类似的原方法）
  * 针对两类的温和权重：关注 x1.25、讨食 x1.5
  * 针对两类的较强权重：关注 x1.5、讨食 x2.0
- 每折权重仅由该折的训练数据计算，避免分类权重的交叉验证泄漏。
- 用开发集 CV 的整体 Macro-F1 选择最优策略，目标类别 F1 仅用作同分时的辅助条件。
- 输出各策略的 OOF 逐条预测、混淆矩阵、分类报告，以及开发集拟合模型。

【生成文件】
results\optimization_step1\
  splits\dev_train.csv
  splits\LOCKED_holdout_DO_NOT_TUNE.csv
  splits\split_protocol.json
  cv_weight_results.csv
  STEP1_结果摘要.md
  selection_summary.json
  selected_CLAP_SVM_DEV_ONLY.joblib
  diagnostics\...

【警告】
- 锁定测试集不参与本轮模型训练、CV、调参或评估。请不要在下一阶段提前打开它的测试成绩。
- 先前基线 29.6% 是“全部训练 + 官方验证”的实验；新一轮 OOF CV 成绩是“开发训练分折”实验，二者不能直接相减当作增幅。
- 无犬只/视频 ID，不能确保同一只狗或视频片段不会跨集合。
- 以前探索性模型曾用过完整训练集；因此这次必须以新划分重新训练所有待比较模型。
- 如果类别权重优化没有提高 Macro-F1，也不是运行失败。这是一个有价值的负结果，不要强行调到正结果。
- 如果报 CLAP cache missing，请先检查原来 10_CLAP_SUPERVISED_FULL.bat 是否跑完，是否移动过录音文件。
