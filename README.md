# Dogtalk — six-scene development branch

最新情绪更新：**E4 rich153 + RBF SVM**。启动、分数含义及验证边界见
[E4 更新说明](PawTalk_Web_Clean/README_E4.md)。已保留场景分支与历史 DogBark 记录。

基于 `GUOLIKey/Dogtalk` 的 `pawtalk-v12-two-datasets-20261115`，在独立分支实施六分类改造。

当前入口：[PawTalk_Web_Clean/README_SIX_SCENES.md](PawTalk_Web_Clean/README_SIX_SCENES.md)，包含标签、24规则、训练评估、启动命令及真实验证边界。

**代码与功能验证已完成；真实六分类重训仍需本地音频、特征与原锁定划分。** 原八类 OOF 映射为六类后，主子集 Macro-F1=0.42426、Accuracy=0.47013。这不是重训成绩，不是原任务提升，也不是意图准确率。

原标签、模型、历史和数据集保留。新模型与缓存独立版本，缺失时明确报错。数据集来源与 Git LFS 说明见 [README_PAWTALK.md](README_PAWTALK.md)，其中旧八类设计属于历史。
