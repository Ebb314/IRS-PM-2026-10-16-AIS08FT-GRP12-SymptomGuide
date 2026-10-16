# 患者—疾病配对训练

## 两种“正负”含义

症状有无：患者明确有症状、明确无症状、未知，是模型输入。

训练配对：患者证据与疾病匹配为正对 `(patient, disease+)`，与对照疾病配对为负对 `(patient, disease-)`。带有阴性症状的患者仍然可以与正确疾病组成正对。它们不是“有症状是正训练样本、无症状是负训练样本”。

原模型采用全部疾病的 Softmax Cross-Entropy：提升记录疾病的相对得分，同时与所有其他疾病竞争。它没有显式抽取正负对，也没有使用 pairwise loss。

新实验比较：

| 版本 | 训练目标 |
| --- | --- |
| cross_entropy | 已训练的三态 CE baseline |
| pairwise | BPR 风格配对排序损失 |
| ce_pairwise | CE + 0.2 × 配对排序损失 |

配对损失：`softplus(score_negative - score_positive)`，鼓励同一个患者的正确疾病得分高于负疾病。[BPR 原始论文](https://proceedings.mlr.press/r7/rendle09a.html)。这里借用其配对排序目标，不表示建立了症状条件概率的贝叶斯模型。

## 正负样本与困难负例

- 正疾病：训练病例中的 PATHOLOGY。
- 负疾病：排除 PATHOLOGY，也排除该病例完整鉴别诊断列表中所有概率大于 0 的疾病。
- 每个训练输入使用四个当前得分最高的允许候选和四个随机允许候选。随机候选允许重复。
- 少于四个允许候选时，只让有效位置参与 loss；没有允许候选时，该病例配对项为零。
- 这些是训练用对照候选，不是已确认临床排除的疾病。完整病例鉴别列表也未必覆盖只观察少量症状时的所有合理候选，所以仍存在假负例风险。
- hybrid 中 CE 仍是单一 PATHOLOGY 标签目标，鉴别列表保护只应用于 pairwise 项；不能宣称整个 hybrid 已解决多疾病合理性问题。

配对元数据只读取训练文件。脚本重建原训练抽样并核对每一行的症状输入及标签，确保正负候选与固定病例一一对应。原数据准备默认每疾病 cap 为 2000；改变该设置后应同步更新配对准备参数和元数据，不能复用旧 mask。

## 对照方式

两种新目标从相同随机初始权重训练，沿用原模型学习率、结构、症状遮蔽和 12 轮预算；原 CE baseline 也使用相同设置。固定 hybrid 权重 0.2，不按测试集搜索权重。每个版本按验证集 MRR 保存最佳轮次，并在观察本轮测试结果前按验证集决定推荐版本。

测试分别使用最多三个阳性症状，以及 0/5/10 个随机模拟阴性答案；固定三个输入种子。该测试集此前已经查看，因此是后续 benchmark 对照，不是新的盲测。最终项目报告还需要新的预留数据或独立评估；当前结果也受二元特征投影后的重复输入影响。

## 运行

```powershell
.venvforgc1/Scripts/python.exe disease_retrieval_pairwise.py prepare
.venvforgc1/Scripts/python.exe disease_retrieval_pairwise.py train --epochs 12
.venvforgc1/Scripts/python.exe disease_retrieval_tristate.py predict --model selected --present cough "shortness of breath" --absent fever
.venvforgc1/Scripts/python.exe -m unittest test_disease_retrieval_pairwise.py test_disease_retrieval_tristate.py
```

应用中可用 `DiseaseRetriever(model_variant='selected')` 加载验证集选出的模型；也可以指定 `pairwise` 或 `ce_pairwise`。默认仍为原三态 `cross_entropy`，不因新实验存在就自动更换。所有版本都接收有/无/未知三态输入，支持 49 种疾病；原 413 疾病模型保持独立。

实验权重、训练历史、负例元数据与验证集选择结果保存于 `artifacts/disease_retrieval_pairwise/`。

## 本轮结论

验证集 MRR：Cross-Entropy 为 0.9051，纯 pairwise 为 0.8720，CE + pairwise 为 0.9046。因此验证集推荐保留 Cross-Entropy。固定测试输入（最多三个阳性、五个随机阴性）的平均 Hits@1 分别为 83.07%、78.11%、83.05%。混合目标与原模型非常接近，本轮没有显示添加 pairwise 的收益；这不代表 pairwise 在所有数据、采样和目标设置下都较差。

配对版本保留用于对照与后续实验，不自动替换默认模型。八项输入和负例采样检查通过，包括避免受保护疾病进入有效负例项，以及排序错误时梯度推动正确疾病得分上升。
