# 773 疾病三态输入与少量模拟阴性训练 v2

本版本覆盖原数据集全部 773 个标签、377 个症状，输入从 377 维升级为 754 维：前半部分是明确存在，后半部分是明确否认。某症状两个位置都为 0 表示未知。旧阳性模型和交接包保留，不覆盖。

## 阴性数据的实验假设

2026-10-15，用户明确接受把原数据集 0 作为模拟阴性进行实验。公开[数据说明](https://data.mendeley.com/datasets/2cxccsxydc/1)说明数据为人工生成，但尚不足以核实每个 0 都代表真实患者明确否认。

因此，我们仅在模拟询问某个 0 状态时揭示一个阴性答案；其他未揭示位置保持未知。不能把实验结果称为“使用了真实否认标签的验证结果”。训练命令要求显式传入 synthetic_zero_assumption，配置和推理输出均保留该标记。

## 两种正负要分开

- 输入证据：阳性/阴性/未知，表示患者对症状的回答。
- 患者—疾病训练对：正确疾病为正对，其他疾病可能被选作训练用对照。阴性症状不等于负疾病样本。

### 阳性为主的输入构造

每轮每条原训练病例产生一个证据视图：

- 阳性症状随机保留 20%–100%；若全部遮蔽则恢复原阳性。
- 约 75% 的视图不输入阴性证据。
- 约 25% 的视图随机揭示 1–3 个原始为 0 的症状作为模拟阴性。
- 未揭示位置保持未知，绝不把原始为 1 的症状翻转为阴性。

### 少量配对辅助

比较两个版本，主监督都用正确的患者—疾病标签：

| 版本 | 训练目标 |
| --- | --- |
| ce | 原类别权重的全量 Cross-Entropy |
| ce_pairwise | Cross-Entropy + 0.05 × 配对排序辅助项 |

配对辅助每批只选择约 25% 的患者，每位选一个对照疾病，按 50% 随机、50% 当前高分候选混合。正确疾病与同一 KG 分组成员候选不进入 pairwise 的负例池。这里是保守的负例保护，不表示分组成员医学等价，也未解决全部假负例问题；CE 项仍为单标签监督。KG 在混合版本中用于负例保护，不提供症状出现概率或诊断硬排除规则。

## 训练与评估

从原 773 类阳性模型继续训练：复制阳性通道、后续层和疾病嵌入，新增阴性通道权重初始化为 0。这样初始化时，在没有阴性输入时可复现原来的评分，避免随机阴性权重对初始推理造成干扰。

每个版本继续训练 10 轮，学习率 0.0005，batch size 512。输入视图与病例顺序使用相同种子；配对采样使用独立随机数发生器，以免对照模型接收不同症状视图。

使用原来固定的组隔离切分，训练/验证/测试记录数为 151,899/18,774/18,974；类别覆盖为 773/628/617。没有重新混合或把遮蔽视图拆到其他集合。

按验证集同样三个阳性症状、0 与 2 个阴性答案的 macro MRR 平均值选择模型和轮次，然后测试三个阳性与 0/1/3/5 个模拟阴性答案，三个固定输入种子平均。五个阴性超出训练的单视图上限，用于额外鲁棒性观察。与原阳性模型使用相同阳性输入对照，再对同一个三态模型擦除阴性通道做消融。

156 类没有独立测试记录；108 类仅训练集覆盖。该基准测试此前已被查看，属于后续实验，不是新的盲测；继续训练和新输入的影响不能简单归因于单个因素。

## 接口

推荐结构：

```python
from disease_retrieval_773_tristate import DiseaseRetriever773Tristate

retriever = DiseaseRetriever773Tristate()
result = retriever.retrieve(
    symptom_states={
        "cough": "positive",
        "fever": "negative",
        "shortness of breath": "unknown",
    },
    topk=10,
)
```

也可使用三个列表：

```python
result = retriever.retrieve(
    present_symptoms=["cough"],
    absent_symptoms=["fever"],
    unknown_symptoms=["shortness of breath"],
    topk=10,
)
```

不能同时使用字典和列表两种格式。同一症状在多个不同状态中出现会报错，即使使用不同别名也会检查。未提供症状默认未知，不会填成阴性。显式未知状态不覆盖已知回答；如需更新问诊状态，上游应先更新证据快照，再完整传入新状态。

全未知输入返回 needs_known_evidence 和空候选，避免凭空给出排名。纯阴性输入允许调用，但训练视图保留至少一个阳性，尚未验证纯阴性场景的可靠性。大量阴性输入也超出当前少量阴性训练分布。

所有标准英文症状名来自 vocabulary.json；少量中文别名仅为演示，不是任意自由文本理解。score 是未校准匹配分数，阴性证据改变相对得分，不会硬性排除某种疾病。

## 运行

```powershell
.venvforgc1/Scripts/python.exe disease_retrieval_773_tristate.py train --negative-policy synthetic_zero_assumption --epochs 10
.venvforgc1/Scripts/python.exe disease_retrieval_773_tristate.py predict --present cough --absent fever --unknown "shortness of breath" --topk 10
.venvforgc1/Scripts/python.exe -m unittest test_symptom_evidence.py test_disease_retrieval_773_tristate.py
```

产物位于 artifacts/disease_retrieval_773_tristate_v2/。config.json 记录假设、授权、采样比例和损失设置；selection.json 记录验证集选择；metrics.json 记录两模型对照与阴性擦除实验；negative_support.json 是潜在模拟阴性状态数量，不能当作独立否认病例数量。
