# 773 疾病三态模型 v2 交接

这个版本接收全部 377 个症状的阳性、阴性、未知状态，754 维输入，召回 773 个疾病标签。此前阳性 v1 和 DDXPlus 49 类三态模型独立保留，不能混用权重。

**关键实验政策：用户已于 2026-10-15 接受将原始 0 作为模拟阴性进行研究训练。训练阴性不是经过核验的真实否认记录。** 具体边界见 SYNTHETIC_NEGATIVE_POLICY.md。

## 运行

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo.py
python check_bundle.py
```

服务启动时加载一次模型：

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

也可改用 present_symptoms、absent_symptoms、unknown_symptoms 三个列表，但不能同时传两种格式。省略即未知；冲突报错；全未知返回 needs_known_evidence 和空候选。上游需要提取并标准化患者描述，不能直接传自由文本。

阴性输入改变匹配分数，不硬性排除疾病。score 不是患病概率。纯阴性以及大量阴性输入虽可表示，但超出了本轮主要训练分布，尚不能宣称这些场景可靠。

## 文件

- disease_retrieval_773_tristate.py：当前接口，默认读取 selection.json 选出的版本。
- symptom_evidence.py：三态编码、别名和冲突校验。
- disease_retrieval.py、disease_retrieval_773.py：模型与评分/评估依赖；旧接口并非本包推荐入口。
- artifacts/disease_retrieval_773_tristate_v2/：两版最佳权重、词表、类别支持量、配置、验证集选择、指标和阴性政策审计。
- data/unified_catalog/：此前建立的两个数据集与共享 KG 实体链接目录。
- requirements.txt：依赖；manifest.json：冻结文件哈希与范围。

保留相对路径。本包不含原始病例、训练矩阵或旧阳性初始化权重，只用于推理和结果检查；不要直接运行训练命令覆盖冻结产物。

## 训练方案

从全量阳性 773 模型继续训练，每版十轮。约 75% 视图只用阳性，25% 视图加 1–3 个模拟阴性。比较 CE 与 CE + 0.05 pairwise；辅助项只为约 25% 视图选一个对照疾病。主监督仍是正确的患者—疾病标签。

模型选择基于验证集；测试基准此前已查看，属于后续实验。训练覆盖 773 类，测试覆盖 617 类，156 类无独立测试记录。不要把可输出 773 类表述为已验证所有类别。

## 与共享 KG 对接

KG 原文件没有修改。kg_links 标明 direct_mondo、exact_name_candidate、group_membership 等依据，分组成员不等于精确概念等价。同一节点不同依据不能重复累计。未匹配与待复核项目仍保留。

相关目录的覆盖是 425/773 个疾病具有候选链接、309/377 个症状具有 HPO/KG 候选，不代表全部医学映射已经核验。混合训练中 KG 分组候选仅用于保护 pairwise 负例，不能被解释成症状概率或硬性诊断规则。

详细方法与结果分别见 DISEASE_RETRIEVAL_773_TRISTATE_README.md、DISEASE_RETRIEVAL_773_TRISTATE_RESULTS.md。
