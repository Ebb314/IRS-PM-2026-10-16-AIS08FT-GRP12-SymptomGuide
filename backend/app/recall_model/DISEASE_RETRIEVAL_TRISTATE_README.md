# 三态疾病召回：DDXPlus 版本

本版本独立保存在 `artifacts/disease_retrieval_tristate/`，不覆盖原来的 413 疾病模型。它在 DDXPlus 的 49 种疾病范围内，支持已知存在、明确不存在、未知三种状态。

## 数据依据与边界

原 Disease-Symptom Dataset 的 0 无法从已有文件确认是否代表患者明确否认，因此不用于训练阴性通道。DDXPlus 是合成数据，EVIDENCES 保存合成患者存在的证据；二元证据默认值为 0。我们据此定义模拟回答：选中某症状进行询问时，代码存在回答“有”，默认状态回答“没有”，尚未揭示则为未知。这是合成问答假设，不是已收集的真实患者否认记录。

资料：[官方数据说明](https://github.com/mila-iqia/ddxplus)、[原始论文](https://papers.nips.cc/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)。论文同时指出生成的 differential diagnosis 不保证考虑了所有相关阴性证据。本版本只使用 PATHOLOGY 标签，不把 differential diagnosis 中的概率用于训练。

选择 95 个根级二元症状，不使用病史、类别、多选证据或年龄性别。E_150 的问题是在问能否排便，与一般症状有无的极性不同，先排除，避免错误解释。对其他代码，接口的“有/没有”始终以词表中的完整问题定义为准。少量中英文名称别名仅便于演示，不构成自由文本抽取。

## 输入及模型

```json
{
  "present_symptoms": ["cough", "shortness of breath"],
  "absent_symptoms": ["fever"]
}
```

Patient Tower 的输入为两个 95 维通道：存在、否认，共 190 维。某位置两个通道都为 0 就是未知；同时存在与否认会报错。不存在于词表的名称会报错，不会静默忽略。仅阴性输入可以运行，但实验主要验证有阳性症状的场景。

Patient Tower：190 → 128 → 64；Disease Tower：49 个可训练的 64 维疾病向量。点积打分、Cross-Entropy 训练。训练每例随机揭示最多 1–5 个存在症状、0–10 个默认阴性状态，其他全部设为未知。没有把所有未提及症状自动输入为否认。

输出 score 是未校准 logit，非患病概率；否认会调整疾病得分，不采用单症状硬排除规则。KG 当前仅附带名称精确匹配的节点候选，没有加入训练或排序。

## 切分和实验

- 沿用官方 train/validate/test 文件，不重新混合切分。
- 为快速 baseline，从每个文件按固定随机优先级抽样，每种疾病 train 最多 2,000 条，val/test 最多 333 条。这个每类上限改变了疾病分布，不代表自然人群患病率。
- 在抽样范围内去除相同完整 EVIDENCES 与疾病标签的重复记录；验证/测试中与此前集合完整 EVIDENCES 相同的记录也删除。
- 清洗后允许验证或测试集合缺失疾病，并在 audit.json 列明；每个训练疾病仍必须有样本。当前喉痉挛（原数据标签 Larygospasm）的测试完整证据全部与此前集合重复，无法得到该类的独立测试结果。宏平均仅涵盖实际测试中出现的疾病。
- 将完整证据投影到 95 个二元症状后，仍可能出现同样输入，这些不同完整证据的病例保留，投影重合数量见 audit.json。
- 不同完整证据不等于已确认独立患者；实验仅评价此合成数据设置。
- 同时训练阳性输入对照模型，使用相同患者、随机种子、结构宽度、训练轮次；输入维数分别为 95 与 190。
- 以验证集最多 3 个阳性、5 个阴性输入的 MRR 选择三态权重；阳性对照只看到相同的阳性部分。
- 测试保持阳性输入相同，分别加入 0/1/3/5/10 个随机阴性答案，三个固定输入种子取平均。随机询问可能包含弱相关症状，结果不等价于自适应问诊。
- 再对同一个三态模型进行消融：保留或擦除同样的 5 个阴性答案，测量指标和 Top-1 改变比例。这可以区分重新训练的影响与实际使用阴性输入的影响。

## 运行与对接

```powershell
.venvforgc1/Scripts/python.exe disease_retrieval_tristate.py prepare
.venvforgc1/Scripts/python.exe disease_retrieval_tristate.py train --epochs 12
.venvforgc1/Scripts/python.exe disease_retrieval_tristate.py predict --present cough "shortness of breath" --absent fever --topk 5
.venvforgc1/Scripts/python.exe -m unittest test_disease_retrieval_tristate.py
```

也可以在应用中加载一次 `DiseaseRetriever`，反复调用 `retrieve(present_symptoms=[...], absent_symptoms=[...], topk=5)`。上游抽取模块需要分开提取明确存在和明确否认，含糊或未提及的信息不应填入任何通道。

代码支持 E_91 等标准证据 ID，或 vocabulary.json 中的完整英文问题；演示别名支持发热、咳嗽、呼吸困难、咽痛、腹泻及对应英文名称。疾病 ID 是该版本本地编号，需与词表和权重一起版本化。重新 prepare 后须重新训练。

产物：vocabulary.json、dataset.npz、audit.json、positive_only.pt、tristate.pt、metrics.json。原模型的结果和当前结果不能直接比较，因为疾病范围、特征和数据来源均不同。

## 当前验证结论

五项检查通过，确认未知与否认分别编码、跨别名矛盾被拒绝、无效输入被拒绝、模拟回答不翻转病例状态，以及加载后的模型会使用否认信息。

随机加入阴性答案未提高总体召回。同模型固定输入种子 101 的消融中，最多三个阳性症状且没有阴性输入时 Hits@1 为 83.23%，再加入五个阴性答案后为 82.88%；4.26% 的病例 Top-1 发生改变。因此当前版本证明了三态输入与模型对接能力，尚未证明阴性证据带来的准确率收益。下一步应在验证集分析错误、检查问询分布与训练方式，并用新预先固定的实验评价改进，避免根据本轮测试集结果反复调参。
