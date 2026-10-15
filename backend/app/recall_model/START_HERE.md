# 疾病召回模块交接：三态 Cross-Entropy v1

这个包用于团队推理联调。模型支持 DDXPlus 的 49 种疾病和 95 个二元症状，保留已知存在、明确否认和未知三种状态。默认使用验证集选出的 Cross-Entropy 版本。

## 文件用途

| 文件 | 是否必须 | 用途 |
| --- | --- | --- |
| disease_retrieval_tristate.py | 是 | 输入编码与 DiseaseRetriever 推理接口 |
| disease_retrieval.py | 是 | 前一个文件导入的模型结构和评分函数 |
| artifacts/disease_retrieval_tristate/tristate.pt | 是 | 冻结模型权重 |
| artifacts/disease_retrieval_tristate/vocabulary.json | 是 | 特征和疾病顺序、完整症状问题、ICD-10、KG 节点候选 |
| requirements.txt | 环境准备 | 所需 Python 库版本 |
| demo.py | 建议 | 可以直接运行的联调示例 |
| test_disease_retrieval_tristate.py | 建议 | 输入契约和模型加载检查 |
| disease_kg_mapping.csv | 建议 | 与共享 KG 对接的疾病候选映射 |
| symptom_vocabulary.csv | 建议 | DDXPlus 症状 ID、完整问题及待补齐的 HPO 映射栏 |
| artifacts/disease_retrieval_tristate/audit.json、metrics.json | 评估参考 | 数据边界与实验结果 |
| DISEASE_RETRIEVAL_DESIGN_STORY.md | 报告参考 | 设计决策和答辩说明 |
| manifest.json | 版本核对 | 交接文件哈希、模型范围及参考 KG nodes.csv 哈希 |

需要保持上表中的相对目录结构；不要只复制权重，模型结构、词表与权重必须一起版本化。这个包没有训练数据，两个原始模块的 prepare/train 命令不属于交接包可运行功能。不要运行 prepare 或 train 覆盖冻结产物。

## 安装与运行

已测试环境为 Python 3.11.9，Windows，CPU 版 PyTorch。使用你们项目自己的虚拟环境，不需要复制原来的 .venvforgc1。

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo.py
python -m unittest test_disease_retrieval_tristate.py
```

直接从命令行推理：

```powershell
python disease_retrieval_tristate.py predict --model cross_entropy --present E_201 E_66 --absent E_91 --topk 5
```

这个包只包含 cross_entropy 权重，勿选择 pairwise、ce_pairwise 或 selected。配对实验记录见报告，但不需要带上实验权重才能协作。

## 推荐接口

```python
from disease_retrieval_tristate import DiseaseRetriever

retriever = DiseaseRetriever(model_variant="cross_entropy")  # 服务启动时只加载一次

result = retriever.retrieve(
    present_symptoms=["E_201", "E_66"],  # 咳嗽、显著呼吸困难
    absent_symptoms=["E_91"],           # 明确否认发热
    topk=5,
)
```

上游同学应优先传标准 evidence ID。支持完整英文问题与少量中英文演示别名，但不支持直接传自由文本。尚未询问、患者不确定、未提及的症状不放入任何列表。同一症状同时存在与否认会报错；词表外输入也会报错。

输出包含 disease_id（本词表的 DDX 编号）、disease_name、icd10、score 和 kg_candidate_node_indices。score 是未校准匹配分数，不是患病概率。DDX 编号不是 KG node_index，也不是医学统一编码。

## 共用 KG 时的对接

你们已有相同版本的 KG，无需重复传 nodes.csv、edges.csv。请使用 manifest.json 中记录的参考 nodes.csv 哈希核对节点索引版本。

当前疾病名称精确匹配为 **14 / 49**。disease_kg_mapping.csv 里为空的项表示“映射未完成”，不表示 KG 不存在对应知识；多个候选也不是经过核验的唯一映射。同学不能直接把疾病本地 DDX 编号用于查询 KG。

symptom_vocabulary.csv 使用 DDXPlus evidence ID，KG 使用 HPO 节点。原来的 symptom_hpo_matched.csv 是另一套 377 症状数据的映射，不能直接当作此 95 症状词表的完整映射。若下游需要在 KG 中选择症状问题，还需要补齐 HPO ↔ DDXPlus evidence ID 的映射；在完成前，可以先用标准 ID 联调疾病接口。

## 验收与边界

- 运行 demo，能得到五个候选疾病，且包含输入阳性、阴性 ID。
- 运行五项测试，全部通过。
- 上游按词表 ID 输入，下游按 kg_candidate_node_indices 查询同版本 KG；未映射项显式处理。
- 阴性回答来自 DDXPlus 合成患者默认状态，并非真实患者否认记录；目前阴性信息尚未提高总体召回。
- 模型输出 49 类，严格清洗后的测试覆盖 48 类。症状投影后的重复输入较多，离线指标不等于现实医疗准确率。
- 这个包不包含原 413 疾病模型，两个版本不能混用词表或权重。

参考：[DDXPlus 官方说明](https://github.com/mila-iqia/ddxplus)。
