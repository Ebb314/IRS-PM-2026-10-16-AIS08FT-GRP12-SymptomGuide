# KG 推理集成记录（REASONING_MODE=kg）

## 做了什么

把 GroupWork 仓库里的推理模块接进对话系统，替换原来的随机测试适配器。前端请求协议、会话状态和停止策略都没有改变，新模式只是实现了 `services/interfaces.py` 中的三个抽象类：

| 接口 | 实现 | 来源 |
|---|---|---|
| `RecallService` | `KGRecallService` | 过渡版 KG 召回（召回模型训练好之后替换） |
| `InformedSearchService` | `KGSearchService` | GroupWork `module3_informed_search.py` |
| `DrugService` | `KGDrugService` | GroupWork `Drug_Retrival.py`（含禁忌过滤） |

数据流：

```
用户文本 → NLP（规则 / 本地 LLM）→ positive / negative / uncertain
        → KGRecallService：召回 + 硬过滤（答 No 删除疾病）+ 排序，取 Top-K
        → 停止判断（候选 ≤ 2 / 达到轮数上限 / 没有可问问题）
        → KGSearchService：加权平衡切分启发式，贪心最佳优先，选下一个症状
        → 停止后 KGDrugService：indication / off-label use，去掉禁忌药物
```

## 新增和修改的文件

新增：

- `backend/app/reasoning/`：从 GroupWork 原样复制的三个模块（`module3_informed_search.py`、`state_update.py`、`drug_retrieval.py`）。如需修改，请先改 GroupWork，再复制过来，保持两边一致。
- `backend/app/services/kg_reasoning_service.py`：三个适配器类，负责在对话系统的数据模型和 GroupWork 模块之间转换。
- `backend/app/data/kg_reasoning.json.gz`（约 1 MB）：打包后的查找表，包括疾病–症状、疾病–缺失症状、疾病–药物和禁忌症。
- `scripts/build_kg_reasoning_data.py`：从 GroupWork 输出重新生成上面的数据文件。
- `backend/tests/test_kg_reasoning.py`：9 个测试。

修改：

- `config.py`：新增 `KG_REASONING_FILE`、`KG_TOP_K`、`KG_DRUGS_PER_DISEASE`、`RECALL_SYMPTOM_WHITELIST`；`MAX_ROUNDS` 改为可配置（kg 模式默认 12，其他模式仍是 5）。
- `dependencies.py`：新增 `kg` 分支。
- `scripts/run.ps1`：未设置 `REASONING_MODE` 时默认使用 `kg`。
- 前端：`ChatView.vue` 中与模式相关的文字支持 kg；会话结束时自动展开结果面板；`DiseaseCard.vue` 用症状名称代替 `P835` 这类 ID 显示支持证据。

`config.py` 中 `REASONING_MODE` 的默认值仍是 `random`，所以已有的评估脚本和测试行为不变。

## 运行

```powershell
$env:REASONING_MODE='kg'
.\scripts\run.ps1 -Profile offline   # 或 -Profile local 使用本地 Qwen
```

`/api/health` 应显示 `"mode":"kg"`。

如果 GroupWork 的数据更新了，重新生成数据文件：

```powershell
python scripts/build_kg_reasoning_data.py --groupwork E:\path\to\GroupWork
```

## 推理行为

**召回（过渡版）**

- 候选疾病 = 能解释最多 positive 症状的疾病。
- 答 No 的症状：所有关联该症状的疾病直接删除。
- 用户确认有某症状，而 KG 记录某疾病"不具有"该症状：该疾病也删除。
- 如果最优一层的疾病全部被删除，退回到下一层。
- 排序分数 = 匹配症状的 IDF 之和 ÷ log2(2 + 该疾病的症状数)。越罕见的症状证据越强；同样的证据，对症状注释更少的疾病更有针对性。
- 分数在返回的 Top-K 内归一化，是排序分数，不是概率。

**提问**：沿用 GroupWork 的启发函数 h(s) = min(P_yes, 1 − P_yes)，按候选分数加权，打平时按名称排序。只会问对话词表里存在的症状。设置 `RECALL_SYMPTOM_WHITELIST` 后，只会问召回模型认识的症状。前端显示的提问理由会说明这个症状把候选分成了几份。

**药物**：每个疾病最多显示 `KG_DRUGS_PER_DISEASE` 个药物，适应症排在前面。对任一剩余候选疾病有禁忌的药物会被去掉，卡片备注里会写明去掉了几个。

## 模拟评估

用真实 `ChatbotService`（规则 NLP）跑模拟患者：开场说两个症状，之后按"真实疾病"在 KG 中的症状自动回答。

| 场景 | MAX_ROUNDS | 达到候选阈值 | 因轮数上限停止 | 真实疾病在最终列表中 | 平均提问数 |
|---|---|---|---|---|---|
| 开场为该疾病的 2 个随机症状（200 例） | 5 | 89% | 9% | 100% | 1.6 |
| 开场为常见症状组合，如 fever + cough（80 例） | 5 | 23% | 78% | 99% | 4.8 |
| 同上 | 8 | 94% | 6% | 100% | 5.4 |
| 同上 | 12 | 100% | 0% | 100% | 5.4（最多 9） |

所以 kg 模式默认 `MAX_ROUNDS=12`。只有达到候选阈值时才会调用药物接口，轮数上限太小，大部分会话就看不到药物关系。

**注意**：模拟患者的回答来自同一个 KG，评估是循环的。这些数字只说明流程能收敛、不会误删，不代表真实诊断准确率。

## 已知局限

1. **召回模型尚未接入。** 召回组的模型还没训练，当前召回完全基于 KG。模型训练好之后，新写一个实现 `RecallService` 的类，在 `dependencies.py` 中替换 `KGRecallService` 即可；输出依然要经过 `state_update.filter_candidates`。
2. **PrimeKG 以罕见病为主。** 常见症状组合（例如 fever + cough）召回的多是罕见病，演示时要说明。
3. **规则 NLP 识别范围有限。** 例如 "feel very tired" 不会被识别为 Fatigue，不过系统接下来通常会主动问到。使用 local profile 时 LLM 可以改善这一点。
4. **本次修改的验证范围。** 修改环境中没有 FastAPI 和前端依赖：后端非 HTTP 测试全部通过（172 passed），`test_api.py` 的 7 个 HTTP 测试和前端 `npm run build` 需要在本机再跑一次。
