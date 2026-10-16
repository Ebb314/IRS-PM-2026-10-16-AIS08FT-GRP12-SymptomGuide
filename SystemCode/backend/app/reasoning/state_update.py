"""
Patient State 更新 + 候选疾病硬过滤（文档 §8 Chatbot Questioning 之后的一步）

对外函数:
    update_patient_state(patient_state, symptom, answer) -> patient_state
    filter_candidates(candidate_diseases, patient_state, disease_symptom_map,
                      disease_lacks_map=None) -> candidate_diseases

规则（小组决定）：
    - Yes      → positive + asked
    - No       → negative + asked，并【直接删除】所有关联该症状的候选疾病
    - Not sure → uncertain + asked，不删除任何疾病
    - 额外（使用问题 11 新抽取的 LACKS_PHENOTYPE）：
      若用户确认有某症状，而 KG 明确记录某候选疾病"不具有"该症状，也删除该疾病。
      不需要时调用 filter_candidates 不传 disease_lacks_map 即可关闭。

用法：filter_candidates 只依赖 patient_state，是幂等的。
      每一轮 Module 2 重新召回/打分之后都应调用一次，
      防止已被排除的疾病在下一轮 Top-K 中又被召回。
"""
from typing import Dict, List, Optional

YES, NO, UNSURE = "yes", "no", "not sure"

_ANSWER_ALIASES = {
    YES: {"yes", "y", "true", "1", "有", "是"},
    NO: {"no", "n", "false", "0", "没有", "否", "无"},
    UNSURE: {"not sure", "unsure", "uncertain", "don't know", "dont know",
             "idk", "不确定", "不知道"},
}


def normalize_answer(answer: str) -> str:
    """把用户/LLM 给出的回答归一化为 'yes' / 'no' / 'not sure'。"""
    a = str(answer).strip().lower()
    for canon, aliases in _ANSWER_ALIASES.items():
        if a in aliases:
            return canon
    raise ValueError(f"无法识别的回答: {answer!r}（应为 Yes / No / Not sure）")


def empty_patient_state() -> dict:
    return {"positive": [], "negative": [], "uncertain": [], "asked": []}


def update_patient_state(patient_state: dict, symptom: str, answer: str) -> dict:
    """根据一次问答更新 patient_state（原地修改并返回）。"""
    ans = normalize_answer(answer)
    for key in ("positive", "negative", "uncertain", "asked"):
        patient_state.setdefault(key, [])
        # 同一症状若被改口，先从三个结论列表里移除旧结论
        if key != "asked" and symptom in patient_state[key]:
            patient_state[key].remove(symptom)

    target = {YES: "positive", NO: "negative", UNSURE: "uncertain"}[ans]
    patient_state[target].append(symptom)
    if symptom not in patient_state["asked"]:
        patient_state["asked"].append(symptom)
    return patient_state


def filter_candidates(
    candidate_diseases: List[dict],
    patient_state: dict,
    disease_symptom_map: Dict[str, List[str]],
    disease_lacks_map: Optional[Dict[str, List[str]]] = None,
    verbose: bool = False,
) -> List[dict]:
    """
    按 patient_state 硬过滤候选疾病，返回保留下来的列表（保持原顺序和原分数）。

    删除条件（满足任一即删）：
      1) 疾病关联了任一 negative 症状（用户答 No）
      2) 传入 disease_lacks_map 时：疾病被 KG 标注为"不具有"任一 positive 症状
    返回空列表表示所有候选都被排除，调用方应提示"未找到匹配疾病，请就医"。
    """
    negative = set(patient_state.get("negative", []))
    positive = set(patient_state.get("positive", []))

    kept = []
    for item in candidate_diseases:
        d = item["disease"]
        hit_neg = negative & set(disease_symptom_map.get(d, []))
        hit_lack = positive & set(disease_lacks_map.get(d, [])) if disease_lacks_map else set()
        if hit_neg or hit_lack:
            if verbose:
                reason = (f"用户否认 {sorted(hit_neg)}" if hit_neg
                          else f"KG 记录该病不具有 {sorted(hit_lack)}")
                print(f"  [filter] 删除 {d}：{reason}")
            continue
        kept.append(item)
    return kept


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    DS = {
        "disease_a": ["chest pain", "night sweats"],
        "disease_b": ["chest pain", "fatigue"],
        "disease_c": ["night sweats", "fatigue"],
        "disease_d": ["fatigue"],
    }
    LACKS = {"disease_d": ["night sweats"]}
    CANDS = [{"disease": "disease_a", "score": 0.82}, {"disease": "disease_b", "score": 0.67},
             {"disease": "disease_c", "score": 0.53}, {"disease": "disease_d", "score": 0.48}]

    print("【测试 1/5】答 No → 删除关联该症状的疾病")
    st = update_patient_state(empty_patient_state(), "night sweats", "No")
    assert st == {"positive": [], "negative": ["night sweats"], "uncertain": [],
                  "asked": ["night sweats"]}, st
    out = filter_candidates(CANDS, st, DS, verbose=True)
    assert [c["disease"] for c in out] == ["disease_b", "disease_d"], out
    print("  ✓ 通过\n")

    print("【测试 2/5】答 Yes → 不删（未传 lacks 表）；传 lacks 表 → 删除 KG 标注'不具有'的 D")
    st = update_patient_state(empty_patient_state(), "night sweats", "yes")
    assert len(filter_candidates(CANDS, st, DS)) == 4
    out = filter_candidates(CANDS, st, DS, LACKS, verbose=True)
    assert [c["disease"] for c in out] == ["disease_a", "disease_b", "disease_c"], out
    print("  ✓ 通过\n")

    print("【测试 3/5】答 Not sure → 只记入 uncertain + asked，不删除")
    st = update_patient_state(empty_patient_state(), "fatigue", "不确定")
    assert st["uncertain"] == ["fatigue"] and st["asked"] == ["fatigue"]
    assert len(filter_candidates(CANDS, st, DS, LACKS)) == 4
    print("  ✓ 通过\n")

    print("【测试 4/5】改口：先 Not sure 再 No → 结论只保留 negative，asked 不重复")
    st = update_patient_state(empty_patient_state(), "fatigue", "not sure")
    st = update_patient_state(st, "fatigue", "no")
    assert st == {"positive": [], "negative": ["fatigue"], "uncertain": [],
                  "asked": ["fatigue"]}, st
    print("  ✓ 通过\n")

    print("【测试 5/5】幂等：重复过滤结果不变；全部排除时返回空列表")
    st = {"positive": [], "negative": ["chest pain", "fatigue"], "uncertain": [], "asked": []}
    once = filter_candidates(CANDS, st, DS)
    assert once == filter_candidates(once, st, DS) == []
    print("  ✓ 通过\n")
    print("🎉 state_update 全部测试通过！")
