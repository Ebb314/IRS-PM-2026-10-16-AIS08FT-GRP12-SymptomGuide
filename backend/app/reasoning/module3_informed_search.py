"""
Module 3: Informed Search（文档 §7）
功能：根据当前候选疾病 + patient_state，挑选下一轮最值得问的症状

对外主函数:
    select_best_next_symptom(candidate_diseases, patient_state, disease_symptom_map,
                             allowed_symptoms=None)
    -> {"symptom": str | None, "heuristic_score": float}

本版改动（对应代码审查编号）：
    [1] 新增 allowed_symptoms：只从召回模型认识的症状里挑问题，
        否则用户对该症状的回答无法作为特征输入召回模型。
    [5] 打平时按固定规则选择（h 降序 → 症状名升序），结果可复现，
        不再受 Python 字符串哈希随机化影响。
    [6] 单元测试按新公式重写，全部可稳定通过。
    [7] 候选疾病 < 2 个，或所有症状 h = 0（问了也区分不开）时返回 symptom=None，
        交给收敛判断结束问诊，不再问无意义的问题。
    [8] 启发函数按候选疾病分数加权（见 _compute_heuristics 注释）。
    [9] 修正 uncertain 相关注释。
    （[4] 非症状词条的过滤在 build_kg_maps.py 里构建查找表时完成。）
"""
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
EPS = 1e-12


# ============================================================
# 数据加载
# ============================================================
def load_disease_symptom_map(path=DATA_DIR / "disease_symptom_map.pkl") -> Dict[str, List[str]]:
    """读取 build_kg_maps.py 生成的 {disease: [symptoms]}（支持 .pkl / .json）"""
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_allowed_symptoms(csv_path, column: str = "matched_name") -> Set[str]:
    """
    [问题 1] 读取召回模型的症状词表，返回允许提问的 KG 症状名集合。

    默认读取召回组的 symptom_hpo_matched.csv（06_disease_mapping.ipynb 使用的文件），
    其中 matched_name 列就是数据集症状映射到的 KG / HPO 表型名。
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return {row[column].strip() for row in csv.DictReader(f) if row.get(column)}


# ============================================================
# 【Mock 数据区】— 单元测试用
# ============================================================
MOCK_DISEASE_SYMPTOM_MAP: Dict[str, List[str]] = {
    "disease_a": ["chest pain", "night sweats"],
    "disease_b": ["chest pain", "fatigue"],
    "disease_c": ["night sweats", "fatigue"],
    "disease_d": ["fatigue"],
}
MOCK_CANDIDATE_DISEASES = [
    {"disease": "disease_a", "score": 0.82},
    {"disease": "disease_b", "score": 0.67},
    {"disease": "disease_c", "score": 0.53},
    {"disease": "disease_d", "score": 0.48},
]
MOCK_PATIENT_STATE_INIT = {
    "positive": [], "negative": [], "uncertain": [], "asked": [],
}


# ============================================================
# Step 1: KG Retrieval — 候选症状收集 + 排除已问/已确认
# ============================================================
def _retrieve_candidate_symptoms(
    candidate_diseases: List[dict],
    patient_state: dict,
    disease_symptom_map: Dict[str, List[str]],
    allowed_symptoms: Optional[Iterable[str]] = None,
) -> List[str]:
    """返回「可能值得问」的症状列表（已去重并排序，保证结果可复现）"""
    # a) 收集所有候选疾病关联症状
    symptom_pool: set = set()
    for item in candidate_diseases:
        d = item["disease"]
        if d not in disease_symptom_map:
            print(f"  [Informed Search WARN] disease '{d}' 不在 disease_symptom_map 里，跳过")
            continue
        symptom_pool.update(disease_symptom_map[d])

    # b) 排除 asked ∪ positive ∪ negative（文档 §7 Step 1 规则）
    #    注意：回答 "Not sure" 的症状会同时记入 uncertain 和 asked，
    #    因此也会在这里被排除，不会重复提问。
    exclude = (
        set(patient_state.get("asked", []))
        | set(patient_state.get("positive", []))
        | set(patient_state.get("negative", []))
    )
    candidates = symptom_pool - exclude

    # c) [问题 1] 只保留召回模型认识的症状（不区分大小写）
    if allowed_symptoms is not None:
        allowed_lower = {s.lower() for s in allowed_symptoms}
        candidates = {s for s in candidates if s.lower() in allowed_lower}

    return sorted(candidates)


# ============================================================
# Step 2: Heuristic Evaluation — 按分数加权的"平衡切分"
# ============================================================
def _normalized_weights(candidate_diseases: List[dict]) -> Dict[str, float]:
    """把候选疾病分数归一化成概率 p(d)；分数缺失或全为 0 时退化为均匀分布。"""
    raw = {item["disease"]: max(float(item.get("score", 0.0) or 0.0), 0.0)
           for item in candidate_diseases}
    total = sum(raw.values())
    if total <= EPS:
        n = len(raw)
        return {d: 1.0 / n for d in raw} if n else {}
    return {d: w / total for d, w in raw.items()}


def _compute_heuristics(
    candidate_symptoms: List[str],
    candidate_diseases: List[dict],
    disease_symptom_map: Dict[str, List[str]],
) -> List[Tuple[str, float]]:
    """
    对每个症状算 h(s)。[问题 8] 按候选疾病分数加权：

        p(d)     = score(d) / Σ score          （归一化后的疾病概率）
        P_yes(s) = Σ p(d)，d 为关联症状 s 的候选疾病
        P_no(s)  = 1 - P_yes(s)
        h(s)     = min(P_yes(s), P_no(s))     取值范围 [0, 0.5]

    含义：一个好问题应当把"概率质量"尽量对半切开。
      - 用户答 No 时，关联 s 的疾病会被直接删除（见 state_update.py），
        期望删除的概率质量为 P_yes · P_no，与 h(s) 同调递增；
      - min(P, 1-P) 也与回答的二元熵（信息增益）同调递增，
        所以选 h 最大的症状 = 选信息增益最大的症状。
    与旧版 min(yes 个数, no 个数) 相比，高分疾病的权重更大，
    系统会优先去区分"最可能的几个病"。
    """
    weights = _normalized_weights(candidate_diseases)
    dis_to_syms = {d: set(disease_symptom_map.get(d, [])) for d in weights}

    results = []
    for s in candidate_symptoms:
        p_yes = sum(w for d, w in weights.items() if s in dis_to_syms[d])
        h = min(p_yes, 1.0 - p_yes)
        results.append((s, max(h, 0.0)))
    return results


# ============================================================
# Step 3: Greedy Best-First Selection
# ============================================================
def _greedy_best_first(heuristics: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
    """
    选 h 最大的 symptom。
    [问题 5] 打平时按症状名字母序选，保证每次运行结果一致。
    [问题 7] 最大 h 为 0（问了也无法区分任何候选）时返回 (None, 0.0)。
    """
    if not heuristics:
        return None, 0.0
    best_sym, best_h = min(heuristics, key=lambda x: (-round(x[1], 12), x[0]))
    if best_h <= EPS:
        return None, 0.0
    return best_sym, best_h


# ============================================================
# 对外主接口
# ============================================================
def select_best_next_symptom(
    candidate_diseases: List[dict],
    patient_state: dict,
    disease_symptom_map: Dict[str, List[str]],
    verbose: bool = False,
    allowed_symptoms: Optional[Iterable[str]] = None,
) -> dict:
    """
    Informed Search 主函数（文档 §7）。

    Parameters
    ----------
    candidate_diseases : list[dict]
        Module 2 输出：[{"disease": str, "score": float}, ...]
    patient_state : dict
        {"positive": [], "negative": [], "uncertain": [], "asked": []}
    disease_symptom_map : dict
        {disease_name: [symptom1, symptom2, ...]}
    verbose : bool
        True 时打印每个症状的 h(s)，方便调试 / Demo
    allowed_symptoms : iterable[str] | None
        [问题 1] 允许提问的症状集合（召回模型的症状词表）。None 表示不限制。

    Returns
    -------
    {"symptom": str | None, "heuristic_score": float}
        symptom 为 None 表示没有值得再问的问题，应直接进入收敛/输出阶段。
    """
    # [问题 7] 少于 2 个候选时无需区分
    if len(candidate_diseases) < 2:
        if verbose:
            print("  [Informed Search] 候选疾病 < 2，无需继续提问")
        return {"symptom": None, "heuristic_score": 0.0}

    # 1) KG Retrieval
    candidate_symptoms = _retrieve_candidate_symptoms(
        candidate_diseases, patient_state, disease_symptom_map, allowed_symptoms
    )
    if verbose:
        print(f"  [Step1] 候选症状 ({len(candidate_symptoms)}): {candidate_symptoms[:20]}")

    # 2) Heuristic Evaluation
    heuristics = _compute_heuristics(candidate_symptoms, candidate_diseases, disease_symptom_map)
    if verbose:
        print("  [Step2] Heuristic table:")
        for s, h in sorted(heuristics, key=lambda x: (-x[1], x[0]))[:10]:
            print(f"          {s:<24}  h = {h:.3f}")

    # 3) Greedy Best-First
    best_sym, best_h = _greedy_best_first(heuristics)
    if verbose:
        print(f"  [Step3] Best: symptom={best_sym!r}, heuristic={best_h:.3f}")

    return {"symptom": best_sym, "heuristic_score": best_h}


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    # Mock 分数 0.82 / 0.67 / 0.53 / 0.48，总和 2.5
    # → p(A)=0.328, p(B)=0.268, p(C)=0.212, p(D)=0.192
    #   chest pain   : A+B   = 0.596 → h = min(0.596, 0.404) = 0.404
    #   night sweats : A+C   = 0.540 → h = min(0.540, 0.460) = 0.460
    #   fatigue      : B+C+D = 0.672 → h = min(0.672, 0.328) = 0.328
    print("=" * 60)
    print("【测试 1/6】加权 h 值与手算一致")
    hmap = dict(_compute_heuristics(
        ["chest pain", "night sweats", "fatigue"],
        MOCK_CANDIDATE_DISEASES, MOCK_DISEASE_SYMPTOM_MAP))
    print(f"   计算结果: { {k: round(v, 3) for k, v in hmap.items()} }")
    assert abs(hmap["chest pain"] - 0.404) < 1e-9
    assert abs(hmap["night sweats"] - 0.460) < 1e-9
    assert abs(hmap["fatigue"] - 0.328) < 1e-9
    print("  ✓ 通过\n")

    print("【测试 2/6】初始状态 → 选 night sweats（它把高分的 A 和 C 与 B、D 分开，切得最均匀）")
    r = select_best_next_symptom(MOCK_CANDIDATE_DISEASES, MOCK_PATIENT_STATE_INIT,
                                 MOCK_DISEASE_SYMPTOM_MAP, verbose=True)
    assert r["symptom"] == "night sweats", r
    print("  ✓ 通过\n")

    print("【测试 3/6】已问过 night sweats → 不再出现，下一个是 chest pain")
    state = {"positive": ["night sweats"], "negative": [], "uncertain": [],
             "asked": ["night sweats"]}
    r = select_best_next_symptom(MOCK_CANDIDATE_DISEASES, state, MOCK_DISEASE_SYMPTOM_MAP)
    assert r["symptom"] == "chest pain", r
    print(f"  → {r}\n  ✓ 通过\n")

    print("【测试 4/6】打平时结果确定：等分数下 chest pain 与 night sweats 打平 → 按字母序选 chest pain")
    equal = [{"disease": d, "score": 1.0} for d in MOCK_DISEASE_SYMPTOM_MAP]
    picks = {select_best_next_symptom(equal, MOCK_PATIENT_STATE_INIT,
                                      MOCK_DISEASE_SYMPTOM_MAP)["symptom"] for _ in range(20)}
    assert picks == {"chest pain"}, picks
    print("  ✓ 通过（跨进程的确定性由 PYTHONHASHSEED 测试另行验证）\n")

    print("【测试 5/6】只剩 1 个候选 / 所有症状都无区分度 → 返回 None")
    r1 = select_best_next_symptom([{"disease": "disease_d", "score": 0.9}],
                                  MOCK_PATIENT_STATE_INIT, MOCK_DISEASE_SYMPTOM_MAP)
    same = {"x": ["fever"], "y": ["fever"]}
    r2 = select_best_next_symptom([{"disease": "x", "score": 0.6}, {"disease": "y", "score": 0.4}],
                                  MOCK_PATIENT_STATE_INIT, same)
    assert r1["symptom"] is None and r2["symptom"] is None, (r1, r2)
    print("  ✓ 通过\n")

    print("【测试 6/6】allowed_symptoms 白名单：只允许问 fatigue")
    r = select_best_next_symptom(MOCK_CANDIDATE_DISEASES, MOCK_PATIENT_STATE_INIT,
                                 MOCK_DISEASE_SYMPTOM_MAP, allowed_symptoms={"Fatigue"})
    assert r["symptom"] == "fatigue", r
    print("  ✓ 通过\n")

    print("=" * 60)
    print("🎉 所有单元测试通过！")
