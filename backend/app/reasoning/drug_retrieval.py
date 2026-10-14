import pickle
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_disease_drug_map(pkl_path=DATA_DIR / "disease_drug_map.pkl") -> Dict[str, List[str]]:
    """加载准备好的 disease→drugs 字典"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_contraindication_map(
    pkl_path=DATA_DIR / "disease_contraindicated_drugs.pkl",
) -> Dict[str, List[str]]:
    """[问题 11] 加载 disease→[禁忌药物] 字典（由 extract_step3_fixed.py 生成）"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def retrieve_drugs_for_diseases(
    final_candidates: List[dict],
    disease_drug_map: Dict[str, List[str]],
    contraindication_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Module 5：KG Drug Retrieval（文档 §9）
    输入:
        final_candidates     = [{"disease": str, "score": float}, ...]
        disease_drug_map     : disease→[drugs]（indication + off-label use）
        contraindication_map : 可选，disease→[禁忌药物]。
                               传入时启用安全过滤：患者可能患有任一最终候选疾病，
                               因此对【任一】候选疾病禁忌的药物都不推荐。
    输出（格式与旧版一致）:
        {"disease_a": ["drug_a", "drug_b"], ...}
    """
    unsafe = set()
    if contraindication_map:
        for item in final_candidates:
            unsafe.update(contraindication_map.get(item["disease"], []))

    result = {}
    for item in final_candidates:
        disease_name = item["disease"]
        # 查不到就返回空列表，不要报错
        drugs = disease_drug_map.get(disease_name, [])
        result[disease_name] = [d for d in drugs if d not in unsafe]
    return result


# ======== 单元测试（直接跑） ========
if __name__ == "__main__":
    dd_map = load_disease_drug_map()

    # 测试 1：直接拿 map 里真实存在的前 2 个 disease
    sample_diseases = list(dd_map.keys())[:2]
    print(f"测试 1：真实 disease = {sample_diseases}")
    candidates_1 = [
        {"disease": sample_diseases[0], "score": 0.90},
        {"disease": sample_diseases[1], "score": 0.75},
    ]
    out_1 = retrieve_drugs_for_diseases(candidates_1, dd_map)
    for d, drugs in out_1.items():
        print(f"  {d}: {drugs[:3]}{'...' if len(drugs) > 3 else ''}  (共{len(drugs)}个)")
    assert all(out_1[d] == dd_map[d] for d in sample_diseases)
    print("  ✓ 通过")

    # 测试 2：文档 §9 的格式 + 不存在的 disease（应返回空列表）
    print("\n测试 2：含不存在的 disease → 必须不报错，且返回 []")
    out_2 = retrieve_drugs_for_diseases([{"disease": "fake_disease_xyz_123", "score": 0.55}], dd_map)
    assert out_2["fake_disease_xyz_123"] == [], "测试 2 失败！"
    print("  ✓ 通过")

    # 测试 3：禁忌过滤（mock）——drug_x 对 B 禁忌，因此 A 的推荐里也要去掉 drug_x
    print("\n测试 3：禁忌过滤（mock 数据）")
    mock_drugs = {"A": ["drug_x", "drug_y"], "B": ["drug_z"]}
    mock_contra = {"B": ["drug_x"]}
    cands = [{"disease": "A", "score": 0.8}, {"disease": "B", "score": 0.6}]
    out_3 = retrieve_drugs_for_diseases(cands, mock_drugs, mock_contra)
    print(f"  {out_3}")
    assert out_3 == {"A": ["drug_y"], "B": ["drug_z"]}, out_3
    assert retrieve_drugs_for_diseases(cands, mock_drugs) == mock_drugs   # 不传则不过滤
    print("  ✓ 通过")

    # 测试 4：禁忌过滤（真实数据）——过滤后的药物都不在任一候选的禁忌表里
    print("\n测试 4：禁忌过滤（真实数据）")
    contra = load_contraindication_map()
    both = [d for d in dd_map if d in contra][:3]
    cands = [{"disease": d, "score": 1.0} for d in both]
    raw = retrieve_drugs_for_diseases(cands, dd_map)
    safe = retrieve_drugs_for_diseases(cands, dd_map, contra)
    unsafe = {x for d in both for x in contra[d]}
    for d in both:
        assert not set(safe[d]) & unsafe
        print(f"  {d}: {len(raw[d])} → {len(safe[d])} 个药物")
    print("  ✓ 通过")

    print("\n✅ 全部测试通过，Drug Retrieval 模块 OK")
