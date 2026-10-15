"""規則の癖の検証ツール（tools/rule_quirk_check.py）。Qwen は使わず偽の行で比較ロジックを固定する。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import rule_quirk_check as rq  # noqa: E402


def _row(doc, start, kind, rule, qwen, verdict="ok", surface="青葉", split="train"):
    return {"doc_id": f"{split}:{doc}", "split": split, "source": "out_ner", "kind": kind, "start": start, "end": start + len(surface),
            "norm": surface, "surface": surface, "rule": rule, "qwen": qwen, "confidence": "high", "verdict": verdict, "drop": None}


def _fake_categorize(kind, norm):
    table = {"青葉の会": "association", "若葉委員会": "department", "青葉電機": "manufacturing", "ミナト市": "foreign"}
    return table.get(norm, {"org": "service", "facility": "other", "location": "other"}[kind])


def test_label_rows_with_marks_rule_now_and_type_word():
    rows = [_row(0, 0, "org", "party", "association", surface="青葉の会"), _row(1, 0, "org", None, "it", surface="アオバ")]
    out = rq.label_rows_with(rows, _fake_categorize)
    assert out[0]["rule_now"] == "association" and out[1]["rule_now"] is None
    assert "type_word" in out[0] and rows[0].get("rule_now") is None            # 元の行は変えない


def test_compare_counts_agreement_newly_firing_and_no_longer_firing():
    rows = [
        _row(0, 0, "org", "party", "association", surface="青葉の会"),          # 旧 party → 新 association = Qwen と一致（✗ 解消）
        _row(1, 0, "org", "government", "department", surface="若葉委員会"),     # 旧 government → 新 department = 一致
        _row(2, 0, "org", "manufacturing", "manufacturing", surface="青葉電機"),  # 一致のまま
        _row(3, 0, "org", None, "it", surface="アオバ"),                         # 発火せず（Qwen のみ）
        _row(4, 0, "location", "city", "foreign", surface="ミナト市"),            # 旧 city → 新 foreign = 一致
        _row(5, 0, "org", "retail", "service", verdict="wrong", surface="みなと商事"),   # 監査 wrong → ✗ に数えない
        _row(6, 0, "org", None, "association", surface="汐見の会", split="valid"),     # 旧 None → 新 association: 新しく発火、Qwen と一致
        _row(7, 0, "org", None, "government", surface="若草の会", split="valid"),      # 新しく発火、Qwen と不一致
    ]
    baseline = {(r["doc_id"], r["start"], r["end"]): {"rule": r["rule"], "type_word": "x"} for r in rows}
    def new_cat(kind, norm):
        if norm.endswith("の会"): return "association"
        if norm.endswith("委員会"): return "department"
        if norm == "青葉電機": return "manufacturing"
        if norm == "ミナト市": return "foreign"
        if norm == "みなと商事": return "service"
        return {"org": "service", "facility": "other", "location": "other"}[kind]
    now = rq.label_rows_with(rows, new_cat)
    cmp = rq.compare(now, baseline)
    a = cmp["agreement"]["org"]["all"]
    assert a["n_baseline"] == 4 and a["cross_baseline"] == 2            # 旧規則: party≠association, government≠department（wrong は数えない）
    assert a["n_now"] == 5 and a["cross_now"] == 1                       # 新規則: 新しく発火した 2 件も母数に入る。若草の会 は Qwen government ≠ association で ✗ 1
    nf = cmp["newly_firing"]["org"]
    assert nf["n"] == 2 and nf["mismatch"] == 1 and abs(nf["mismatch_rate"] - 0.5) < 1e-9
    assert cmp["no_longer_firing"]["org"]["n"] == 1 and cmp["no_longer_firing"]["org"]["qwen_dist"] == {"service": 1}   # みなと商事: 旧 retail → 新 service(OTHER)
    assert abs(cmp["other_rate"]["org"]["baseline"] - 3 / 7) < 1e-9 and abs(cmp["other_rate"]["org"]["now"] - 2 / 7) < 1e-9   # 旧: アオバ・汐見の会・若草の会 / 新: アオバ・みなと商事
    assert cmp["agreement"]["org"]["valid"]["n_now"] == 2
    report = rq.build_report(cmp)
    assert "青葉" not in report and "アオバ" not in report
    assert "の会" in report and "不一致" in report


def test_type_word_label_never_returns_the_whole_name():
    """最終修正波 B1: 部署・委員会の規則（department の `^[^株合有]{1,8}(部|…|委員会)$`、government の自治体接頭の
    委員会）は文字列先頭から一致するため、type_word_span の一致部分をそのまま type_word にすると組織名全体が
    report / baseline に出ていた。_type_word_label は種別語そのもの・"(whole)"・"(none)" だけを返す。"""
    # 一致範囲の末尾が部署・委員会の語 → その語だけ
    assert rq._type_word_label("org", "若葉委員会") == "委員会"            # department（名前全体に一致）
    assert rq._type_word_label("org", "青葉営業部") == "部"                # department
    assert rq._type_word_label("org", "朝霧市教育委員会") == "委員会"      # government の自治体接頭（名前全体に一致）
    assert rq._type_word_label("org", "若葉教育委員会") == "委員会"        # government の修飾語つき（名前の一部に一致）
    # 部署・委員会の語でなく、一致が名前全体 → "(whole)"
    assert rq._type_word_label("org", "電機") == "(whole)"
    assert rq._type_word_label("facility", "病院") == "(whole)"
    # それ以外 → 一致部分（種別語そのもの）
    assert rq._type_word_label("org", "青葉電機") == "電機"
    assert rq._type_word_label("org", " 青葉電機") == "電機"              # 位置は strip 後の文字列上
    assert rq._type_word_label("facility", "汐見クリニック") == "クリニック"
    # 種別語なし・org / facility 以外 → "(none)"
    assert rq._type_word_label("org", "アオバ") == "(none)"
    assert rq._type_word_label("location", "ミナト市") == "(none)"


def test_report_and_baseline_never_contain_department_or_committee_names(tmp_path):
    """最終修正波 B1: 部署・委員会の不一致行（規則 department、Qwen association）を含めても、report と dump
    （baseline）に組織名そのもの・その固有部分が出ないこと。"""
    rows = [
        _row(0, 0, "org", "department", "association", surface="若葉委員会"),            # ✗（規則 ≠ Qwen、監査 ok）
        _row(1, 0, "org", None, "association", surface="朝霧営業部"),                     # 新しく発火し Qwen と不一致
        _row(2, 0, "org", None, "government", surface="ハルカゼ市教育委員会", split="valid"),  # 新しく発火し Qwen と一致
    ]
    baseline = {(r["doc_id"], r["start"], r["end"]): {"rule": r["rule"], "type_word": "x"} for r in rows}

    def cat(kind, norm):
        return {"若葉委員会": "department", "朝霧営業部": "department", "ハルカゼ市教育委員会": "government"}[norm]

    now = rq.label_rows_with(rows, cat)
    assert [r["type_word"] for r in now] == ["委員会", "部", "委員会"]
    report = rq.build_report(rq.compare(now, baseline))
    assert "department/委員会" in report and "department/部" in report     # ✗・新規発火の内訳は種別語で出る
    path = tmp_path / "baseline.json"
    rq.dump(now, str(path))
    dumped = path.read_text(encoding="utf-8")
    for leaked in ("若葉", "朝霧", "営業", "ハルカゼ", "若葉委員会", "朝霧営業部", "ハルカゼ市教育委員会"):
        assert leaked not in report, leaked
        assert leaked not in dumped, leaked
