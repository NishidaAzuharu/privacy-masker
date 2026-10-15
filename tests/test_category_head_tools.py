import argparse, hashlib, json, math, os, random, struct, sys
import pytest

# tools/train_category_head.py はトップレベルで numpy / torch / safetensors を import する
# （学習ツール専用。CI の minimal profile = `.[dev]` のみでは skip する）
pytest.importorskip("numpy")
pytest.importorskip("safetensors.numpy")
pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import category_head_data as chd  # noqa: E402
import train_category_head as tch  # noqa: E402


def test_form_label_mapping():
    assert chd.form_label("location", "city") == "city" and chd.form_label("location", "foreign_kata") == "foreign"
    assert chd.form_label("location", "compound") is None and chd.form_label("org", "dept") == "department"
    assert chd.form_label("org", "corporate") is None and chd.form_label("facility", None) is None
    assert chd.form_label("location", "station") is None                                     # station は FORM_LABELS に無い


def test_recover_forms_aligns_entities_with_slots_in_order():
    # NOTE: SLOT_RE (tools/synth_v11_plan.py) only recognizes the full type name (LOCATION, not LOC) as a
    # placeholder prefix, so slot/placeholder names below use "LOCATION1" to match the real template contract.
    tpl = {"template": "{ORG1}の{PERSON1}は{LOCATION1}へ。{EMAIL1}", "slots": [
        {"slot": "PERSON1", "type": "PERSON", "form": None}, {"slot": "ORG1", "type": "ORG", "form": "dept"},
        {"slot": "LOCATION1", "type": "LOCATION", "form": "city"}, {"slot": "EMAIL1", "type": "EMAIL", "form": None}]}
    doc = {"text": "総務部の山田太郎は札幌へ。a@b.c", "entities": [[0, 3, "ORG"], [4, 8, "PERSON"], [9, 11, "LOCATION"]], "role_substituted": 0}
    assert chd.recover_forms(doc, tpl) == ["dept", None, "city"]
    bad = dict(doc, entities=[[0, 3, "LOCATION"], [4, 8, "PERSON"], [9, 11, "LOCATION"]])   # 型が合わない → None
    assert chd.recover_forms(bad, tpl) is None
    assert chd.recover_forms(dict(doc, role_substituted=1), tpl) is None                     # 役職置換があると位置が合わない


def test_mask_type_word_removes_the_word_and_keeps_offsets_consistent():
    text = "青葉電機株式会社の総務部で汐見クリニックの件。"
    out = chd.mask_type_word(text, 0, 8, "org", "manufacturing")       # 種別語「電機」を削る
    assert out is not None
    new_text, s, e = out
    assert new_text[s:e] == "青葉株式会社" and new_text.startswith("青葉株式会社の")
    assert chd.mask_type_word(text, 13, 20, "facility", "medical") == ("青葉電機株式会社の総務部で汐見の件。", 13, 15)
    assert chd.mask_type_word("札幌市", 0, 3, "location", "city") == ("札幌", 0, 2)
    assert chd.mask_type_word("アオバ", 0, 3, "org", "service") is None                      # 種別語が無い


def test_mask_type_word_controller_refinement_rejects_leftover_type_word():
    # 「ネットワークス商船」: 表の順で transport（商船）が it（ネットワーク）より先に当たるため
    # categorize は transport。type_word_span も同じ「商船」を指すが、それを削った残り
    # 「ネットワークス」は規則で it と判定できてしまう → 隠しきれていないので None
    text = "ネットワークス商船"
    assert chd.RuleCategorizer.categorize("org", text) == "transport"
    assert chd.mask_type_word(text, 0, len(text), "org", "transport") is None


def test_mask_type_word_controller_refinement_rejects_leftover_type_word_location():
    # 「田村市」→末尾の「市」を削った残り「田村」は _LOC_CITY（末尾が市区町村なら city）に
    # 再度マッチしてしまう → 種別語（末尾の市区町村を示す語）を隠せていないので None。
    # 「札幌市」→「札幌」は市区町村で終わらないので city に戻らず、写しとして成立する。
    assert chd.RuleCategorizer.categorize("location", "田村") == "city"
    assert chd.mask_type_word("田村市", 0, 3, "location", "city") is None
    assert chd.mask_type_word("札幌市", 0, 3, "location", "city") == ("札幌", 0, 2)


def test_build_spans_labels_and_masking(tmp_path):
    docs = [
        {"text": "青葉電機に勤める。", "entities": [[0, 4, "ORG"]], "source": "out_ner"},
        {"text": "アオバに勤める。", "entities": [[0, 3, "ORG"]], "source": "out_ner"},             # 規則で決まらない → 落とす
        {"text": "総務部の件。", "entities": [[0, 3, "ORG"]], "source": "qwen38_synth_v11", "template_id": "t1", "fill_index": 0, "role_substituted": 0},
    ]
    templates = {"t1": {"template": "{ORG1}の件。", "slots": [{"slot": "ORG1", "type": "ORG", "form": "dept"}]}}
    rows = chd.build_spans(docs, templates, seed=1, mask_rate=1.0)
    labels = {(r["text"], r["label"], r["label_source"], r["masked"]) for r in rows}
    assert ("青葉電機に勤める。", "manufacturing", "rule", False) in labels
    assert ("青葉に勤める。", "manufacturing", "rule", True) in labels                          # mask_rate=1.0 で写しが必ず入る
    assert ("総務部の件。", "department", "rule", False) in labels                              # 規則が優先、form は一致
    assert not any(r["text"].startswith("アオバ") for r in rows)
    assert all(r["kind"] in ("org", "facility", "location") for r in rows)


def test_build_spans_rejects_a_label_outside_categories_with_value_error(monkeypatch):
    # assert ではなく ValueError（python -O でも検査が消えない）
    monkeypatch.setattr(chd, "form_label", lambda kind, form: "not_a_category")
    docs = [{"text": "アオバの件。", "entities": [[0, 3, "ORG"]], "source": "qwen38_synth_v11", "template_id": "t1",
             "fill_index": 0, "role_substituted": 0}]
    templates = {"t1": {"template": "{ORG1}の件。", "slots": [{"slot": "ORG1", "type": "ORG", "form": "dept"}]}}
    with pytest.raises(ValueError, match="invalid label"):
        chd.build_spans(docs, templates, seed=1, mask_rate=0.0)


def test_mask_type_word_v2_covers_prefecture_and_chome_but_not_hokkaido():
    assert chd.mask_type_word("大阪府", 0, 3, "location", "prefecture") == ("大阪", 0, 2)
    assert chd.mask_type_word("東京都", 0, 3, "location", "prefecture") == ("東京", 0, 2)
    assert chd.mask_type_word("北海道", 0, 3, "location", "prefecture") is None            # 「道」は削らない
    assert chd.mask_type_word("青葉三丁目", 0, 5, "location", "town") == ("青葉三", 0, 3)
    assert chd.mask_type_word("青葉町", 0, 3, "location", "town") is None                  # 「町」は市区町村と紛れるので削らない
    assert chd.mask_type_word("市", 0, 1, "location", "city") is None                      # 削った後が 2 字未満


def test_build_spans_from_labels_joins_text_sets_flags_and_splits(tmp_path):
    docs = {"train": [{"text": "青葉電機とアオバの件", "entities": [], "source": "out_ner"},
                      {"text": "汐見病院とヒカリノへ", "entities": [], "source": "gsd_train"}],
            "valid": [{"text": "札幌市の会議", "entities": [], "source": "concat"}]}
    labels = [
        {"doc_id": "train:0", "split": "head_train", "source": "out_ner", "kind": "org", "start": 0, "end": 4, "category": "manufacturing", "label_source": "rule", "confidence": None, "surface_sha": "x"},
        {"doc_id": "train:0", "split": "head_train", "source": "out_ner", "kind": "org", "start": 5, "end": 8, "category": "it", "label_source": "qwen", "confidence": "high", "surface_sha": "x"},
        {"doc_id": "train:1", "split": "head_dev", "source": "gsd_train", "kind": "facility", "start": 0, "end": 4, "category": "medical", "label_source": "rule", "confidence": None, "surface_sha": "x"},
        {"doc_id": "train:1", "split": "head_dev", "source": "gsd_train", "kind": "facility", "start": 5, "end": 9, "category": "medical", "label_source": "qwen", "confidence": "medium", "surface_sha": "x"},
        {"doc_id": "valid:0", "split": "head_judge", "source": "concat", "kind": "location", "start": 0, "end": 3, "category": "city", "label_source": "rule", "confidence": None, "surface_sha": "x"},
    ]
    out = chd.build_spans_from_labels(docs, labels, seed=1, mask_rate=1.0)
    assert set(out) == {"head_train", "head_dev", "head_judge"}
    tr = {(r["text"], r["label"], r["label_source"], r["masked"], r["no_type_word"]) for r in out["head_train"]}
    assert ("青葉電機とアオバの件", "manufacturing", "rule", False, False) in tr
    assert ("青葉とアオバの件", "manufacturing", "rule", True, True) in tr                  # mask_rate=1.0 で写しが必ず入る
    assert ("青葉電機とアオバの件", "it", "qwen", False, True) in tr                        # Qwen ラベル = 種別語なし
    assert not any(r["label_source"] == "qwen" and r["masked"] for r in out["head_train"])  # Qwen ラベルの写しは作らない
    assert {r["kind"] for r in out["head_dev"]} == {"facility"} and all(r["split"] == "head_dev" for r in out["head_dev"])
    judge = out["head_judge"]
    assert [(r["text"], r["label"], r["masked"]) for r in judge] == [("札幌市の会議", "city", False), ("札幌の会議", "city", True)]
    assert all(r["doc_id"].startswith(("train:", "valid:")) for rows in out.values() for r in rows)


def test_build_spans_from_labels_rejects_bad_label_or_split():
    docs = {"train": [{"text": "青葉電機", "entities": [], "source": "out_ner"}], "valid": []}
    base = {"doc_id": "train:0", "split": "head_train", "source": "out_ner", "kind": "org", "start": 0, "end": 4, "label_source": "rule", "confidence": None, "surface_sha": "x"}
    with pytest.raises(ValueError, match="invalid label"):
        chd.build_spans_from_labels(docs, [dict(base, category="nope")])
    with pytest.raises(ValueError, match="split"):
        chd.build_spans_from_labels(docs, [dict(base, category="manufacturing", split="valid")])


def _write_labels_manifest(labels, **over):
    """`tools/category_labels.py filter` が --labels の隣に書く manifest の最小形（門の結果と labels の SHA）。"""
    m = {"agreement_ok": True, "files": {"labels": {"name": labels.name, "sha256": hashlib.sha256(labels.read_bytes()).hexdigest()}}}
    m.update(over)
    (labels.parent / chd.LABELS_MANIFEST).write_text(json.dumps(m), encoding="utf-8")


def _labels_cli_env(tmp_path):
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps({"text": "青葉電機とアオバの件", "entities": [], "source": "out_ner"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text(json.dumps({"text": "札幌市の会議", "entities": [], "source": "concat"}, ensure_ascii=False) + "\n", encoding="utf-8")
    labels = tmp_path / "category_labels_v1.jsonl"
    labels.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"doc_id": "train:0", "split": "head_train", "source": "out_ner", "kind": "org", "start": 0, "end": 4, "category": "manufacturing", "label_source": "rule", "confidence": None, "surface_sha": "x"},
        {"doc_id": "train:0", "split": "head_dev", "source": "out_ner", "kind": "org", "start": 5, "end": 8, "category": "it", "label_source": "qwen", "confidence": "high", "surface_sha": "x"},
        {"doc_id": "valid:0", "split": "head_judge", "source": "concat", "kind": "location", "start": 0, "end": 3, "category": "city", "label_source": "rule", "confidence": None, "surface_sha": "x"},
    ]), encoding="utf-8")
    return data, labels, tmp_path / "spans"


def test_cli_labels_mode_writes_three_span_files_and_manifest(tmp_path):
    data, labels, out = _labels_cli_env(tmp_path)
    _write_labels_manifest(labels)
    assert chd.main(["--data", str(data), "--labels", str(labels), "--out", str(out), "--mask-rate", "0"]) == 0
    assert {p.name for p in out.iterdir()} == {"spans_train.jsonl", "spans_dev.jsonl", "spans_judge.jsonl", "manifest.json"}
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["inputs"]["labels"]["sha256"] and m["files"]["train"]["rows"] == 1 and m["files"]["dev"]["rows"] == 1 and m["files"]["judge"]["rows"] == 1
    assert m["counts"]["train"] and m["mode"] == "labels"


def test_labels_manifest_name_matches_the_filter_tool():
    import category_labels as cl
    assert chd.LABELS_MANIFEST == cl.LABELS_MANIFEST


@pytest.mark.parametrize("case, message", [
    ("missing", "labels manifest not found next to --labels"),
    ("gate_failed", "labels manifest reports agreement_ok=false; refusing to build spans"),
    ("gate_absent", "does not report agreement_ok=true"),
    ("sha_mismatch", "does not match files.labels.sha256"),
    ("sha_absent", "does not match files.labels.sha256"),
    ("not_json", "not valid JSON"),
])
def test_cli_labels_mode_refuses_without_a_passing_matching_manifest(tmp_path, case, message):
    # 門（tools/category_labels.py filter の agreement_ok）を下流で強制する: manifest が無い・不合格・labels の SHA が
    # 違う（別の実行の manifest・書き換えたラベル）なら、スパンを 1 行も書かずに止まる
    data, labels, out = _labels_cli_env(tmp_path)
    if case == "gate_failed":
        _write_labels_manifest(labels, agreement_ok=False)
    elif case == "gate_absent":
        _write_labels_manifest(labels, agreement_ok=None)
    elif case == "sha_mismatch":
        _write_labels_manifest(labels, files={"labels": {"name": labels.name, "sha256": "0" * 64}})
    elif case == "sha_absent":
        _write_labels_manifest(labels, files={"report": "filter_report.md"})
    elif case == "not_json":
        (labels.parent / chd.LABELS_MANIFEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        chd.main(["--data", str(data), "--labels", str(labels), "--out", str(out), "--mask-rate", "0"])
    assert message in str(ei.value)
    assert not out.exists()


def test_cli_labels_mode_refuses_labels_edited_after_filter(tmp_path):
    # manifest は合格・SHA も書いた時点では一致 → ラベルを後から書き換えると拒否する
    data, labels, out = _labels_cli_env(tmp_path)
    _write_labels_manifest(labels)
    labels.write_text(labels.read_text(encoding="utf-8").replace('"category": "it"', '"category": "finance"'), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match files.labels.sha256"):
        chd.main(["--data", str(data), "--labels", str(labels), "--out", str(out), "--mask-rate", "0"])
    assert not out.exists()


def test_split_judge_is_deterministic_stratified_and_moves_dev_into_train():
    def row(doc, source, kind="org", label="it", masked=False):
        return {"doc_id": doc + ("#masked" if masked else ""), "source": source, "kind": kind, "label": label, "masked": masked,
                "text": "x", "start": 0, "end": 1, "label_source": "rule", "no_type_word": masked, "split": "head_judge"}
    judge = [row(f"valid:{i}", "out_ner") for i in range(10)] + [row(f"valid:{i}", "gsd_train") for i in range(10, 14)]
    judge += [row("valid:0", "out_ner", masked=True)]                     # 写しは基底文書と同じ側へ
    by_split = {"head_train": [dict(row("train:0", "out_ner"), split="head_train")],
                "head_dev": [dict(row("train:1", "out_ner"), split="head_dev")], "head_judge": judge}
    out, m = chd.split_judge(by_split, seed=42)
    out2, _ = chd.split_judge(by_split, seed=42)
    assert out == out2
    assert len(out["head_train"]) == 2 and {r["doc_id"] for r in out["head_train"]} == {"train:0", "train:1"}
    a_docs = {r["doc_id"].split("#")[0] for r in out["head_dev"]}; b_docs = {r["doc_id"].split("#")[0] for r in out["head_judge"]}
    assert a_docs.isdisjoint(b_docs) and a_docs | b_docs == {f"valid:{i}" for i in range(14)}
    assert m["docs"] == {"a": 7, "b": 7} and m["by_source"]["gsd_train"] == {"a": 2, "b": 2}
    side = "head_dev" if "valid:0" in a_docs else "head_judge"
    assert any(r["masked"] for r in out[side]) and not any(r["masked"] for r in out["head_judge" if side == "head_dev" else "head_dev"])
    assert m["table"]["org/it"] == {"a": 7, "b": 7}


def test_cli_judge_split_writes_a_b_and_manifest(tmp_path):
    data = tmp_path / "data_v11"; data.mkdir()
    (data / "train_v11.jsonl").write_text(json.dumps({"text": "青葉電機とアオバの件", "entities": [], "source": "out_ner"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "valid_v11.jsonl").write_text("".join(json.dumps({"text": f"札幌市の会議{i}", "entities": [], "source": "gsd_train"}, ensure_ascii=False) + "\n" for i in range(4)), encoding="utf-8")
    labels = tmp_path / "category_labels_v1.jsonl"
    rows = [{"doc_id": "train:0", "split": "head_dev", "source": "out_ner", "kind": "org", "start": 0, "end": 4, "category": "manufacturing", "label_source": "rule", "confidence": None, "surface_sha": "x"}]
    rows += [{"doc_id": f"valid:{i}", "split": "head_judge", "source": "gsd_train", "kind": "location", "start": 0, "end": 3, "category": "city", "label_source": "rule", "confidence": None, "surface_sha": "x"} for i in range(4)]
    labels.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    _write_labels_manifest(labels)                                        # Task 10 の修正波で足したフィクスチャ（agreement_ok true + SHA）
    out = tmp_path / "spans"
    assert chd.main(["--data", str(data), "--labels", str(labels), "--out", str(out), "--mask-rate", "0", "--judge-split"]) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["mode"] == "labels+judge-split" and m["judge_split"]["docs"] == {"a": 2, "b": 2}
    assert m["files"]["train"]["rows"] == 1 and m["files"]["dev"]["rows"] == 2 and m["files"]["judge"]["rows"] == 2


def test_cli_judge_split_requires_labels(tmp_path, capsys):
    data = tmp_path / "data_v11"; data.mkdir()
    with pytest.raises(SystemExit):
        chd.main(["--data", str(data), "--templates", "x", "--out", str(tmp_path / "spans"), "--judge-split"])
    assert "--judge-split requires --labels" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# tools/train_category_head.py（Task 3）
# ---------------------------------------------------------------------------

def _clusters(n=60, dim=8, k=3, seed=0):
    rng = random.Random(seed); X, y = [], []
    for c in range(k):
        center = [1.0 if i == c else 0.0 for i in range(dim)]
        for _ in range(n):
            v = [x + rng.gauss(0, 0.15) for x in center]; norm = math.sqrt(sum(t * t for t in v))
            X.append([t / norm for t in v]); y.append(f"c{c}")
    return X, y


def test_train_and_evaluate_linear_head():
    X, y = _clusters(); cats = ["c0", "c1", "c2"]
    heads = tch.train_heads({"org": X}, {"org": y}, {"org": cats}, epochs=200)
    W, b = heads["org"]
    assert len(W) == 3 and len(W[0]) == 8 and len(b) == 3
    m = tch.evaluate(W, b, X, y, cats)
    assert m["macro_accuracy"] > 0.97 and set(m["per_class"]) == set(cats)


def test_train_heads_rejects_unknown_label():
    X, y = _clusters(n=5)
    with pytest.raises(ValueError, match="not in categories"):
        tch.train_heads({"org": X}, {"org": ["nope"] * len(y)}, {"org": ["c0", "c1", "c2"]}, epochs=1)


def test_choose_min_margin_picks_smallest_margin_meeting_precision():
    probs = [[0.9, 0.05, 0.05], [0.5, 0.45, 0.05], [0.6, 0.3, 0.1], [0.34, 0.33, 0.33]]
    y_idx = [0, 1, 0, 2]                      # 2 件目は誤り（margin 0.05）、4 件目は誤り（margin 0.01）
    m, table = tch.choose_min_margin(probs, y_idx, target_precision=0.95, grid=(0.05, 0.1, 0.2, 0.4))
    assert m == 0.1 and table[0]["margin"] == 0.05 and table[0]["precision"] < 0.95


def test_choose_min_margin_none_when_grid_never_reaches_precision():
    probs = [[0.6, 0.4], [0.55, 0.45]]
    y_idx = [1, 1]                            # 両方外れる: margin ≥ 0.05 でも precision 0
    m, table = tch.choose_min_margin(probs, y_idx, target_precision=0.95, grid=(0.05, 0.1))
    assert m is None and all(row["precision"] == 0.0 for row in table)


def test_choose_min_margin_precision_is_none_when_nothing_adopted():
    probs = [[0.6, 0.4]]
    y_idx = [0]
    m, table = tch.choose_min_margin(probs, y_idx, target_precision=0.95, grid=(0.5,))
    assert m is None and table[0]["precision"] is None and table[0]["n"] == 0


def test_save_head_roundtrip_and_encoder_sha(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake-encoder")
    heads = {"org": ([[0.1] * 4, [0.2] * 4], [0.0, 0.1]), "location": ([[0.3] * 4], [0.5])}
    tch.save_head(str(tmp_path), heads, {"org": ["a", "b"], "location": ["x"]}, {"org": 0.15, "location": 0.2},
                  {"data_sha256": "d" * 64, "metrics": {"org": {"macro_accuracy": 0.9}}})
    import json
    from safetensors.numpy import load_file
    cfg = json.load(open(tmp_path / "category_head.json", encoding="utf-8"))
    assert cfg["encoder_sha256"] == tch.encoder_sha256(str(tmp_path)) and cfg["categories"]["org"] == ["a", "b"]
    assert cfg["min_margin"] == {"org": 0.15, "location": 0.2} and cfg["version"] == 2
    assert cfg["head_type"] == "linear" and cfg["features"] == "span_mean" and cfg["hidden"] is None
    w = load_file(str(tmp_path / "category_head.safetensors"))
    assert w["org.weight"].shape == (2, 4) and w["location.bias"].shape == (1,)


def test_save_head_uses_meta_encoder_sha_when_out_dir_differs(tmp_path):
    model_dir = tmp_path / "model"; model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"real-encoder")
    out_dir = tmp_path / "out"
    tch.save_head(str(out_dir), {"org": ([[0.1, 0.2]], [0.0])}, {"org": ["a"]}, {"org": None},
                  {"encoder_sha256": tch.encoder_sha256(str(model_dir)), "encoder_dir": "model"})
    cfg = json.load(open(out_dir / "category_head.json", encoding="utf-8"))
    assert cfg["encoder_sha256"] == tch.encoder_sha256(str(model_dir)) and cfg["encoder_dir"] == "model"
    assert not (out_dir / "model.safetensors").exists()          # out 側に encoder は要らない


def test_parse_merge_valid_and_invalid():
    assert tch._parse_merge("org:party=association") == ("org", "party", "association")
    with pytest.raises(argparse.ArgumentTypeError):
        tch._parse_merge("org-party-association")


def test_effective_categories_applies_merge_and_validates():
    cats = tch._effective_categories(["org"], {"org": [("party", "association")]})
    assert "party" not in cats["org"] and "association" in cats["org"]
    with pytest.raises(ValueError, match="unknown"):
        tch._effective_categories(["org"], {"org": [("not_a_category", "association")]})
    with pytest.raises(ValueError, match="unknown kind"):
        tch._effective_categories(["not_a_kind"], {})


def test_apply_merges_and_by_kind_split_by_masked_flag():
    dataset = {
        "vectors": [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        "labels": ["party", "association", "party"],
        "kinds": ["org", "org", "org"],
        "masked": [False, False, True],
        "label_source": ["rule", "form", "rule"],
    }
    tch._apply_merges(dataset, {"org": [("party", "association")]})
    assert dataset["labels"] == ["association", "association", "association"]
    X_all, y_all = tch._by_kind(dataset)
    assert X_all["org"] == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] and y_all["org"] == ["association"] * 3
    X_masked, y_masked = tch._by_kind(dataset, masked=True)
    assert X_masked["org"] == [[1.0, 1.0]] and y_masked["org"] == ["association"]
    X_unmasked, _ = tch._by_kind(dataset, masked=False)
    assert len(X_unmasked["org"]) == 2
    X_rule, y_rule = tch._by_kind(dataset, masked=False, label_source="rule")
    assert X_rule["org"] == [[1.0, 0.0]] and y_rule["org"] == ["association"]      # 3 件目は masked=True で除外
    X_form, y_form = tch._by_kind(dataset, label_source="form")
    assert X_form["org"] == [[0.0, 1.0]] and y_form["org"] == ["association"]


def test_by_kind_no_type_word_is_separate_from_masked_and_falls_back_to_masked():
    # v2: Qwen ラベルは masked=False でも種別語が無い（no_type_word=True）。min_margin はこちらで絞る
    dataset = {"vectors": [(1.0,), (2.0,), (3.0,)], "labels": ["it", "it", "it"], "kinds": ["org"] * 3,
               "masked": [False, True, False], "no_type_word": [False, True, True], "label_source": ["rule", "rule", "qwen"]}
    X_ntw, _ = tch._by_kind(dataset, no_type_word=True)
    assert X_ntw["org"] == [[2.0], [3.0]]
    X_rule, _ = tch._by_kind(dataset, label_source="rule", no_type_word=False)
    assert X_rule["org"] == [[1.0]]
    v1 = {k: v for k, v in dataset.items() if k != "no_type_word"}           # v1 の形: no_type_word が無ければ masked
    assert tch._by_kind(v1, no_type_word=True)[0]["org"] == [[2.0]]


def test_by_kind_label_source_isolates_macro_accuracy_from_mislabeled_form_rows():
    # 設計 §8 の (i) は「規則ラベル付きスパン」限定。form ラベルが誤っていても (i) には影響しない
    # （macro_accuracy_all には影響する）ことを、手作りの線形ヘッドで確かめる。
    W = [[1.0, 0.0], [0.0, 1.0]]
    b = [0.0, 0.0]
    cats = ["c0", "c1"]
    rule_vectors = [(1.0, 0.0)] * 10 + [(0.0, 1.0)] * 10
    rule_labels = ["c0"] * 10 + ["c1"] * 10                    # 正しいラベル
    form_vectors = [(1.0, 0.0)] * 5 + [(0.0, 1.0)] * 5
    form_labels = ["c1"] * 5 + ["c0"] * 5                      # 故意に取り違えたラベル
    dataset = {
        "vectors": rule_vectors + form_vectors,
        "labels": rule_labels + form_labels,
        "kinds": ["org"] * 30,
        "masked": [False] * 30,
        "label_source": ["rule"] * 20 + ["form"] * 10,
    }
    X_rule, y_rule = tch._by_kind(dataset, masked=False, label_source="rule")
    X_all, y_all = tch._by_kind(dataset, masked=False)
    macro_rule = tch.evaluate(W, b, X_rule["org"], y_rule["org"], cats)
    macro_all = tch.evaluate(W, b, X_all["org"], y_all["org"], cats)
    assert macro_rule["macro_accuracy"] == 1.0
    assert macro_all["macro_accuracy"] < 1.0


def _write_spans(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_or_build_dataset_skips_none_vectors_and_caches(tmp_path, monkeypatch):
    spans_path = tmp_path / "spans_train.jsonl"
    rows = [
        {"doc_id": "d1", "text": "青葉電機に勤める。", "start": 0, "end": 4, "kind": "org", "label": "manufacturing",
         "label_source": "rule", "masked": False, "source": "out_ner"},
        {"doc_id": "d1", "text": "青葉電機に勤める。", "start": 0, "end": 2, "kind": "org", "label": "it",
         "label_source": "rule", "masked": False, "source": "out_ner"},
    ]
    _write_spans(spans_path, rows)

    import struct
    calls = []

    def fake_encode(texts, spans_per_doc, model, tokenizer, *, device, batch_size, features="span_mean"):
        calls.append(len(texts))
        return [[struct.pack("<2e", 1.0, 0.0) for _ in spans] for spans in spans_per_doc]

    monkeypatch.setattr(tch, "encode_span_vectors", fake_encode)
    cache_path = tmp_path / "vectors_train.npz"

    ds1 = tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None)
    assert calls == [1]                                  # 1 文書だけ（2 スパンをまとめて 1 回）
    assert ds1["total"] == 2 and ds1["skipped"] == 0 and len(ds1["vectors"]) == 2
    assert ds1["vectors"][0] == pytest.approx((1.0, 0.0))
    assert ds1["label_source"] == ["rule", "rule"]

    ds2 = tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None)
    assert calls == [1]                                   # キャッシュを再利用（encoder を再度呼ばない）
    assert ds2["labels"] == ds1["labels"] and ds2["kinds"] == ds1["kinds"]
    assert ds2["label_source"] == ds1["label_source"]

    ds3 = tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None, force=True)
    assert calls == [1, 1]                                 # --force は再計算する


def test_load_or_build_dataset_cache_invalidated_by_encoder_sha_change(tmp_path, monkeypatch):
    spans_path = tmp_path / "spans_train.jsonl"
    rows = [{"doc_id": "d1", "text": "text-a", "start": 0, "end": 1, "kind": "org", "label": "it",
             "label_source": "rule", "masked": False, "source": "out_ner"}]
    _write_spans(spans_path, rows)

    import struct
    calls = []

    def fake_encode(texts, spans_per_doc, model, tokenizer, *, device, batch_size, features="span_mean"):
        calls.append(len(texts))
        return [[struct.pack("<1e", 1.0)]]

    monkeypatch.setattr(tch, "encode_span_vectors", fake_encode)
    cache_path = tmp_path / "vectors_train.npz"

    tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None, encoder_sha="sha-a")
    assert calls == [1]
    tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None, encoder_sha="sha-a")
    assert calls == [1]                                    # 同じ encoder_sha ならキャッシュを再利用
    tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None, encoder_sha="sha-b")
    assert calls == [1, 1]                                 # encoder_sha が変わればキャッシュを無効化して再計算
    tch._load_or_build_dataset(str(spans_path), str(cache_path), None, None, encoder_sha="sha-b")
    assert calls == [1, 1]                                 # 新しい encoder_sha でも再度キャッシュされる


def test_load_or_build_dataset_counts_skipped_none_vectors(tmp_path, monkeypatch):
    spans_path = tmp_path / "spans_valid.jsonl"
    rows = [
        {"doc_id": "d1", "text": "text-a", "start": 0, "end": 1, "kind": "org", "label": "it",
         "label_source": "rule", "masked": False, "source": "out_ner"},
        {"doc_id": "d2", "text": "text-b", "start": 0, "end": 1, "kind": "org", "label": "it",
         "label_source": "rule", "masked": False, "source": "out_ner"},
    ]
    _write_spans(spans_path, rows)

    import struct

    def fake_encode(texts, spans_per_doc, model, tokenizer, *, device, batch_size, features="span_mean"):
        # 1 文書目はベクトルが取れる、2 文書目は None（重なるチャンクが無い想定）
        return [[struct.pack("<1e", 1.0)], [None]]

    monkeypatch.setattr(tch, "encode_span_vectors", fake_encode)
    ds = tch._load_or_build_dataset(str(spans_path), str(tmp_path / "vectors_valid.npz"), None, None)
    assert ds["total"] == 2 and ds["skipped"] == 1 and len(ds["vectors"]) == 1


def _dev_metrics(**over):
    """報告テスト用の v2 の `metrics[kind]["dev"]`。"""
    dev = {"macro_accuracy": 0.91, "n_rule": 5, "ntw_accuracy": 0.87, "n_ntw": 2,
           "margin_table": [{"margin": 0.1, "precision": 1.0, "coverage": 0.5, "n": 2}],
           "per_class": {"it": 0.4, "finance": 1.0}, "support": {"it": 2, "finance": 3}, "merge_suggestions": ["it"]}
    dev.update(over)
    return dev


def test_build_report_has_no_surface_text_only_counts():
    manifest = {
        "created": "2026-01-01T00:00:00", "model": "m", "spans": "s", "kinds": ["org"],
        "vectors": {"train": {"total": 10, "skipped": 1}, "dev": {"total": 5, "skipped": 0}},
        "metrics": {"org": {"dev": _dev_metrics(), "min_margin": 0.1, "judge": None, "accepted": None, "n_train": 10}},
        "merged": {"org": {"party": "association"}},
    }
    report = tch._build_report(manifest)
    assert "0.9100" in report and "0.8700" in report and "統合候補" in report and "it" in report
    assert "party" in report and "association" in report          # 適用した統合の記録
    assert "未判定" in report and "| judge |" not in report          # --judge 無し: judge の行も判定も出さない
    assert "## 選択は dev だけ。judge は 1 回" in report.splitlines()   # 選択と判定の分離は見出しで示す


def test_evaluate_macro_accuracy_ignores_classes_without_valid_support():
    # 規則ラベルは OTHER（org の service 等）を構成上持たない。valid に 1 件も無いクラスを recall 0 として
    # 平均に入れると、(i) の上限が (k-1)/k に張り付く（facility は 7/8=0.875 < 0.90 で到達不能）。
    # macro は valid に出現したクラスだけで平均し、出現しないクラスの recall は None（未定義）にする。
    W = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    b = [0.0, 0.0, -5.0]
    cats = ["c0", "c1", "other"]
    X = [(1.0, 0.0)] * 3 + [(0.0, 1.0)] * 2
    y = ["c0"] * 3 + ["c1"] * 2
    m = tch.evaluate(W, b, X, y, cats)
    assert m["macro_accuracy"] == 1.0
    assert m["per_class"] == {"c0": 1.0, "c1": 1.0, "other": None}
    assert m["support"] == {"c0": 3, "c1": 2, "other": 0}
    assert tch.evaluate(W, b, [], [], cats)["macro_accuracy"] == 0.0


def test_build_report_shows_unsupported_classes_as_dash():
    manifest = {
        "created": "2026-01-01T00:00:00", "model": "m", "spans": "s", "kinds": ["org"],
        "vectors": {"train": {"total": 10, "skipped": 0}, "dev": {"total": 5, "skipped": 0}},
        "metrics": {"org": {
            "dev": _dev_metrics(macro_accuracy=0.95, margin_table=[{"margin": 0.1, "precision": None, "coverage": 0.0, "n": 0}],
                                per_class={"it": 0.9, "service": None}, support={"it": 5, "service": 0}, merge_suggestions=[]),
            "min_margin": None, "judge": None, "accepted": None, "n_train": 10,
        }},
        "merged": {},
    }
    report = tch._build_report(manifest)
    assert "| service | - | 0 |" in report and "| it | 0.900 | 5 |" in report


def _fake_cli_env(tmp_path, monkeypatch, *, judge=True, with_facility=False):
    """main() を実モデル無しで動かす（v2: train / dev / judge の 3 分割、`no_type_word` 付き）。

    2 次元（span_mean）/ 6 次元（span_ctx）の偽ベクトルで動く。I/F/L/M/S の先頭文字でベクトルが決まる。
    org は分離できるベクトル、location は区別できないベクトルにする。``judge=False`` は
    `spans_judge.jsonl` を書かない（`--judge` 無しの実行が judge を必要としないことの確認用）。

    ``with_facility=True`` は facility を足す: 種別語の無いスパン（`no_type_word=True`）は正しく分離でき
    （dev で min_margin が見つかる）、dev の規則ラベル付きスパンも正しいが、**judge の**規則ラベル付き
    スパン（種別語あり）だけはベクトルとラベルを入れ違えてある（judge の (i) macro accuracy が 0.90 に
    届かない）。「dev では選べるが judge の受け入れ条件を満たさない」kind の再現用。
    """
    spans = tmp_path / "spans"; spans.mkdir()
    model = tmp_path / "model"; model.mkdir()
    (model / "model.safetensors").write_bytes(b"enc")

    def rows(n_per, swap_facility=False):
        out = []
        if with_facility:
            for i in range(n_per):
                for lab, tag, wrong in (("medical", "M", "S"), ("education", "S", "M")):
                    out.append({"doc_id": f"f{lab}{i}", "text": f"{wrong if swap_facility else tag}{i}", "start": 0, "end": 1,
                                "kind": "facility", "label": lab, "label_source": "rule", "masked": False,
                                "no_type_word": False, "source": "t", "split": "x"})
                    out.append({"doc_id": f"fm{lab}{i}", "text": f"{tag}{i}", "start": 0, "end": 1, "kind": "facility",
                                "label": lab, "label_source": "rule", "masked": True, "no_type_word": True,
                                "source": "t", "split": "x"})
        for i in range(n_per):
            for lab, tag in (("it", "I"), ("finance", "F")):
                for masked in (False, True):
                    out.append({"doc_id": f"o{lab}{i}{masked}", "text": f"{tag}{i}", "start": 0, "end": 1, "kind": "org",
                                "label": lab, "label_source": "rule", "masked": masked, "no_type_word": masked, "source": "t", "split": "x"})
                out.append({"doc_id": f"q{lab}{i}", "text": f"{tag}{i}", "start": 0, "end": 1, "kind": "org",
                            "label": lab, "label_source": "qwen", "masked": False, "no_type_word": True, "source": "t", "split": "x"})
            for lab in ("city", "town"):
                out.append({"doc_id": f"l{lab}{i}", "text": f"L{lab}{i}", "start": 0, "end": 1, "kind": "location",
                            "label": lab, "label_source": "rule", "masked": False, "no_type_word": False, "source": "t", "split": "x"})
            # 種別語なしの location は city / town を交互にする: sqrt のクラス重みは頻度の偏りを打ち消さないので、
            # 全部 city だと多数派の city を当て続けるだけで precision が 0.95 に届いてしまう（v1 の inv 重みでは
            # 釣り合っていた）。交互にして「区別できない」kind のままにする
            out.append({"doc_id": f"lm{i}", "text": f"Lm{i}", "start": 0, "end": 1, "kind": "location",
                        "label": ("city", "town")[i % 2], "label_source": "rule", "masked": True, "no_type_word": True,
                        "source": "t", "split": "x"})
        return out
    _write_spans(spans / "spans_train.jsonl", rows(20))
    _write_spans(spans / "spans_dev.jsonl", rows(5))
    if judge:
        _write_spans(spans / "spans_judge.jsonl", rows(5, swap_facility=True))
    seen = {"features": []}

    def fake_encode(texts, spans_per_doc, model, tokenizer, *, device, batch_size, features="span_mean"):
        seen["features"].append(features)
        vec = {"I": (1.0, 0.0), "F": (0.0, 1.0), "L": (0.5, 0.5), "M": (1.0, 0.0), "S": (0.0, 1.0)}
        dim = 2 if features == "span_mean" else 6
        return [[struct.pack(f"<{dim}e", *(list(vec[t[0]]) * (dim // 2))) for _ in sp] for t, sp in zip(texts, spans_per_doc)]

    monkeypatch.setattr(tch, "encode_span_vectors", fake_encode)
    monkeypatch.setattr(tch, "load_char_model", lambda path, device="cpu": None)
    monkeypatch.setattr(tch, "encoder_sha256", lambda path: "e" * 64)
    import privacy_masker.category_head as chm
    monkeypatch.setattr(chm, "encoder_sha256", lambda path: "e" * 64)
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    return spans, model, seen


def test_cli_saves_only_kinds_with_min_margin_and_caches_next_to_spans(tmp_path, monkeypatch, capsys):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100"]) == 0
    cfg = json.loads((out / "category_head.json").read_text(encoding="utf-8"))
    # location は区別できない（種別語なしの precision が 0.95 に届く margin が無い）→ ランタイムが読めない
    # min_margin=None の kind は保存しない。指標は manifest に残る
    assert list(cfg["categories"]) == ["org"] and cfg["min_margin"]["org"] is not None
    manifest = json.loads((out / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["location"]["min_margin"] is None and manifest["saved_kinds"] == ["org"]
    assert "location" in capsys.readouterr().err
    # ベクトルのキャッシュはスパンの隣（モデルフォルダ・--out には置かない）。--judge 無しなので judge は計算しない
    assert (spans / "vectors_train_span_mean.npz").exists() and not (out / "vectors_train_span_mean.npz").exists()
    assert (spans / "vectors_dev_span_mean.npz").exists() and not (spans / "vectors_judge_span_mean.npz").exists()


def test_cli_kinds_restricts_saved_head_and_rejects_unknown(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100",
                     "--kinds", "org"]) == 0
    cfg = json.loads((out / "category_head.json").read_text(encoding="utf-8"))
    assert list(cfg["categories"]) == ["org"] and list(cfg["metrics"]) == ["org"]

    # 保存できる kind が 1 つも無ければヘッドのファイルを書かない（manifest だけ残す）
    out2 = tmp_path / "out2"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out2), "--epochs", "100",
                     "--kinds", "location"]) == 0
    assert not (out2 / "category_head.json").exists() and not (out2 / "category_head.safetensors").exists()
    assert (out2 / "category_head_train_manifest.json").exists()

    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--kinds", "org,nope"])
    manifest = json.loads((out2 / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["saved_kinds"] == [] and manifest["encoder_sha256"] == "e" * 64   # 保存しない実行でも記録する


def _head_files(d):
    return sorted(p.name for p in d.iterdir() if p.name.startswith("category_head"))


def test_cli_default_out_is_candidate_next_to_spans_never_the_model_dir(tmp_path, monkeypatch):
    # 既定の書き出し先はモデルフォルダではなく <spans>/candidate（検出器は category_head="auto" で
    # モデルフォルダの category_head.json を自動で読むので、学習し直すだけで黙って採用されないように）
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100"]) == 0
    cand = spans / "candidate"
    assert _head_files(cand) == ["category_head.json", "category_head.safetensors", "category_head_train_manifest.json"]
    assert _head_files(model) == []


def test_cli_refuses_model_dir_as_out_without_install_and_out_with_install(tmp_path, monkeypatch, capsys):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--out", str(model), "--epochs", "1"])
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--out", str(model) + os.sep, "--epochs", "1"])
    capsys.readouterr()
    with pytest.raises(SystemExit):          # --judge を付けても --install と --out は併用できない
        tch.main(["--model", str(model), "--spans", str(spans), "--out", str(tmp_path / "x"), "--install", "--judge",
                  "--epochs", "1"])
    assert "do not combine it with --out" in capsys.readouterr().err       # usage 行ではなくエラー文で確かめる
    assert _head_files(model) == []


def test_cli_install_refuses_limit(tmp_path, monkeypatch, capsys):
    # --limit は動作確認用の部分データ。部分データで判定したヘッドをモデルフォルダへ書かない
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--install", "--judge", "--epochs", "1", "--limit", "10"])
    assert "--install must not be combined with --limit (partial data)" in capsys.readouterr().err
    assert _head_files(model) == [] and not (spans / "vectors_train_span_mean.npz").exists()   # 何も計算・書き出ししない
    # 候補の書き出し（--install なし）は --limit を許す
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(tmp_path / "lim"), "--epochs", "1",
                     "--limit", "10"]) == 0


def test_cli_install_refuses_target_precision_below_acceptance_gate(tmp_path, monkeypatch, capsys):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--install", "--judge", "--epochs", "1",
                  "--target-precision", "0.8"])
    assert "--install requires --target-precision" in capsys.readouterr().err
    assert _head_files(model) == []
    # 候補の書き出し（--install なし）は緩い目標でも許す（manifest に記録される）
    out = tmp_path / "loose"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100",
                     "--target-precision", "0.8"]) == 0
    assert json.load(open(out / tch.TRAIN_MANIFEST, encoding="utf-8"))["acceptance"]["target_precision"] == 0.8


def test_cli_manifest_records_encoder_data_sha_and_metrics_for_every_kind(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100"]) == 0
    manifest = json.loads((spans / "candidate" / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["encoder_sha256"] == "e" * 64
    assert set(manifest["data_sha256"]) == {"train", "dev"}                     # --judge 無しでは judge を読まない
    assert all(v.startswith(tch._sha256_file(str(spans / f"spans_{k}.jsonl"))) for k, v in manifest["data_sha256"].items())
    assert set(manifest["metrics"]) == {"org", "location"}                     # 保存しない kind の指標も残す
    assert manifest["install"] is False and manifest["out"] == str(spans / "candidate")
    assert manifest["acceptance"] == {"macro_accuracy_min": 0.90, "target_precision": 0.95}
    assert manifest["judge"] is False and set(manifest["vectors"]) == {"train", "dev"}
    assert manifest["features"] == "span_mean" and manifest["head_type"] == "linear" and manifest["hidden"] is None
    assert manifest["class_weight"] == "sqrt" and manifest["lr"] == 0.05 and manifest["patience"] == 5


def test_cli_accepted_needs_both_gates_and_install_writes_only_accepted_kinds(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch, with_facility=True)
    report = tmp_path / "report.md"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge",
                     "--report", str(report)]) == 0
    cand = spans / "candidate"
    cfg = json.loads((cand / "category_head.json").read_text(encoding="utf-8"))
    # 候補（--install 無し）は min_margin のある kind をすべて含むが、kind ごとに accepted を持つ
    assert sorted(cfg["categories"]) == ["facility", "org"]
    fac = cfg["metrics"]["facility"]
    # facility は dev では選べる（min_margin あり・dev の macro も満点）が、judge の (i) が 0.90 に届かない
    assert fac["min_margin"] is not None and fac["dev"]["macro_accuracy"] == 1.0
    assert fac["judge"]["macro_accuracy"] < 0.90 and fac["accepted"] is False
    assert cfg["metrics"]["org"]["accepted"] is True
    manifest = json.loads((cand / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["location"]["accepted"] is False                 # min_margin 無し
    assert manifest["accepted_kinds"] == ["org"] and manifest["saved_kinds"] == ["facility", "org"]
    text = report.read_text(encoding="utf-8")
    assert "受け入れ条件" in text and "accepted_kinds" not in text
    assert _head_files(model) == []

    # --install: 受け入れた kind（org）だけをモデルフォルダに書く。manifest もモデルフォルダへ
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge", "--install"]) == 0
    installed = json.loads((model / "category_head.json").read_text(encoding="utf-8"))
    assert list(installed["categories"]) == ["org"] and list(installed["min_margin"]) == ["org"]
    assert list(installed["metrics"]) == ["org"] and installed["metrics"]["org"]["accepted"] is True
    from safetensors.numpy import load_file
    assert sorted(load_file(str(model / "category_head.safetensors"))) == ["org.bias", "org.weight"]
    manifest = json.loads((model / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["install"] is True and manifest["saved_kinds"] == ["org"]
    assert set(manifest["metrics"]) == {"org", "facility", "location"}


def test_cli_install_with_no_accepted_kind_writes_no_head_files(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch, with_facility=True)
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge", "--install",
                     "--kinds", "facility,location"]) == 0
    assert _head_files(model) == ["category_head_train_manifest.json"]
    manifest = json.loads((model / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["saved_kinds"] == [] and manifest["accepted_kinds"] == []


# ---------------------------------------------------------------------------
# tools/train_category_head.py v2（features / MLP / sqrt クラス重み / dev で選び --judge で判定）
# ---------------------------------------------------------------------------

def test_class_weights_sqrt_is_capped_and_mean_one():
    y = ["a"] * 100 + ["b"] * 1 + ["c"] * 4
    w = tch._class_weights(y, ["a", "b", "c"], mode="sqrt", cap=5.0)
    # k=3 では 1 クラスの重みは最大でも 3 なので上限 5 はかからず、平均はちょうど 1（最大 1.875）
    assert max(w) <= 5.0 + 1e-9 and sum(w) / 3 == pytest.approx(1.0)
    assert w[1] > w[2] > w[0]
    assert tch._class_weights(y, ["a", "b", "c", "d"], mode="sqrt")[3] == 0.0


def test_class_weights_cap_binds_only_the_rare_class():
    # 正規化は訓練データに無いクラスも含む全カテゴリ数 k で割る。k=8（facility / location の大きさ）で
    # 1 件しかないクラスは 8/1.01 ≈ 7.92 になり上限 5 で切られる。もう一方は上限にかからず値が変わらない
    cats = ["a", "b", "c", "d", "e", "f", "g", "h"]
    y = ["a"] * 10000 + ["b"]
    for mode, inv_a in (("sqrt", 1 / 100), ("inv", 1 / 10000)):
        w = tch._class_weights(y, cats, mode=mode, cap=5.0)
        assert w[1] == 5.0                                                          # 上限そのもの
        assert w[0] == pytest.approx(inv_a * 8 / (inv_a + 1.0))                     # 上限なしの正規化値のまま
        assert w[2:] == [0.0] * 6 and sum(w) / 8 < 1.0                              # 上限で切った分だけ平均は 1 より下
    # 2 カテゴリでは最大でも k=2 なので、極端な偏りでも上限はかからない
    assert max(tch._class_weights(y, ["a", "b"], mode="sqrt", cap=5.0)) < 2.0


def test_is_accepted_truth_table():
    target = 0.95
    cases = [
        ((1.0, 0.95, 0.3), True),        # (ii) がちょうど target
        ((0.90, 1.0, 0.3), True),        # (i) がちょうど 0.90
        ((0.89, 1.0, 0.3), False),       # (i) 未達
        ((1.0, 0.949, 0.3), False),      # (ii) 未達
        ((1.0, 1.0, None), False),       # dev で min_margin が決まらない
        ((None, 1.0, 0.3), False),       # judge の規則ラベル付きスパンが無い
        ((1.0, None, 0.3), False),       # min_margin 以上の予測が judge に 1 件も無い
        ((None, None, None), False),
    ]
    for (macro, precision, min_margin), expected in cases:
        assert tch.is_accepted(macro, precision, min_margin, target) is expected, (macro, precision, min_margin)
    assert tch.is_accepted(1.0, 0.96, 0.3, 0.97) is False                          # target は引数で決まる


def test_train_heads_mlp_returns_numpy_weights_and_predicts_with_mlp_softmax():
    X = {"org": [[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20}
    y = {"org": ["it"] * 20 + ["finance"] * 20}
    heads = tch.train_heads(X, y, {"org": ["it", "finance"]}, head="mlp", hidden=4, epochs=300, lr=0.01, X_dev=X, y_dev=y)
    W1, b1, W2, b2 = heads["org"]
    assert W1.shape == (4, 2) and W2.shape == (2, 4) and W1.dtype.name == "float32"
    probs = tch.predict_probs("mlp", heads["org"], [[1.0, 0.0], [0.0, 1.0]])
    assert probs[0][0] > 0.9 and probs[1][1] > 0.9
    assert tch.evaluate_probs(probs, ["it", "finance"], ["it", "finance"])["macro_accuracy"] == 1.0


def test_train_heads_mlp_early_stopping_restores_best_dev_weights_and_is_deterministic(monkeypatch):
    X = {"org": [[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20}
    y = {"org": ["it"] * 20 + ["finance"] * 20}
    # dev は同じ点で 3 割のラベルを入れ違える: macro accuracy は 0.7 で頭打ちになり、確信が 0.7 を越えると
    # dev の NLL が上がり始めるので、patience で早く止まり、NLL が最小だった時点の重みに戻るはず
    y_dev = {"org": ["it"] * 14 + ["finance"] * 6 + ["finance"] * 14 + ["it"] * 6}
    cats = {"org": ["it", "finance"]}
    kw = dict(head="mlp", hidden=4, epochs=300, lr=0.01, X_dev=X, y_dev=y_dev)
    seen = []
    orig = tch.evaluate_probs

    def spy(probs, labels, categories):              # train_heads が早期終了の判定で見た dev の値を控える
        r = orig(probs, labels, categories)
        seen.append((r["macro_accuracy"], tch._mean_nll(probs, labels, categories)))
        return r

    monkeypatch.setattr(tch, "evaluate_probs", spy)
    heads = tch.train_heads(X, y, cats, **kw)
    monkeypatch.setattr(tch, "evaluate_probs", orig)
    assert 0 < len(seen) < 300 // tch.EARLY_STOP_EVERY                              # patience で途中で止まった
    probs = tch.predict_probs("mlp", heads["org"], X["org"])
    best = max(score for score, _ in seen)
    assert orig(probs, y_dev["org"], cats["org"])["macro_accuracy"] == pytest.approx(best, abs=1e-9)
    best_nll = min(nll for score, nll in seen if score >= best - 1e-9)
    restored_nll = tch._mean_nll(probs, y_dev["org"], cats["org"])
    assert restored_nll == pytest.approx(best_nll, rel=1e-6) and seen[-1][1] > restored_nll   # 最後ではなく最良の重み
    # (a) 同じ seed なら重みはバイト単位で一致する（dropout も torch の乱数）
    again = tch.train_heads(X, y, cats, **kw)
    assert all(heads["org"][i].tobytes() == again["org"][i].tobytes() for i in range(4))
    # dev を渡さない mlp は最後まで学習する（= dev が結果を変えている）
    assert tch.train_heads(X, y, cats, head="mlp", hidden=4, epochs=300, lr=0.01)["org"][0].tobytes() != heads["org"][0].tobytes()
    # (c) linear は X_dev を使わない（渡しても渡さなくても同じ重み）
    lin = dict(head="linear", epochs=50)
    assert tch.train_heads(X, y, cats, **lin, X_dev=X, y_dev=y_dev) == tch.train_heads(X, y, cats, **lin)


def test_precision_at_margin():
    probs = [[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]]
    r = tch.precision_at_margin(probs, [0, 1, 0], 0.5)          # 3 件目は margin 0.6 で採用され、予測 finance ≠ 正解 it
    assert r == {"precision": 0.5, "coverage": 2 / 3, "n": 2, "n_total": 3}
    assert tch.precision_at_margin([], [], 0.5)["precision"] is None


def test_cli_v2_dev_only_run_never_needs_judge_and_records_features(tmp_path, monkeypatch):
    spans, model, seen = _fake_cli_env(tmp_path, monkeypatch, judge=False)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100", "--features", "span_ctx"]) == 0
    assert set(seen["features"]) == {"span_ctx"}
    cfg = json.loads((out / "category_head.json").read_text(encoding="utf-8"))
    assert cfg["features"] == "span_ctx" and cfg["head_type"] == "linear" and cfg["version"] == 2
    m = json.loads((out / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    assert m["features"] == "span_ctx" and m["metrics"]["org"]["judge"] is None and m["metrics"]["org"]["accepted"] is None
    assert m["metrics"]["org"]["dev"]["n_ntw"] > 0 and m["metrics"]["org"]["min_margin"] is not None
    assert (spans / "vectors_train_span_ctx.npz").exists()                      # キャッシュ名に features を含める


def test_cli_v2_judge_computes_acceptance_and_install_requires_judge(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--install", "--epochs", "1"])          # --judge 無し
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100", "--judge"]) == 0
    m = json.loads((out / "category_head_train_manifest.json").read_text(encoding="utf-8"))
    j = m["metrics"]["org"]["judge"]
    assert j["macro_accuracy"] == 1.0 and j["precision"] == 1.0 and j["coverage"] > 0 and m["metrics"]["org"]["accepted"] is True
    assert m["metrics"]["location"]["min_margin"] is None and m["metrics"]["location"]["accepted"] is False
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge", "--install"]) == 0
    cfg = json.loads((model / "category_head.json").read_text(encoding="utf-8"))
    assert list(cfg["categories"]) == ["org"]


def test_cli_v2_mlp_head_is_loadable_by_head_categorizer(tmp_path, monkeypatch):
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "300", "--lr", "0.01", "--head", "mlp", "--hidden", "4"]) == 0
    cfg = json.loads((out / "category_head.json").read_text(encoding="utf-8"))
    assert cfg["head_type"] == "mlp" and cfg["hidden"] == 4
    from privacy_masker.category_head import HeadCategorizer
    h = HeadCategorizer.load(str(model), head_dir=str(out))
    assert h.head_type == "mlp" and h("org", struct.pack("<2e", 1.0, 0.0))[0] == "it"


def test_cli_v2_judge_is_never_used_for_selection(tmp_path, monkeypatch):
    # 設計 §3.1: 選択（重み・min_margin・統合候補）は dev だけで行う。同じ train / dev なら、judge を変えても
    # 読まなくても選択は変わらず、変わるのは judge の指標と accepted だけ。--judge 無しでは judge を一切読まない
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch, with_facility=True)

    def run(name, *extra):
        out = tmp_path / name
        assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100", *extra]) == 0
        manifest = json.loads((out / tch.TRAIN_MANIFEST).read_text(encoding="utf-8"))
        selection = {k: (m["min_margin"], m["dev"]) for k, m in manifest["metrics"].items()}
        return manifest, selection, (out / "category_head.safetensors").read_bytes()

    judged, sel_a, w_a = run("a", "--judge")
    assert judged["metrics"]["facility"]["accepted"] is False                   # judge の規則ラベルは入れ違え
    (spans / "spans_judge.jsonl").write_text("{not json\n", encoding="utf-8")   # 読めば壊れる judge
    plain, sel_b, w_b = run("b")
    assert "judge" not in plain["vectors"] and plain["metrics"]["facility"]["accepted"] is None
    (spans / "spans_judge.jsonl").write_text((spans / "spans_dev.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    rejudged, sel_c, w_c = run("c", "--judge")
    assert rejudged["metrics"]["facility"]["accepted"] is True                  # judge を変えると判定だけが変わる
    assert sel_a == sel_b == sel_c and w_a == w_b == w_c
    (spans / "spans_judge.jsonl").unlink()
    with pytest.raises(SystemExit):
        tch.main(["--model", str(model), "--spans", str(spans), "--out", str(tmp_path / "d"), "--epochs", "1", "--judge"])


def _rewrite_spans(path, fn):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        fn(r)
    _write_spans(path, rows)


def test_cli_v2_judge_precision_gate_alone_rejects_and_blocks_install(tmp_path, monkeypatch):
    # (i) は満点のまま、judge の種別語なしスパン（(ii) の対象）だけラベルを入れ違える → (ii) だけで不合格
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)

    def swap_org_ntw(r):
        if r["kind"] == "org" and r["no_type_word"]:
            r["label"] = {"it": "finance", "finance": "it"}[r["label"]]
    _rewrite_spans(spans / "spans_judge.jsonl", swap_org_ntw)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100", "--judge"]) == 0
    m = json.loads((out / tch.TRAIN_MANIFEST).read_text(encoding="utf-8"))
    org = m["metrics"]["org"]
    assert org["min_margin"] is not None and org["judge"]["macro_accuracy"] == 1.0
    assert org["judge"]["precision"] < 0.95 and org["accepted"] is False
    assert list(json.loads((out / "category_head.json").read_text(encoding="utf-8"))["categories"]) == ["org"]  # 候補には残る
    # --install は accepted の kind だけ → org は入らない（location も min_margin 無し）ので、ヘッドは書かない
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge", "--install"]) == 0
    assert _head_files(model) == ["category_head_train_manifest.json"]
    installed = json.loads((model / tch.TRAIN_MANIFEST).read_text(encoding="utf-8"))
    assert installed["saved_kinds"] == [] and "org" not in installed["accepted_kinds"]


def test_cli_v2_judge_precision_uses_the_dev_min_margin_not_one_chosen_on_judge(tmp_path, monkeypatch):
    # 設計 §3.1: (ii) は dev で選んだ min_margin での judge の precision。judge で margin を選び直さない。
    # 固定の線形ヘッド（org の上位 2 クラスだけが効く）と、margin を狙って置いたベクトルで作る:
    #   dev の種別語なし: 正解 20 件（margin 0.72）+ 誤り 2 件（0.27）+ 誤り 2 件（0.12）→ precision 0.95 に届く最小は 0.30
    #   judge の種別語なし: 正解 10 件（0.72）+ 誤り 2 件（0.52）→ 0.30 では 10/12 < 0.95、0.55 以上なら 10/10
    # judge で margin を選び直す退行（choose_min_margin を judge に掛ける）なら 0.55 で 1.0 になり accepted が True に変わる
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)
    cats = list(tch.CATEGORIES["org"])
    vec_by_text: dict = {}

    def row(split, i, *, correct, margin, pred, ntw):
        d = 2 * math.atanh(margin)                    # 2 クラスの softmax で margin = tanh(ロジット差 / 2)
        text = f"{split}-{'n' if ntw else 'r'}{i}"
        vec_by_text[text] = (d, 0.0) if pred == 0 else (0.0, d)
        return {"doc_id": text, "text": text, "start": 0, "end": 1, "kind": "org", "label": cats[pred if correct else 1 - pred],
                "label_source": ("qwen" if ntw else "rule"), "masked": False, "no_type_word": ntw, "source": "t", "split": split}

    def rule_rows(split, n):
        return [row(split, i, correct=True, margin=0.72, pred=i % 2, ntw=False) for i in range(n)]

    def ntw_rows(split, spec):
        out, i = [], 0
        for n, correct, margin in spec:
            for _ in range(n):
                out.append(row(split, i, correct=correct, margin=margin, pred=i % 2, ntw=True)); i += 1
        return out

    _write_spans(spans / "spans_train.jsonl", rule_rows("train", 20))
    _write_spans(spans / "spans_dev.jsonl", rule_rows("dev", 10) + ntw_rows("dev", [(20, True, 0.72), (2, False, 0.27), (2, False, 0.12)]))
    _write_spans(spans / "spans_judge.jsonl", rule_rows("judge", 10) + ntw_rows("judge", [(10, True, 0.72), (2, False, 0.52)]))

    def fake_encode(texts, spans_per_doc, model, tokenizer, *, device, batch_size, features="span_mean"):
        return [[struct.pack("<2e", *vec_by_text[t]) for _ in sp] for t, sp in zip(texts, spans_per_doc)]

    def fixed_heads(X, y, categories, **kw):           # 学習の揺れを外し、margin をベクトルだけで決める
        heads = {}
        for kind in X:
            k = len(categories[kind])
            W = [[1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (k - 2)
            b = [0.0, 0.0] + [-50.0] * (k - 2)
            heads[kind] = (W, b)
        return heads

    monkeypatch.setattr(tch, "encode_span_vectors", fake_encode)
    monkeypatch.setattr(tch, "train_heads", fixed_heads)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "1", "--judge"]) == 0
    org = json.loads((out / tch.TRAIN_MANIFEST).read_text(encoding="utf-8"))["metrics"]["org"]
    assert org["min_margin"] == pytest.approx(0.30)                                 # dev で選んだ値
    assert org["judge"]["macro_accuracy"] == 1.0                                    # (i) は合格（(ii) だけで決まる）
    assert org["judge"]["precision"] < 0.95 and org["accepted"] is False           # judge で選び直すと 1.0 / True になる
    assert org["judge"]["precision"] == pytest.approx(10 / 12)
    assert org["judge"]["n_kept"] == 12 and org["judge"]["n_ntw"] == 12             # 0.30 以上の judge 12 件を採用


def test_cli_v2_majority_class_head_passes_precision_but_fails_macro_accuracy(tmp_path, monkeypatch):
    # ブリーフの元のフィクスチャ（location の種別語なし行が全部 city）: 区別できない location でも sqrt 重みの
    # ヘッドは多数派 city を当て続け、種別語なしの precision（micro）は 1.0 で min_margin も選べてしまう。
    # それを止めるのは (i)（規則ラベル付き city / town が半々 → macro accuracy 0.5）
    spans, model, _ = _fake_cli_env(tmp_path, monkeypatch)

    def all_city(r):
        if r["kind"] == "location" and r["no_type_word"]:
            r["label"] = "city"
    for split in ("train", "dev", "judge"):
        _rewrite_spans(spans / f"spans_{split}.jsonl", all_city)
    out = tmp_path / "out"
    assert tch.main(["--model", str(model), "--spans", str(spans), "--out", str(out), "--epochs", "100", "--judge"]) == 0
    loc = json.loads((out / tch.TRAIN_MANIFEST).read_text(encoding="utf-8"))["metrics"]["location"]
    assert loc["min_margin"] is not None and loc["judge"]["precision"] == 1.0
    assert loc["judge"]["macro_accuracy"] < 0.90 and loc["accepted"] is False
    assert tch.main(["--model", str(model), "--spans", str(spans), "--epochs", "100", "--judge", "--install"]) == 0
    assert list(json.loads((model / "category_head.json").read_text(encoding="utf-8"))["categories"]) == ["org"]


def test_build_report_marks_acceptance_per_kind():
    def kind(macro, precision, min_margin, accepted):
        return {"dev": _dev_metrics(margin_table=[], per_class={}, support={}, merge_suggestions=[]),
                "min_margin": min_margin, "n_train": 1, "accepted": accepted,
                "judge": {"macro_accuracy": macro, "n_rule": 1, "precision": precision, "coverage": 0.5,
                          "n_kept": 1, "n_ntw": 2, "per_class": {}}}
    manifest = {
        "created": "2026-01-01T00:00:00", "model": "m", "spans": "s", "kinds": ["facility", "org"], "judge": True,
        "vectors": {"train": {"total": 1, "skipped": 0}, "dev": {"total": 1, "skipped": 0}, "judge": {"total": 1, "skipped": 0}},
        "metrics": {"facility": kind(0.5, 1.0, 0.1, False), "org": kind(0.95, 0.96, 0.2, True)},
        "merged": {}, "accepted_kinds": ["org"],
    }
    report = tch._build_report(manifest)
    assert "- 受け入れ条件（judge: (i) macro accuracy ≥ 0.90 かつ (ii) precision ≥ 0.95）: ×" in report
    assert "- 受け入れ条件（judge: (i) macro accuracy ≥ 0.90 かつ (ii) precision ≥ 0.95）: ○" in report
    assert "## 受け入れた kind" in report and "| judge | 1 | 0 |" in report and "0.9600" in report
