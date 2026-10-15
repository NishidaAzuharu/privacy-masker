"""分類ヘッド（計画 B）の HeadCategorizer。重みの書き出しに numpy / safetensors を使うので、
CI の minimal profile（``.[dev]`` のみ）では skip する。torch 無しで import できることの確認は
``tests/test_category_head_import.py``（numpy / safetensors も不要）。"""
import json

import pytest

np = pytest.importorskip("numpy")
save_file = pytest.importorskip("safetensors.numpy").save_file

from privacy_masker.category_head import (  # noqa: E402
    CATEGORY_HEAD_CONFIG,
    CATEGORY_HEAD_WEIGHTS,
    HeadCategorizer,
    _decode_vector,
    encoder_sha256,
)
from privacy_masker.exceptions import ConfigurationError  # noqa: E402


def _write_head(d, sha=None, dim=4):
    (d / "model.safetensors").write_bytes(b"encoder-bytes")
    W = np.zeros((2, dim), dtype=np.float32)
    W[0, 0] = 4.0
    W[1, 1] = 4.0
    b = np.zeros(2, dtype=np.float32)
    save_file({"org.weight": W, "org.bias": b}, str(d / CATEGORY_HEAD_WEIGHTS))
    cfg = {"version": 1, "encoder_sha256": sha or encoder_sha256(str(d)), "categories": {"org": ["it", "finance"]},
           "min_margin": {"org": 0.2}}
    (d / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")


def _vec(*xs):
    return np.array(xs, dtype="<f2").tobytes()


def test_load_and_predict(tmp_path):
    _write_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    assert h.categories["org"] == ["it", "finance"] and h.min_margin["org"] == 0.2
    cat, margin = h("org", _vec(1, 0, 0, 0))
    assert cat == "it" and margin > 0.9
    cat2, margin2 = h("org", _vec(0.5, 0.5, 0, 0))
    assert cat2 in ("it", "finance") and margin2 < 0.05          # 決めきれない → margin が小さい


def test_unknown_kind_or_bad_dimension_falls_back(tmp_path):
    _write_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    assert h("facility", _vec(1, 0, 0, 0)) == ("other", 0.0)
    assert h("org", _vec(1, 0)) == ("service", 0.0)


def test_encoder_sha_mismatch_and_corruption(tmp_path):
    _write_head(tmp_path, sha="0" * 64)
    with pytest.raises(ConfigurationError, match="encoder"):
        HeadCategorizer.load(str(tmp_path))
    _write_head(tmp_path)
    (tmp_path / CATEGORY_HEAD_WEIGHTS).write_bytes(b"garbage")
    with pytest.raises(ConfigurationError):
        HeadCategorizer.load(str(tmp_path))



@pytest.mark.parametrize("min_margin", [{"org": None}, {}, None, {"org": "0.2"}, {"org": True}, {"org": 1.5}, {"org": float("nan")}, "missing"])
def test_null_or_missing_min_margin_is_configuration_error(tmp_path, min_margin):
    # 学習ツールは min_margin が見つからない kind を null で書いていた時期がある（計画 B Task 6 で保存しないよう修正）。
    # null・欠落・数値でない値は TypeError ではなく、kind を名指しした ConfigurationError にする
    _write_head(tmp_path)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    if min_margin == "missing":
        del cfg["min_margin"]
    else:
        cfg["min_margin"] = min_margin
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="min_margin.*'org'"):
        HeadCategorizer.load(str(tmp_path))


def test_scalar_min_margin_applies_to_every_kind(tmp_path):
    _write_head(tmp_path)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["min_margin"] = 0.3
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    assert HeadCategorizer.load(str(tmp_path)).min_margin == {"org": 0.3}

@pytest.mark.parametrize("cfg", [[], "x", {"encoder_sha256": None, "categories": ["org"]},
                                 {"encoder_sha256": None, "categories": {"org": "it"}},
                                 {"encoder_sha256": None, "categories": {"org": {"it": 1}}}])
def test_malformed_config_is_configuration_error(tmp_path, cfg):
    # JSON としては読めるが形が違う（dict でない・categories が kind → list の dict でない）→ AttributeError 等
    # ではなく ConfigurationError。encoder_sha256 の照合より先に形を確かめる
    _write_head(tmp_path)
    if isinstance(cfg, dict):
        cfg = dict(cfg, encoder_sha256=encoder_sha256(str(tmp_path)), min_margin={"org": 0.2})
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="category_head.json"):
        HeadCategorizer.load(str(tmp_path))


@pytest.mark.parametrize("weight, bias", [
    (np.array([4.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32)),              # W が 1 次元（len は合う）
    (np.zeros((2, 4), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)),             # b が 2 次元（len は合う）
    (np.zeros((2, 4, 1), dtype=np.float32), np.zeros(2, dtype=np.float32)),
])
def test_weight_rank_mismatch_is_configuration_error(tmp_path, weight, bias):
    # 行数だけを見ると通ってしまい、推論時に TypeError になる形 → 読み込み時に ConfigurationError
    _write_head(tmp_path)
    save_file({"org.weight": weight, "org.bias": bias}, str(tmp_path / CATEGORY_HEAD_WEIGHTS))
    with pytest.raises(ConfigurationError, match="shape"):
        HeadCategorizer.load(str(tmp_path))


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_empty_categories_for_a_kind_is_configuration_error(tmp_path, head_type):
    # categories が空の kind は 0 行の重みなら形状検査を素通りし、推論では常に OTHER に落ちる → 読み込み時に拒否する
    _write_head(tmp_path)
    if head_type == "linear":
        weights = {"org.weight": np.zeros((0, 4), dtype=np.float32), "org.bias": np.zeros(0, dtype=np.float32)}
    else:
        weights = {"org.w1": np.zeros((3, 4), dtype=np.float32), "org.b1": np.zeros(3, dtype=np.float32),
                   "org.w2": np.zeros((0, 3), dtype=np.float32), "org.b2": np.zeros(0, dtype=np.float32)}
    save_file(weights, str(tmp_path / CATEGORY_HEAD_WEIGHTS))
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg.update(head_type=head_type, categories={"org": []})
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no categories for 'org'"):
        HeadCategorizer.load(str(tmp_path))


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeadCategorizer.load(str(tmp_path))


def test_missing_weights_file_is_file_not_found(tmp_path):
    # category_head.json はあるが category_head.safetensors が無い → FileNotFoundError
    # （ConfigurationError に丸め込まない。壊れたファイルと違い「無い」ことは明確なので）
    _write_head(tmp_path)
    (tmp_path / CATEGORY_HEAD_WEIGHTS).unlink()
    with pytest.raises(FileNotFoundError):
        HeadCategorizer.load(str(tmp_path))


def test_decode_vector_matches_char_head_vector_to_floats():
    # category_head._decode_vector は char_head.vector_to_floats と同じロジックの複製
    # （torch をトップレベル import する char_head から呼ばずに済ませるため）。バイト互換を確認する。
    pytest.importorskip("torch")
    from privacy_masker.char_head import vector_to_floats

    data = _vec(1.0, -0.5, 0.25, 0.0)
    assert _decode_vector(data) == vector_to_floats(data)


def test_detector_autoloads_head(tmp_path, monkeypatch):
    """CharHeadNerDetector(model=dir) はフォルダの category_head.json を読む（load_char_model はフェイク）。

    ``CharHeadNerDetector.__init__`` は ``from .. import char_head as ch`` を遅延 import するので
    （``privacy_masker.detectors`` を torch 無しで import できる既存の方針を壊さない）、パッチ対象は
    ``dch`` 自身ではなく、その import 先である共有モジュール ``privacy_masker.char_head`` にする
    （同じモジュールオブジェクトの属性を書き換えるので、どちらの別名からでも効く）。
    """
    pytest.importorskip("torch")
    import privacy_masker.char_head as shared_ch
    _write_head(tmp_path)
    from privacy_masker.detectors import char_head as dch
    from tests.test_char_head_vectors import HiddenRuleModel, _setup

    tok, doc, gold = _setup()
    model = HiddenRuleModel(doc, tok, gold)
    monkeypatch.setattr(shared_ch, "load_char_model", lambda path, device="cpu", **kw: model)
    det = dch.CharHeadNerDetector(model=str(tmp_path), tokenizer=tok)
    assert isinstance(det.category_head, HeadCategorizer)
    det2 = dch.CharHeadNerDetector(model=str(tmp_path), tokenizer=tok, category_head=False)
    assert det2.category_head is None
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        dch.CharHeadNerDetector(model=str(missing_dir), tokenizer=tok, category_head=True)


@pytest.mark.parametrize("value", [1, 0, "yes", None, "AUTO"])
def test_detector_rejects_non_bool_category_head(value):
    # 1 / 0 は == で True / False と等しいが bool ではない → 受け付けない（"auto" 以外の文字列・None も）
    pytest.importorskip("torch")
    from privacy_masker.detectors import char_head as dch

    with pytest.raises(ConfigurationError, match="category_head"):
        dch.CharHeadNerDetector(model=object(), category_head=value)


def test_load_category_head_only_true_forces_loading(tmp_path):
    # 「必ず読む（無ければ FileNotFoundError）」は bool の True だけ。"auto" は category_head.json が無ければ None
    from privacy_masker.detectors import char_head as dch

    load = dch.CharHeadNerDetector._load_category_head
    assert load(str(tmp_path), "auto") is None and load(str(tmp_path), False) is None
    assert load(str(tmp_path), 1) is None                                  # 真値でも bool でなければ強制しない
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path), True)
    _write_head(tmp_path)
    assert isinstance(load(str(tmp_path), "auto"), HeadCategorizer)


def test_features_is_read_from_config_and_validated(tmp_path):
    _write_head(tmp_path)
    assert HeadCategorizer.load(str(tmp_path)).features == "span_mean"          # 省略時（v1 のヘッド）
    _write_head(tmp_path, dim=6)          # span_ctx は次元が 3 の倍数でなければならない（次元照合、設計 §2.1）
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    assert HeadCategorizer.load(str(tmp_path)).features == "span_ctx"
    cfg["features"] = "span_max"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="features"):
        HeadCategorizer.load(str(tmp_path))


def test_features_constant_matches_char_head():
    ch = pytest.importorskip("privacy_masker.char_head")
    from privacy_masker.category_head import FEATURES
    assert FEATURES == ch.FEATURES


def _write_mlp_head(d, hidden=3, sha=None):
    torch = pytest.importorskip("torch")
    (d / "model.safetensors").write_bytes(b"encoder-bytes")
    torch.manual_seed(0)
    lin1, lin2 = torch.nn.Linear(4, hidden), torch.nn.Linear(hidden, 2)
    save_file({"org.w1": lin1.weight.detach().numpy(), "org.b1": lin1.bias.detach().numpy(),
               "org.w2": lin2.weight.detach().numpy(), "org.b2": lin2.bias.detach().numpy()}, str(d / CATEGORY_HEAD_WEIGHTS))
    cfg = {"version": 2, "head_type": "mlp", "hidden": hidden, "features": "span_mean", "encoder_sha256": sha or encoder_sha256(str(d)),
           "categories": {"org": ["it", "finance"]}, "min_margin": {"org": 0.2}}
    (d / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    return lin1, lin2


def test_mlp_head_matches_torch_reference(tmp_path):
    torch = pytest.importorskip("torch")
    lin1, lin2 = _write_mlp_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    assert h.head_type == "mlp" and h.categories["org"] == ["it", "finance"]
    x = np.array([0.3, -0.2, 0.9, 0.1], dtype="<f2")
    with torch.no_grad():
        ref = torch.softmax(lin2(torch.nn.functional.gelu(lin1(torch.tensor(x.astype(np.float32))), approximate="tanh")), dim=-1).tolist()
    cat, margin = h("org", x.tobytes())
    assert cat == ["it", "finance"][int(np.argmax(ref))] and abs(margin - abs(ref[0] - ref[1])) < 1e-3


def test_mlp_head_shape_and_type_errors(tmp_path):
    _write_mlp_head(tmp_path)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["head_type"] = "cnn"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="head_type"):
        HeadCategorizer.load(str(tmp_path))
    _write_mlp_head(tmp_path)
    w = pytest.importorskip("safetensors.numpy").load_file(str(tmp_path / CATEGORY_HEAD_WEIGHTS))
    w["org.w2"] = np.zeros((3, 3), dtype=np.float32)                       # 出力行数がカテゴリ数と違う
    save_file(w, str(tmp_path / CATEGORY_HEAD_WEIGHTS))
    with pytest.raises(ConfigurationError, match="shape"):
        HeadCategorizer.load(str(tmp_path))
    _write_mlp_head(tmp_path)
    (tmp_path / CATEGORY_HEAD_WEIGHTS).write_bytes(b"")
    with pytest.raises(ConfigurationError):
        HeadCategorizer.load(str(tmp_path))


def test_mlp_head_bad_dimension_falls_back_to_other(tmp_path):
    _write_mlp_head(tmp_path)
    h = HeadCategorizer.load(str(tmp_path))
    assert h("org", _vec(1, 0)) == ("service", 0.0) and h("org", b"") == ("service", 0.0)


def test_load_head_dir_reads_head_files_elsewhere_but_checks_the_model_encoder(tmp_path):
    model = tmp_path / "model"; cand = tmp_path / "cand"; model.mkdir(); cand.mkdir()
    (model / "model.safetensors").write_bytes(b"encoder-bytes")
    _write_head(cand, sha=encoder_sha256(str(model)))                       # cand には別の model.safetensors が書かれるが無視される
    h = HeadCategorizer.load(str(model), head_dir=str(cand))
    assert h("org", _vec(1, 0, 0, 0))[0] == "it"
    (model / "model.safetensors").write_bytes(b"other-encoder")
    with pytest.raises(ConfigurationError, match="encoder"):
        HeadCategorizer.load(str(model), head_dir=str(cand))


def test_input_dim_matches_hidden_size_times_features_when_config_json_present(tmp_path):
    # 実モデルフォルダには encoder の config.json（hidden_size）が必ずある。Task 4 のレビュー（設計 §2.1）:
    # 「HeadCategorizer.load は features と重みの入力次元が合うことを検査し、合わなければ ConfigurationError」。
    # span_mean は hidden_size × 1、span_ctx は × 3 で厳密に照合する。
    _write_head(tmp_path, dim=4)
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
    assert HeadCategorizer.load(str(tmp_path)).features == "span_mean"           # 4 == hidden_size(4) * 1 → OK

    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="features"):                    # 4 != hidden_size(4) * 3 == 12
        HeadCategorizer.load(str(tmp_path))

    _write_head(tmp_path, dim=12)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    assert HeadCategorizer.load(str(tmp_path)).features == "span_ctx"            # 12 == hidden_size(4) * 3 → OK


def test_input_dim_strict_hidden_size_check_is_not_just_divisibility_by_3(tmp_path):
    # config.json の hidden_size が分かるときは厳密に照合する。12 列は 3 の倍数なので、弱い検査（3 の倍数か）
    # だけなら通ってしまうが、span_mean は hidden_size(4) × 1 = 4 列でなければならない
    _write_head(tmp_path, dim=12)
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_mean"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="hidden_size 4"):
        HeadCategorizer.load(str(tmp_path))
    # span_ctx でも同じ: 6 列は 3 の倍数だが hidden_size(4) × 3 = 12 ではない
    _write_head(tmp_path, dim=6)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="hidden_size 4"):
        HeadCategorizer.load(str(tmp_path))


def test_input_dim_only_checked_for_divisibility_by_3_without_config_json(tmp_path):
    # encoder の config.json（HF config）が無いフィクスチャ（既存の _write_head はこれを書かない。だから
    # 既存のテストはこの検査の追加で壊れない）では hidden_size と厳密には照合できない。span_ctx は 3 区画の
    # 連結のはずなので、せめて 3 の倍数であることだけを要求する。
    _write_head(tmp_path, dim=4)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="divisible by 3"):
        HeadCategorizer.load(str(tmp_path))

    _write_head(tmp_path, dim=6)
    cfg = json.loads((tmp_path / CATEGORY_HEAD_CONFIG).read_text(encoding="utf-8"))
    cfg["features"] = "span_ctx"
    (tmp_path / CATEGORY_HEAD_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    assert HeadCategorizer.load(str(tmp_path)).features == "span_ctx"
