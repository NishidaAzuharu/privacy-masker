"""本番モデルの学習ノートブック（`train_ja_mask_charhead_v11_scratch.ipynb`）を読み、第 2 段の多タスク版
（train_ja_mask_charhead_v12_multitask.ipynb）を文字列置換で生成する。元ノートが変わって置換元が見つからなければ
AssertionError で止まる（黙って一部だけ書き換えない）。

多タスク版は NER の損失に補助カテゴリ損失 ``CAT_LAMBDA × CE`` を足して encoder を学習する
（設計は開発記録、非公開）。補助層は学習にだけ使い、
本番のカテゴリヘッドは tools/train_category_head.py で新 encoder に対して作り直す。

usage:
  PYTHONPATH=. .venv/bin/python tools/make_multitask_notebook.py            # データフォルダと notebooks_dir に書く
  PYTHONPATH=. .venv/bin/python tools/make_multitask_notebook.py --out PATH [--out PATH2 ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402

SOURCE_NAME = "train_ja_mask_charhead_v11_scratch.ipynb"
NOTEBOOK_NAME = "train_ja_mask_charhead_v12_multitask.ipynb"
HEADER_MARK = "v12 多タスク版"

# --- 規則 1: 先頭の markdown -------------------------------------------------
HEADER_NOTE = (f"> **{HEADER_MARK}**（`tools/make_multitask_notebook.py` が元ノートから生成。手で編集しない）。"
               "NER の損失にカテゴリ損失 `0.3 × CE` を足して encoder を学習する。"
               "設計は開発記録（非公開）。\n>\n"
               "> 既定のデータは `plus_old`（`DATA_VARIANT_DEFAULT`）、出力先は "
               "`outputs_ruri-v3-30m-charhead-v11-plus-old-frompre-multitask/`。以下の説明は元ノートのまま（履歴）。\n\n")

# --- 規則 2: 設定セル ---------------------------------------------------------
OUTPUT_DIR_LINE = 'OUTPUT_DIR = os.path.join(_OUT_ROOT, f"outputs_{MODEL_TAG}-{RUN_TAG}")'
CONFIG_TAIL = '''# --- 第 2 段: 補助カテゴリ損失 ---------------------------------------------
CAT_LAMBDA = 0.3                      # loss = ner + CAT_LAMBDA * cat（設計 v2 §2.3）
CAT_LR = 1e-3                         # 補助層の lr（文字ヘッドと同じ）
CAT_LABELS_PATH = _d("category_labels_v1/category_labels_v1.jsonl")
CAT_TRAIN_V11 = _d("data_v11/train_v11.jsonl")   # doc_id "train:<行>" → 本文
print("category aux :", f"lambda={CAT_LAMBDA} lr={CAT_LR}", CAT_LABELS_PATH)
'''

# --- 規則 3: char_head 読み込みセル -------------------------------------------
CH_ASSERT = '''# 第 2 段で使う関数がデータフォルダの char_head.py にあるか（古いまま学習を始めない）
for _n in ("category_loss", "CategoryAuxHead", "span_ctx_batch", "category_spans_by_text", "sqrt_class_weights"):
    assert hasattr(ch, _n), f"データフォルダの char_head.py が古い: {_n} が無い（cp privacy_masker/char_head.py <データフォルダ>/char_head.py）"
'''

# --- 規則 4: データ読み込みセルの直後に挿入する結合セル -------------------------
LOAD_SPLIT_MARK = "all_train, all_valid, SPLIT_MANIFEST = load_split("
JOIN_MARKDOWN = "## 3b. カテゴリ教師の結合（第 2 段）\n"
JOIN_CELL = '''# 第 2 段: カテゴリ教師を本文で結合する（学習レコードは dedup・shuffle されるので doc_id ではなく本文で引く）
assert not SMOKE_LIMIT, "SMOKE_LIMIT は結合元も切り詰めるので 3b は本番設定（SMOKE_LIMIT=0）でだけ動く"
from collections import Counter as _Counter
_cat_rows = read_jsonl(CAT_LABELS_PATH)
_text_by_doc = {f"train:{i}": r["text"] for i, r in enumerate(read_jsonl(CAT_TRAIN_V11))}
with open(CAT_TRAIN_V11, "rb") as _f:
    CAT_TRAIN_V11_SHA = hashlib.sha256(_f.read()).hexdigest()
# 本文の取り違えを防ぐ: ラベル行の surface_sha（生の表層の sha256 先頭 16 桁）を、結合先の本文の同じ位置と照合する
_n_bad_sha = sum(1 for r in _cat_rows if r["split"] in ("head_train", "head_dev") and r["doc_id"] in _text_by_doc
                 and hashlib.sha256(_text_by_doc[r["doc_id"]][r["start"]:r["end"]].encode("utf-8")).hexdigest()[:16] != r["surface_sha"])
CAT_BY_TEXT, CAT_STATS = ch.category_spans_by_text(_cat_rows, _text_by_doc)     # head_train + head_dev だけ
assert CAT_STATS["n_rows"] > 0 and CAT_STATS["n_used"] == CAT_STATS["n_rows"], f"カテゴリ教師が空か結合できていない: {CAT_STATS}"
CAT_STATS["n_surface_sha_mismatch"] = _n_bad_sha
assert _n_bad_sha == 0, f"ラベル行の surface_sha が本文と合わない（train_v11 が作り直された可能性）: {CAT_STATS}"
# 同じ本文を持つ train:<行> が複数あると同じスパンが重複して入るので、本文ごとに重複を落として数える
_n_before = sum(len(v) for v in CAT_BY_TEXT.values())
CAT_BY_TEXT = {t: sorted(set(v)) for t, v in CAT_BY_TEXT.items()}
CAT_STATS["n_duplicate_spans_dropped"] = _n_before - sum(len(v) for v in CAT_BY_TEXT.values())
_train_texts = {r["text"] for r in all_train}
_valid_texts = {r["text"] for r in all_valid}
CAT_STATS["n_texts"] = len(CAT_BY_TEXT)
CAT_STATS["n_texts_in_train"] = sum(1 for t in CAT_BY_TEXT if t in _train_texts)
CAT_STATS["n_texts_in_valid"] = sum(1 for t in CAT_BY_TEXT if t in _valid_texts)
_counts = {k: _Counter() for k in ch.CATEGORY_KINDS}
for _spans in CAT_BY_TEXT.values():
    for _s, _e, _k, _c in _spans:
        _kind = ch.CATEGORY_KINDS[_k]
        _counts[_kind][ch.CATEGORY_NAMES[_kind][_c]] += 1
CAT_CLASS_WEIGHTS = {k: torch.tensor(ch.sqrt_class_weights(_counts[k], ch.CATEGORY_NAMES[k]), dtype=torch.float) for k in ch.CATEGORY_KINDS}
with open(CAT_LABELS_PATH, "rb") as _f:
    CAT_LABELS_SHA = hashlib.sha256(_f.read()).hexdigest()
print(json.dumps({"stats": CAT_STATS, "class_weights": {k: [round(x, 3) for x in v.tolist()] for k, v in CAT_CLASS_WEIGHTS.items()}}, ensure_ascii=False, indent=1))
assert CAT_STATS["n_unjoined"] == 0, f"doc_id が train_v11 に無い行がある: {CAT_STATS}"
assert CAT_STATS["n_texts_in_valid"] == 0, "カテゴリ教師が valid の本文に付いている（judge の行が混ざった、または train/valid が交差）"
assert CAT_STATS["n_texts_in_train"] == CAT_STATS["n_texts"], f"学習レコードに無い本文がある: {CAT_STATS}"
'''

# --- 規則 5: build_samples セル -----------------------------------------------
BUILD_SAMPLES_MARK = 'def build_samples(records, tok, tag=""):'
BUILD_SAMPLES_COUNTERS_OLD = "samples, n_split, n_cut_ent, n_uncov, n_flat = [], 0, 0, 0, 0"
BUILD_SAMPLES_COUNTERS_NEW = "samples, n_split, n_cut_ent, n_uncov, n_flat, n_cat, n_cat_cut = [], 0, 0, 0, 0, 0, 0"
BUILD_SAMPLES_ENCODE_OLD = "            f = ch.encode_chunk(piece, tok, local, piece_cache=piece_cache, strict=False)"
BUILD_SAMPLES_ENCODE_NEW = '''            _cats = CAT_BY_TEXT.get(text, [])
            local_cats = [(s - lo, e - lo, k, c) for s, e, k, c in _cats if s >= lo and e <= hi]
            n_cat += len(local_cats)
            n_cat_cut += sum(1 for s, e, _k, _c in _cats if not (s >= lo and e <= hi) and s < hi and e > lo)
            f = ch.encode_chunk(piece, tok, local, piece_cache=piece_cache, strict=False, category_spans=local_cats)'''
BUILD_SAMPLES_PRINT_OLD = 'f"未被覆文字 {n_uncov} / 平坦化で統合・除去 {n_flat})")'
BUILD_SAMPLES_PRINT_NEW = 'f"未被覆文字 {n_uncov} / 平坦化で統合・除去 {n_flat} / カテゴリスパン {n_cat}（境界で切れて落とした {n_cat_cut}）)")'

# --- 規則 6: モデルセル -------------------------------------------------------
MODEL_MARK = "model = ch.CharBioNerModel.from_pretrained_encoder("
MODEL_TAIL = '''# 第 2 段: 補助カテゴリ層（学習にだけ使う。推論の保存形式には入らない）
aux = ch.CategoryAuxHead(model.config.hidden_size).to(device)
print("category aux head:", sum(p.numel() for p in aux.parameters()), "params")
'''

# --- 規則 7: 学習ループセル ---------------------------------------------------
TRAIN_MARK = "def train_one_epoch("
TRAIN_HEAD = '''# 第 2 段: 補助層の保存と、カテゴリの class weight を device に載せる
from safetensors.torch import save_file as _save_st
CAT_CLASS_WEIGHTS_DEV = {k: v.to(device) for k, v in CAT_CLASS_WEIGHTS.items()}
'''
GROUP_SPEC_OLD = '    ("head/no_decay",    _split(no_decay, True),  LR_HEAD,    0.0),\n]'
GROUP_SPEC_NEW = ('    ("head/no_decay",    _split(no_decay, True),  LR_HEAD,    0.0),\n'
                  '    ("aux",              list(aux.parameters()),  CAT_LR,     WEIGHT_DECAY),   # 第 2 段の補助カテゴリ層\n'
                  ']')
TRAIN_MODE_OLD = "    model.train()\n    if encoder_eval:"
TRAIN_MODE_NEW = "    model.train(); aux.train()\n    if encoder_eval:"
RUN_LOSS_INIT_OLD = "    run_loss, n_loss_batches = 0.0, 0"
RUN_LOSS_INIT_NEW = "    run_loss, n_loss_batches, run_cat, n_cat_total = 0.0, 0, 0.0, 0"
LOSS_OLD = '            loss = model(**batch)["loss"]'
LOSS_NEW = '''            out = model(**batch)
            cat_loss, n_cat_spans = ch.category_loss(aux, out["token_hidden"], batch, class_weights=CAT_CLASS_WEIGHTS_DEV)
            loss = out["loss"] + CAT_LAMBDA * cat_loss'''
CLIP_OLD = "clip_grad_norm_(model.parameters(),"
CLIP_NEW = "clip_grad_norm_(list(model.parameters()) + list(aux.parameters()),"
RUN_LOSS_ACC_OLD = "        run_loss += loss.item(); n_loss_batches += 1"
RUN_LOSS_ACC_NEW = ("        run_loss += loss.item(); n_loss_batches += 1; "
                    "run_cat += float(cat_loss.item()); n_cat_total += n_cat_spans")
RETURN_OLD = '    return {"train_loss": run_loss / max(n_loss_batches, 1),'
RETURN_NEW = ('    return {"train_loss": run_loss / max(n_loss_batches, 1), '
              '"cat_loss": run_cat / max(n_loss_batches, 1), "n_cat_spans": n_cat_total,')
EPOCH_LOG_OLD = r'''print(f"\n[epoch {epoch}] train_loss={st['train_loss']:.4f} ({st['train_seconds']:.0f}s) "'''
EPOCH_LOG_NEW = EPOCH_LOG_OLD + '''\n          f"cat_loss={st['cat_loss']:.4f} ({st['n_cat_spans']} spans) "'''
EPOCH_SAVE_OLD = '        print(f"  [save] {_ep_dir}")'
EPOCH_SAVE_NEW = ('        _save_st({k: v.detach().cpu().contiguous() for k, v in aux.state_dict().items()}, '
                  'os.path.join(_ep_dir, "category_aux.safetensors"))\n' + EPOCH_SAVE_OLD)

# --- 規則 8: 保存セル ---------------------------------------------------------
MANIFEST_OPEN = 'with open(os.path.join(SAVE_DIR, "training_manifest.json"), "w", encoding="utf-8") as f:'
MANIFEST_BLOCK = '''manifest["category_aux"] = {
    "lambda": CAT_LAMBDA, "lr": CAT_LR, "labels_path": CAT_LABELS_PATH, "labels_sha256": CAT_LABELS_SHA,
    "train_v11_path": CAT_TRAIN_V11, "train_v11_sha256": CAT_TRAIN_V11_SHA,
    "stats": CAT_STATS, "kinds": list(ch.CATEGORY_KINDS), "names": {k: list(v) for k, v in ch.CATEGORY_NAMES.items()},
    "class_weights": {k: v.tolist() for k, v in CAT_CLASS_WEIGHTS.items()},
    "note": "補助層は学習にだけ使う。本番のヘッドは tools/train_category_head.py で新 encoder に対して作り直す",
    "aux_weights_epoch": history[-1]["epoch"],
    "note_best": "best/category_aux.safetensors は最後のエポックの補助層（best のものではない。学習専用で読む側は無い）",
}
_save_st({k: v.detach().cpu().contiguous() for k, v in aux.state_dict().items()}, os.path.join(SAVE_DIR, "category_aux.safetensors"))
'''

# --- 規則 9: 実行後手順の markdown（最後のセル） --------------------------------
AFTER_NOTE = ("6. 第 2 段: `tools/average_checkpoints.py --out <avg_dir> <epoch_dir> ...` → NER 評価（4 セット + 盲検セット）"
              "→ 合格なら `category_head_data.py --judge-split` と `train_category_head.py --model <avg_dir>`\n")


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _set(cell: dict, src: str) -> None:
    cell["source"] = src.splitlines(keepends=True)


def _replace(src: str, old: str, new: str, count: int = 1) -> str:
    """``old`` がちょうど ``count`` 回あるときだけ置換する（元ノートの変化に気づくため）。"""
    n = src.count(old)
    assert n == count, f"replacement source found {n} times (expected {count}): {old[:60]!r}"
    return src.replace(old, new)


def _find(cells: list[dict], marker: str) -> int:
    """``marker`` を含むコードセルがちょうど 1 つならその位置を返す。"""
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and marker in _src(c)]
    assert len(hits) == 1, f"expected exactly one code cell containing {marker!r}, found {len(hits)}"
    return hits[0]


def _append(src: str, tail: str, sep: str = "\n\n") -> str:
    """セル末尾に ``tail`` を足す。元のセルが改行で終わっていなくても行が繋がらないよう ``sep`` を挟む。"""
    return src.rstrip("\n") + sep + tail.strip("\n")


def _code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def _markdown_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def transform(nb: dict) -> dict:
    """v11 ノートブック JSON から多タスク版の JSON を作る（入力は変更しない）。"""
    nb = copy.deepcopy(nb)
    cells = nb["cells"]

    # 1. 先頭の markdown
    assert cells and cells[0]["cell_type"] == "markdown", "first cell must be markdown"
    assert HEADER_MARK not in _src(cells[0]), "notebook already transformed"
    _set(cells[0], HEADER_NOTE + _src(cells[0]))

    # 2. 設定セル
    i = _find(cells, 'DATA_VARIANT_DEFAULT = "v11"')
    s = _src(cells[i])
    s = _replace(s, 'DATA_VARIANT_DEFAULT = "v11"', 'DATA_VARIANT_DEFAULT = "plus_old"')
    s = _replace(s, OUTPUT_DIR_LINE, 'RUN_TAG = RUN_TAG + "-multitask"\n' + OUTPUT_DIR_LINE)
    _set(cells[i], _append(s, CONFIG_TAIL))

    # 3. char_head 読み込みセル
    i = _find(cells, "_spec.loader.exec_module(ch)")
    _set(cells[i], _append(_src(cells[i]), CH_ASSERT))

    # 4. データ読み込みセルの直後にカテゴリ教師の結合
    i = _find(cells, LOAD_SPLIT_MARK)
    cells.insert(i + 1, _markdown_cell(JOIN_MARKDOWN))
    cells.insert(i + 2, _code_cell(JOIN_CELL))

    # 5. build_samples
    i = _find(cells, BUILD_SAMPLES_MARK)
    s = _src(cells[i])
    s = _replace(s, BUILD_SAMPLES_COUNTERS_OLD, BUILD_SAMPLES_COUNTERS_NEW)
    s = _replace(s, BUILD_SAMPLES_ENCODE_OLD, BUILD_SAMPLES_ENCODE_NEW)
    s = _replace(s, BUILD_SAMPLES_PRINT_OLD, BUILD_SAMPLES_PRINT_NEW)
    _set(cells[i], s)

    # 6. モデルセル（device はこのセルより前で定義されていること）
    i = _find(cells, MODEL_MARK)
    assert any("device =" in _src(c) for c in cells[:i] if c["cell_type"] == "code"), \
        "device must be defined in a code cell before the model cell"
    _set(cells[i], _append(_src(cells[i]), MODEL_TAIL))

    # 7. 学習ループ
    i = _find(cells, TRAIN_MARK)
    s = TRAIN_HEAD + _src(cells[i])
    s = _replace(s, GROUP_SPEC_OLD, GROUP_SPEC_NEW)
    s = _replace(s, TRAIN_MODE_OLD, TRAIN_MODE_NEW)
    s = _replace(s, RUN_LOSS_INIT_OLD, RUN_LOSS_INIT_NEW)
    s = _replace(s, LOSS_OLD, LOSS_NEW)
    s = _replace(s, CLIP_OLD, CLIP_NEW, count=2)
    s = _replace(s, RUN_LOSS_ACC_OLD, RUN_LOSS_ACC_NEW)
    s = _replace(s, RETURN_OLD, RETURN_NEW)
    s = _replace(s, EPOCH_LOG_OLD, EPOCH_LOG_NEW)
    s = _replace(s, EPOCH_SAVE_OLD, EPOCH_SAVE_NEW)
    _set(cells[i], s)

    # 8. 保存セル（"training_manifest.json" だけでは 3 節の check_encoder_history にも当たるので open 行で探す）
    i = _find(cells, MANIFEST_OPEN)
    s = _src(cells[i])
    s = _replace(s, MANIFEST_OPEN, MANIFEST_BLOCK + MANIFEST_OPEN)
    _set(cells[i], s)

    # 9. 実行後手順
    last = cells[-1]
    assert last["cell_type"] == "markdown", "last cell must be the post-training markdown"
    _set(last, _append(_src(last), AFTER_NOTE, sep="\n"))
    return nb


def write_notebook(paths: list[str]) -> None:
    src = os.path.join(notebooks_dir(), SOURCE_NAME)
    with open(src, encoding="utf-8") as f:
        nb = transform(json.load(f))
    for path in paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=None, help="出力先（繰り返し可。既定はデータフォルダと notebooks_dir）")
    args = ap.parse_args(argv)
    write_notebook(args.out or [os.path.join(drive_dir(), NOTEBOOK_NAME), os.path.join(notebooks_dir(), NOTEBOOK_NAME)])


if __name__ == "__main__":
    main()
