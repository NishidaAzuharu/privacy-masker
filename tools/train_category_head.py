"""分類ヘッド（計画 B の v2、設計 §8。設計書は開発記録、非公開）の学習ツール。

本番 NER encoder（凍結）で `tools/category_head_data.py` が作ったラベル付きスパン（`spans_train.jsonl` /
`spans_dev.jsonl`、`--judge` のときだけ `spans_judge.jsonl`）のベクトルを `--features`（`span_mean` =
スパンの平均、`span_ctx` = スパン + 左右の文脈の平均を連結）で計算し、種別（org / facility / location）ごとに
`--head` のヘッドを学習する（`linear` = `nn.Linear` 1 層 + softmax、`mlp` = Linear → GELU(tanh) → Dropout →
Linear + softmax）。クラス重みは既定で頻度の平方根の逆数（`--class-weight sqrt`、平均 1 に正規化して上限 5）。

選択と判定を分ける（v2 設計 §3.1）:

  * 選択（mlp の早期終了、min_margin、統合候補）は **dev だけ**で行う。min_margin は dev の種別語の無い
    スパン（`no_type_word=True`: Qwen ラベル + 種別語削除写し）で precision ≥ `--target-precision`
    （既定 0.95）になる最小の margin。
  * judge は `--judge` を付けたときだけ読み（付けなければファイルの存在も見ない）、kind ごとに 1 回の判定に
    使う（v2 設計 §3.2）: (i) judge の規則ラベル付きスパン（種別語あり）の macro accuracy ≥ 0.90、かつ
    (ii) judge の種別語の無いスパンで、dev で選んだ min_margin における precision ≥ `--target-precision`。
    両方を満たした kind が `accepted: true`。`--judge` 無しの実行は `judge: null` / `accepted: null`（未判定）。
  * `accepted` は実文書での確認の前の必要条件にすぎない（計画 B では ORG が valid で受け入れ条件を満たしたが、
    実文書の precision が 0.741 で不採用にした。開発記録、非公開）。

`category_head.safetensors` と `category_head.json`（version 2。`head_type` / `features` / `hidden` を宣言する）は
既定で**候補**として `<--spans>/candidate` に書く（モデルフォルダには書かない。検出器は `category_head="auto"` で
モデルフォルダの `category_head.json` を自動で読むので、学習し直すだけで黙って採用されないようにするため）。
実文書で確かめたうえでモデルフォルダに入れるときだけ `--install` を付ける。`--install` は `--judge` が必須で、
`accepted` の kind だけをモデルフォルダに書く（1 つも無ければヘッドのファイルは書かない）。

usage:
  PYTHONPATH=. .venv/bin/python tools/train_category_head.py \\
      --model <NER モデルのフォルダ（model.safetensors・tokenizer を含む）> \\
      --spans <tools/category_head_data.py --out のディレクトリ> \\
      [--features span_mean|span_ctx] [--head linear|mlp] [--hidden 128] [--dropout 0.2] \\
      [--class-weight sqrt|inv] [--lr LR] [--patience 5] [--judge] \\
      [--out <書き出し先、既定は <--spans>/candidate> | --install] [--report <Markdown のパス>] \\
      [--merge kind:from=to ...] [--kinds org,...] [--cache-dir DIR] [--force] [--limit N]

候補（`--install` 無し）に保存するのは min_margin が見つかった kind（`--kinds` を指定すればさらにその中に
限定）。`--out` にモデルフォルダは指定できない（`--install` を使う）。train manifest
（`category_head_train_manifest.json`: encoder / data の SHA-256・学習の設定・全 kind の指標）は書き出し先の隣に
毎回書く。ベクトルキャッシュ（`vectors_{split}_{features}.npz`）は既定で `--spans` のディレクトリに置く
（モデルフォルダに置かない）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import torch
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from privacy_masker.category_head import (  # noqa: E402
    CATEGORY_HEAD_CONFIG,
    CATEGORY_HEAD_WEIGHTS,
    FEATURES,
    HEAD_TYPES,
    _linear_softmax,
    _mlp_softmax,
    encoder_sha256,
)
from privacy_masker.char_head import encode_span_vectors, load_char_model, vector_to_floats  # noqa: E402
from privacy_masker.operators.semantic import CATEGORIES  # noqa: E402

# 0.05, 0.10, ..., 0.60
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 13))
# 受け入れ条件 (i): judge の規則ラベル付きスパン（種別語あり）の macro accuracy の下限
ACCEPT_MACRO_ACCURACY = 0.90
# 受け入れ条件 (ii) の precision 目標。--install はこの値未満の --target-precision を受け付けない
# （緩めた目標で選んだ min_margin を「受け入れ」扱いにして本番へ書けてしまうのを防ぐ）
ACCEPT_TARGET_PRECISION = 0.95
# クラス重み: sqrt = 頻度の平方根の逆数（既定）、inv = 頻度の逆数（v1）。平均 1 に正規化してから上限で切る
CLASS_WEIGHT_MODES: tuple[str, ...] = ("sqrt", "inv")
CLASS_WEIGHT_CAP = 5.0
# --lr を省略したときの学習率（ヘッドの種類ごと）
DEFAULT_LR: dict[str, float] = {"linear": 0.05, "mlp": 1e-3}
# mlp の早期終了で dev の macro accuracy を見る間隔（epoch）
EARLY_STOP_EVERY = 10
# --out を省略したときの書き出し先（--spans の下）
CANDIDATE_DIRNAME = "candidate"
TRAIN_MANIFEST = "category_head_train_manifest.json"


def is_accepted(
    judge_macro: float | None, judge_precision: float | None, min_margin: float | None, target: float
) -> bool:
    """v2 設計 §3.2 の 2 条件（どちらも judge で見る）。

    (i) judge の規則ラベル付きスパン（種別語あり）の macro accuracy ≥ 0.90、かつ (ii) judge の種別語の無い
    スパンで、dev で選んだ `min_margin` における precision ≥ `target`。`min_margin` が無い（dev で target に
    届かない）・judge の値が未定義（該当スパンが 0 件、または margin 以上の予測が 0 件）なら False。
    """
    return (min_margin is not None and judge_macro is not None and judge_precision is not None
            and judge_macro >= ACCEPT_MACRO_ACCURACY and judge_precision >= target)


# ---------------------------------------------------------------------------
# 学習・評価
# ---------------------------------------------------------------------------

def _class_weights(
    y: Sequence[str], categories: Sequence[str], mode: str = "sqrt", cap: float = CLASS_WEIGHT_CAP
) -> list[float]:
    """`CrossEntropyLoss` の `weight`: クラス頻度の平方根の逆数（`sqrt`、既定）か逆数（`inv`）。

    正規化は `categories` の**全カテゴリ数 k（訓練データに 1 件も無いクラスも数える）**で割る:
    `w_c * k / Σw`（全カテゴリでの平均が 1）。そのあと各重みを `cap` で切る（上限にかかったときだけ、
    切った分だけ平均は 1 より下になる）。1 つのクラスの重みは最大でも k なので、上限 5 がかかるのは
    k > 5 の kind（facility / location は 8、org は 14）だけ。`inv` は、上限 5 を超える重みが無い限り
    v1 の重み（逆数を同じく k で正規化、上限なし）と一致する。
    訓練データに 1 件も無いクラスは重み 0（そのクラスに向けた勾配は出ない。学習データを増やす以外に
    直しようが無いので、ここでは黙って 0 にする）。
    """
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"unknown class weight mode {mode!r} (use one of {CLASS_WEIGHT_MODES})")
    counts = Counter(y)
    inv = [
        ((1.0 / counts[c]) if mode == "inv" else (1.0 / math.sqrt(counts[c]))) if counts.get(c, 0) > 0 else 0.0
        for c in categories
    ]
    total = sum(inv)
    k = len(categories)
    if total <= 0:
        return [1.0] * k
    return [min(w * k / total, cap) for w in inv]


def _build_model(head: str, dim: int, k: int, hidden: int, dropout: float) -> torch.nn.Module:
    """linear: `nn.Linear(dim, k)`。mlp: Linear(dim, hidden) → GELU(tanh) → Dropout → Linear(hidden, k)
    （推論側 `category_head._mlp_softmax` と同じ式。GELU は tanh 近似で揃える）。"""
    if head == "linear":
        return torch.nn.Linear(dim, k)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, k),
    )


def _export(head: str, model: torch.nn.Module) -> tuple:
    """学習済みの torch モデルを推論側の形にする。linear は `(W, b)` の Python list（推論は純 Python）、
    mlp は `(W1, b1, W2, b2)` の numpy float32（`astype` は常に複製するので torch の重みとは共有しない）。"""
    if head == "linear":
        return model.weight.detach().tolist(), model.bias.detach().tolist()
    l1, l2 = model[0], model[3]
    return (
        l1.weight.detach().numpy().astype(np.float32),
        l1.bias.detach().numpy().astype(np.float32),
        l2.weight.detach().numpy().astype(np.float32),
        l2.bias.detach().numpy().astype(np.float32),
    )


def predict_probs(head_type: str, head: tuple, X: Sequence[Sequence[float]]) -> list[list[float]]:
    """推論側と同じ関数（linear は `_linear_softmax`、mlp は `_mlp_softmax`）で各行の確率を計算する。"""
    if head_type not in HEAD_TYPES:
        raise ValueError(f"unknown head type {head_type!r} (use one of {HEAD_TYPES})")
    if head_type == "linear":
        W, b = head
        return [_linear_softmax(W, b, x) for x in X]
    W1, b1, W2, b2 = head
    return [_mlp_softmax(W1, b1, W2, b2, x) for x in X]


def train_heads(
    X: dict[str, list[list[float]]],
    y: dict[str, list[str]],
    categories: dict[str, list[str]],
    *,
    head: str = "linear",
    hidden: int = 128,
    dropout: float = 0.2,
    epochs: int = 300,
    lr: float | None = None,
    weight_decay: float = 1e-4,
    seed: int = 0,
    class_weight: str = "sqrt",
    X_dev: dict[str, list[list[float]]] | None = None,
    y_dev: dict[str, list[str]] | None = None,
    patience: int = 5,
) -> dict[str, tuple]:
    """kind ごとにヘッドを full-batch の Adam で学習する（`lr` 省略時は linear 0.05 / mlp 1e-3）。

    linear は `epochs` 回まわして最後の重みを返す（従来どおり。`X_dev` は使わない）。mlp は
    `EARLY_STOP_EVERY` epoch ごとに dev（`X_dev[kind]` / `y_dev[kind]`）の macro accuracy を見て、`patience` 回
    続けて改善しなければ止め、最良だった時点の重みを返す（dev が無い kind は最後の重み）。macro accuracy が
    同点のときは dev の平均 NLL が下がれば改善とみなす（accuracy は早く頭打ちになりやすく、厳密な改善だけを
    数えると最初に同点へ届いた学習の浅い重みで止まってしまうため）。

    返り値は linear が `(W, b)` の Python list（推論側の `_linear_softmax` は純 Python なので torch に
    依存させない）、mlp が `(W1, b1, W2, b2)` の numpy float32。`torch.manual_seed(seed)` は kind ごとに
    呼び直すので、複数 kind を学習する順序に依存しない（mlp の dropout も同じ乱数列を使う）。
    """
    if head not in HEAD_TYPES:
        raise ValueError(f"unknown head {head!r} (use one of {HEAD_TYPES})")
    lr = lr if lr is not None else DEFAULT_LR[head]
    heads: dict[str, tuple] = {}
    for kind, xs in X.items():
        labels = y[kind]
        cats = list(categories[kind])
        if not xs:
            raise ValueError(f"no training vectors for kind {kind!r}")
        if len(xs) != len(labels):
            raise ValueError(f"X and y length mismatch for kind {kind!r}")
        unknown = sorted({lab for lab in labels if lab not in cats})
        if unknown:
            raise ValueError(f"labels {unknown!r} for kind {kind!r} are not in categories {cats!r}")

        idx = {c: i for i, c in enumerate(cats)}
        torch.manual_seed(seed)
        model = _build_model(head, len(xs[0]), len(cats), hidden, dropout)
        x_t = torch.tensor(xs, dtype=torch.float32)
        y_t = torch.tensor([idx[lab] for lab in labels], dtype=torch.long)
        weight = torch.tensor(_class_weights(labels, cats, class_weight), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        dev_x = X_dev.get(kind) if (head == "mlp" and X_dev is not None) else None
        dev_y = y_dev.get(kind) if (dev_x and y_dev is not None) else None
        best_state: dict[str, torch.Tensor] | None = None
        best_score, best_nll = -1.0, math.inf
        bad = 0
        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_t), y_t)
            loss.backward()
            optimizer.step()
            if dev_x and dev_y is not None and ep % EARLY_STOP_EVERY == 0:
                model.eval()
                probs = predict_probs(head, _export(head, model), dev_x)
                score = evaluate_probs(probs, dev_y, cats)["macro_accuracy"]
                nll = _mean_nll(probs, dev_y, cats)
                # 主は dev の macro accuracy。同点なら dev の平均 NLL が下がったときも改善とみなす（accuracy が
                # 早々に頭打ちになる dev で、学習の浅い最初の同点の重みに固定されないように）
                if score > best_score + 1e-9 or (score >= best_score - 1e-9 and nll < best_nll - 1e-9):
                    best_score, best_nll, bad = max(score, best_score), nll, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        heads[kind] = _export(head, model)
    return heads


def _mean_nll(probs: Sequence[Sequence[float]], y: Sequence[str], categories: Sequence[str]) -> float:
    """正解クラスの負の対数尤度の平均（mlp の早期終了で macro accuracy が同点のときの比較に使う）。"""
    idx = {c: i for i, c in enumerate(categories)}
    if not probs:
        return math.inf
    return sum(-math.log(max(p[idx[yi]], 1e-12)) for p, yi in zip(probs, y)) / len(probs)


def evaluate_probs(probs: Sequence[Sequence[float]], y: Sequence[str], categories: Sequence[str]) -> dict:
    """確率（`predict_probs` の出力）→ argmax で評価する。`macro_accuracy` はクラスごとの recall の平均
    （**評価データに 1 件以上あるクラスだけ**で平均する）、`accuracy` は全体の正解率、`per_class` は
    {ラベル: recall}（評価データに無いクラスは None = 未定義）、`support` は {ラベル: 評価データの件数}。

    評価データに無いクラスを recall 0.0 として平均に入れない理由: 規則ラベルは構成上 OTHER
    （org の service、facility / location の other）を持たない（`category_head_data.build_spans`
    は OTHER のスパンを捨てる）ので、それを 0 と数えると (i) の上限が (k-1)/k に張り付き、
    facility（8 クラス）は 7/8=0.875 で設計 §8 の 0.90 に構造的に届かない。
    """
    cats = list(categories)
    if len(probs) != len(y):
        raise ValueError("probs and y length mismatch")
    hits = {c: 0 for c in cats}
    totals = {c: 0 for c in cats}
    correct = 0
    for p, yi in zip(probs, y):
        if yi not in totals:
            raise ValueError(f"label {yi!r} is not in categories {cats!r}")
        pred = cats[max(range(len(p)), key=p.__getitem__)]
        totals[yi] += 1
        if pred == yi:
            correct += 1
            hits[yi] += 1
    per_class: dict[str, float | None] = {c: (hits[c] / totals[c] if totals[c] else None) for c in cats}
    supported = [r for r in per_class.values() if r is not None]
    macro_accuracy = sum(supported) / len(supported) if supported else 0.0
    accuracy = correct / len(probs) if len(probs) else 0.0
    return {"macro_accuracy": macro_accuracy, "accuracy": accuracy, "per_class": per_class,
            "support": dict(totals)}


def evaluate(
    W: Sequence[Sequence[float]],
    b: Sequence[float],
    X: Sequence[Sequence[float]],
    y: Sequence[str],
    categories: Sequence[str],
) -> dict:
    """線形ヘッド `(W, b)` の互換ラッパ（`evaluate_probs(predict_probs("linear", (W, b), X), y, categories)`）。"""
    return evaluate_probs(predict_probs("linear", (W, b), X), y, categories)


def _margin_and_pred(p: Sequence[float]) -> tuple[float, int]:
    """確率 `p` の margin（1 位と 2 位の確率差。クラスが 1 つなら 1 位の確率）と 1 位の添字。"""
    order = sorted(range(len(p)), key=p.__getitem__, reverse=True)
    top2 = p[order[1]] if len(order) > 1 else 0.0
    return p[order[0]] - top2, order[0]


# 確率差の浮動小数誤差（例: 0.5 - 0.45 が 0.05 よりわずかに小さくなる）を境界で吸収する
_MARGIN_EPS = 1e-9


def choose_min_margin(
    probs: list[list[float]],
    y_idx: list[int],
    target_precision: float = 0.95,
    grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[float | None, list[dict]]:
    """grid の各 m について「margin（1 位と 2 位の確率差）≥ m」の予測だけを採用したときの
    precision と被覆率（採用率）を表にし、precision ≥ `target_precision` を満たす最小の m を
    返す（無ければ None）。採用件数が 0 の m は precision を None にする（未定義）。
    """
    margins: list[float] = []
    correct: list[bool] = []
    for p, yi in zip(probs, y_idx):
        margin, pred = _margin_and_pred(p)
        margins.append(margin)
        correct.append(pred == yi)

    n_total = len(probs)
    chosen: float | None = None
    table: list[dict] = []
    for m in grid:
        kept = [i for i, mg in enumerate(margins) if mg >= m - _MARGIN_EPS]
        n = len(kept)
        precision = (sum(1 for i in kept if correct[i]) / n) if n else None
        coverage = (n / n_total) if n_total else 0.0
        table.append({"margin": m, "precision": precision, "coverage": coverage, "n": n})
        if chosen is None and precision is not None and precision >= target_precision:
            chosen = m
    return chosen, table


def precision_at_margin(probs: Sequence[Sequence[float]], y_idx: Sequence[int], margin: float) -> dict:
    """margin ≥ `margin` の予測だけを採用したときの precision（採用 0 件なら None）・被覆率・採用件数・全件数。

    judge の受け入れ条件 (ii) 用: `margin` は dev で選んだ値をそのまま使う（judge では選ばない）。
    """
    kept: list[bool] = []
    for p, yi in zip(probs, y_idx):
        mg, pred = _margin_and_pred(p)
        if mg >= margin - _MARGIN_EPS:
            kept.append(pred == yi)
    n_total = len(probs)
    return {"precision": (sum(kept) / len(kept)) if kept else None,
            "coverage": (len(kept) / n_total) if n_total else 0.0,
            "n": len(kept), "n_total": n_total}


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
# encoder_sha256 は privacy_masker/category_head.py に移した（ランタイムの HeadCategorizer.load
# と同じ実装を共有するため）。ここでは import して使う。

def save_head(
    out_dir: str,
    heads: dict[str, tuple],
    categories: dict[str, list[str]],
    min_margin: dict[str, float | None],
    meta: dict,
    *,
    head_type: str = "linear",
    features: str = "span_mean",
    hidden: int | None = None,
) -> None:
    """`category_head.safetensors` と `category_head.json`（version 2）を `out_dir`（書き出し先）に書く。

    tensors は linear が kind ごとの `{kind}.weight` / `{kind}.bias`、mlp が `{kind}.w1`（hidden×入力）/
    `{kind}.b1` / `{kind}.w2`（クラス数×hidden）/ `{kind}.b2`（`HeadCategorizer.load` が読む形）。
    `category_head.json` には `head_type` / `features` / `hidden`（mlp の中間次元。linear は None。
    mlp で省略すれば重みの形から取る）を書く。

    `meta["encoder_sha256"]` が無ければ `out_dir` の `model.safetensors` から計算する
    （書き出し先がモデルフォルダ = `--install` ならこれで足りる）。書き出し先をモデルフォルダと
    分ける運用（CLI の既定 `<--spans>/candidate`）では、呼び出し側が `meta["encoder_sha256"]` /
    `meta["encoder_dir"]` を明示すること（CLI は常に明示する）。
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(f"unknown head type {head_type!r} (use one of {HEAD_TYPES})")
    if features not in FEATURES:
        raise ValueError(f"unknown features {features!r} (use one of {FEATURES})")
    os.makedirs(out_dir, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for kind, head in heads.items():
        if head_type == "linear":
            W, b = head
            tensors[f"{kind}.weight"] = np.asarray(W, dtype=np.float32)
            tensors[f"{kind}.bias"] = np.asarray(b, dtype=np.float32)
        else:
            W1, b1, W2, b2 = head
            w1 = np.asarray(W1, dtype=np.float32)
            if hidden is None:
                hidden = int(w1.shape[0])
            elif w1.shape[0] != hidden:
                raise ValueError(f"mlp head for {kind!r} has hidden {w1.shape[0]}, expected {hidden}")
            tensors[f"{kind}.w1"] = w1
            tensors[f"{kind}.b1"] = np.asarray(b1, dtype=np.float32)
            tensors[f"{kind}.w2"] = np.asarray(W2, dtype=np.float32)
            tensors[f"{kind}.b2"] = np.asarray(b2, dtype=np.float32)
    save_file(tensors, os.path.join(out_dir, CATEGORY_HEAD_WEIGHTS))

    cfg = {
        "version": 2,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "head_type": head_type,
        "features": features,
        "hidden": hidden if head_type == "mlp" else None,
        "encoder_sha256": meta.get("encoder_sha256") or encoder_sha256(out_dir),
        "encoder_dir": meta.get("encoder_dir", os.path.basename(os.path.normpath(out_dir))),
        "data_sha256": meta.get("data_sha256"),
        "categories": {k: list(v) for k, v in categories.items()},
        "min_margin": dict(min_margin),
        "metrics": meta.get("metrics", {}),
        "merged": meta.get("merged", {}),
    }
    with open(os.path.join(out_dir, CATEGORY_HEAD_CONFIG), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI: 入力の読み込み・ベクトル計算・キャッシュ
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_dir(a: str, b: str) -> bool:
    """`a` と `b` が同じディレクトリを指すか（末尾の区切り・シンボリックリンク・大小文字を区別しない FS も吸収する）。"""
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:          # どちらかが存在しない → 別物
        return False


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dataset_key(
    spans_path: str, limit: int | None, encoder_sha: str | None = None, features: str | None = None
) -> str:
    """`spans_path` の SHA-256（+ `--limit`・encoder の SHA-256・features）。

    features を省略した鍵（manifest の `data_sha256`）はデータだけを表す。ベクトルキャッシュの鍵は
    features を必ず含める（同じスパンでも `span_mean` と `span_ctx` ではベクトルが違う）。
    """
    sha = _sha256_file(spans_path)
    key = sha if limit is None else f"{sha}:limit{limit}"
    if encoder_sha:
        key = f"{key}:enc{encoder_sha}"
    if features:
        key = f"{key}:feat{features}"
    return key


def _cache_path(cache_dir: str, split: str, features: str) -> str:
    """ベクトルキャッシュのパス `vectors_{split}_{features}.npz`（features ごとに別ファイル）。"""
    return os.path.join(cache_dir, f"vectors_{split}_{features}.npz")


def _load_or_build_dataset(
    spans_path: str,
    cache_path: str,
    model,
    tokenizer,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    force: bool = False,
    limit: int | None = None,
    encoder_sha: str | None = None,
    features: str = "span_mean",
) -> dict:
    """`spans_path`（`tools/category_head_data.py` の `spans_*.jsonl`）の行から encoder ベクトルを
    `features` で計算し、ベクトルの取れた行だけを `{"vectors", "labels", "kinds", "masked",
    "no_type_word", "label_source", "skipped", "total"}` にまとめて返す。`vectors` は float のタプルの
    list（純 Python）。`no_type_word` は行の `no_type_word`（無い v1 の行は `masked` で代用する）。

    `cache_path`（`.npz`）が `spans_path` の SHA-256・`--limit`・`encoder_sha`（渡されていれば）・`features` を
    束ねた鍵と一致すれば、encoder を呼ばずにキャッシュから復元する。`encoder_sha` を鍵に含めるのは、
    キャッシュの置き場所（CLI の既定は `--spans` のディレクトリで、複数の NER モデルで共有されうる）で
    別の encoder に差し替えたときに、古いベクトルを誤って再利用しないため。`force=True` は常に再計算する。

    同じ `doc_id` のスパンは同じ文書に属す前提（`category_head_data` の出力の性質）だが、
    念のため `(doc_id, text)` の組でまとめるので、その前提が崩れていても文書ごとに
    正しく別扱いされる（重複エンコードにはなるが誤ったベクトルにはならない）。
    """
    if features not in FEATURES:
        raise ValueError(f"unknown features {features!r} (use one of {FEATURES})")
    key = _dataset_key(spans_path, limit, encoder_sha, features)
    if not force and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["spans_sha256"].item()) == key:
            masked_cached = cached["masked"].astype(bool).tolist()
            return {
                "vectors": [tuple(row) for row in cached["vectors"].astype(np.float32)],
                "labels": cached["labels"].tolist(),
                "kinds": cached["kinds"].tolist(),
                "masked": masked_cached,
                "no_type_word": (cached["no_type_word"].astype(bool).tolist()
                                 if "no_type_word" in cached.files else list(masked_cached)),
                "label_source": cached["label_source"].tolist(),
                "skipped": int(cached["skipped"]),
                "total": int(cached["total"]),
            }

    rows = _read_jsonl(spans_path)
    if limit is not None:
        rows = rows[:limit]

    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["doc_id"], r["text"]), []).append(i)
    texts = [text for _doc_id, text in groups]
    span_lists = [[(rows[i]["start"], rows[i]["end"]) for i in idxs] for idxs in groups.values()]
    group_vectors = encode_span_vectors(texts, span_lists, model, tokenizer, device=device, batch_size=batch_size,
                                        features=features)

    raw: list[bytes | None] = [None] * len(rows)
    for (_key, idxs), vecs in zip(groups.items(), group_vectors):
        for pos, i in enumerate(idxs):
            raw[i] = vecs[pos]

    vectors: list[tuple[float, ...]] = []
    labels: list[str] = []
    kinds: list[str] = []
    masked: list[bool] = []
    no_type_word: list[bool] = []
    label_sources: list[str] = []
    skipped = 0
    for r, v in zip(rows, raw):
        if v is None:
            skipped += 1
            continue
        vectors.append(vector_to_floats(v))
        labels.append(r["label"])
        kinds.append(r["kind"])
        masked.append(bool(r["masked"]))
        no_type_word.append(bool(r.get("no_type_word", r["masked"])))
        label_sources.append(r["label_source"])

    dim = len(vectors[0]) if vectors else 0
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(
        cache_path,
        vectors=np.asarray(vectors, dtype=np.float16) if vectors else np.zeros((0, dim), dtype=np.float16),
        labels=np.array(labels),
        kinds=np.array(kinds),
        masked=np.array(masked, dtype=bool),
        no_type_word=np.array(no_type_word, dtype=bool),
        label_source=np.array(label_sources),
        spans_sha256=np.asarray(key),
        skipped=np.asarray(skipped),
        total=np.asarray(len(rows)),
    )
    return {"vectors": vectors, "labels": labels, "kinds": kinds, "masked": masked, "no_type_word": no_type_word,
            "label_source": label_sources, "skipped": skipped, "total": len(rows)}


def _apply_merges(dataset: dict, merges: dict[str, list[tuple[str, str]]]) -> None:
    """`dataset["labels"]` を in-place で書き換える（kind ごとの from → to）。"""
    labels = dataset["labels"]
    kinds = dataset["kinds"]
    for i in range(len(labels)):
        for frm, to in merges.get(kinds[i], []):
            if labels[i] == frm:
                labels[i] = to
                break


def _by_kind(
    dataset: dict,
    *,
    masked: bool | None = None,
    label_source: str | None = None,
    no_type_word: bool | None = None,
) -> tuple[dict[str, list[list[float]]], dict[str, list[str]]]:
    """`dataset` を kind ごとの `(X, y)` に分ける。`masked` / `no_type_word` を指定すればそのフラグの行だけに、
    `label_source`（"rule" / "qwen"、v1 のデータでは "form"）を指定すればそのラベル出所の行だけに絞る。

    受け入れ条件 (i) の macro accuracy は「規則ラベル付きスパン（種別語あり）」= `label_source="rule",
    no_type_word=False`、min_margin と (ii) は「種別語の無いスパン」= `no_type_word=True`（Qwen ラベル +
    種別語削除写し）で見る。`dataset` に `no_type_word` が無ければ（v1 の形）`masked` で代用する。
    """
    ntw_col = dataset.get("no_type_word", dataset["masked"])
    X: dict[str, list[list[float]]] = defaultdict(list)
    y: dict[str, list[str]] = defaultdict(list)
    for vec, label, kind, is_masked, is_ntw, src in zip(
        dataset["vectors"], dataset["labels"], dataset["kinds"], dataset["masked"], ntw_col, dataset["label_source"]
    ):
        if masked is not None and bool(is_masked) != masked:
            continue
        if no_type_word is not None and bool(is_ntw) != no_type_word:
            continue
        if label_source is not None and src != label_source:
            continue
        X[kind].append(list(vec))
        y[kind].append(label)
    return dict(X), dict(y)


def _parse_merge(spec: str) -> tuple[str, str, str]:
    try:
        kind, rest = spec.split(":", 1)
        frm, to = rest.split("=", 1)
        if not kind or not frm or not to:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid --merge {spec!r}, expected kind:from=to") from None
    return kind, frm, to


def _effective_categories(
    kinds: Sequence[str], merges: dict[str, list[tuple[str, str]]]
) -> dict[str, list[str]]:
    """kind ごとの学習対象カテゴリ = `semantic.CATEGORIES[kind]` から統合元（from）を除いたもの。"""
    categories: dict[str, list[str]] = {}
    for kind in kinds:
        if kind not in CATEGORIES:
            raise ValueError(f"unknown kind {kind!r} (not in semantic.CATEGORIES)")
        merged_away: set[str] = set()
        for frm, to in merges.get(kind, []):
            if frm not in CATEGORIES[kind]:
                raise ValueError(f"--merge: unknown category {frm!r} for kind {kind!r}")
            if to not in CATEGORIES[kind]:
                raise ValueError(f"--merge: unknown target category {to!r} for kind {kind!r}")
            merged_away.add(frm)
        categories[kind] = [c for c in CATEGORIES[kind] if c not in merged_away]
    return categories


# ---------------------------------------------------------------------------
# CLI: 報告
# ---------------------------------------------------------------------------

# 報告の見出し（選択と判定の分離を明示する。v2 設計 §3.1）
REPORT_SELECTION_HEADING = "## 選択は dev だけ。judge は 1 回"


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def _build_report(manifest: dict) -> str:
    """指標だけから Markdown を組み立てる（表層は一切含めない）。dev（選択）と judge（判定）の節に分ける。"""
    acceptance = manifest.get("acceptance", {})
    macro_min = acceptance.get("macro_accuracy_min", ACCEPT_MACRO_ACCURACY)
    target = acceptance.get("target_precision", ACCEPT_TARGET_PRECISION)
    judged = bool(manifest.get("judge", "judge" in manifest["vectors"]))
    hidden = manifest.get("hidden")
    lines = [
        "# 分類ヘッド 学習・評価レポート", "",
        f"作成: {manifest['created']}", f"model: `{manifest['model']}`", f"spans: `{manifest['spans']}`",
        f"ヘッド: features=`{manifest.get('features', 'span_mean')}`, head=`{manifest.get('head_type', 'linear')}`"
        + (f"（hidden={hidden}, dropout={manifest.get('dropout')}）" if hidden is not None else "")
        + f", class weight={manifest.get('class_weight', '-')}, lr={manifest.get('lr', '-')}", "",
        REPORT_SELECTION_HEADING, "",
        "選択（mlp の早期終了・min_margin・統合候補）は dev だけで行う。judge は kind ごとに 1 回の判定にだけ使う"
        "（`--judge` のときだけ読む）。",
    ]
    if not judged:
        lines.append("この実行は `--judge` なし: judge は読んでいない（受け入れは未判定）。")
    lines += ["", "## 件数", "", "| split | 行数 | ベクトル無し(skip) |", "|---|---|---|"]
    for split in ("train", "dev", "judge"):
        v = manifest["vectors"].get(split)
        if v is not None:
            lines.append(f"| {split} | {v['total']} | {v['skipped']} |")

    lines += ["", "## 種別ごとの指標", ""]
    for kind in manifest["kinds"]:
        m = manifest["metrics"][kind]
        d = m["dev"]
        chosen = m["min_margin"]
        chosen_row = next((row for row in d["margin_table"] if chosen is not None and abs(row["margin"] - chosen) < 1e-9),
                          None)
        lines += [
            f"### {kind}", "", "#### dev（選択）", "",
            f"- (i) macro accuracy（dev, 規則ラベル・種別語あり, n={d['n_rule']}）: {d['macro_accuracy']:.4f}",
            f"- 種別語なし accuracy（dev, n={d['n_ntw']}）: {d['ntw_accuracy']:.4f}",
            f"- 採用 min_margin（dev の種別語なしで precision ≥ {target:.2f} の最小値）: {chosen}"
            + (f"（dev の被覆 {chosen_row['coverage']:.3f}）" if chosen_row is not None else ""),
            "", "| margin | precision | coverage | n |", "|---|---|---|---|",
        ]
        for row in d["margin_table"]:
            lines.append(f"| {row['margin']:.2f} | {_fmt(row['precision'], '.3f')} | {row['coverage']:.3f} | {row['n']} |")
        lines += ["", "| category | recall | n |", "|---|---|---|"]
        support = d.get("support", {})
        for cat, recall in sorted(d["per_class"].items()):
            lines.append(f"| {cat} | {_fmt(recall, '.3f')} | {support.get(cat, '-')} |")
        if d["merge_suggestions"]:
            lines += ["", f"- 統合候補（dev の recall < 0.5）: {', '.join(d['merge_suggestions'])}"]
        lines += ["", "#### judge（判定）", ""]
        j = m.get("judge")
        if j is None:
            lines.append("- 未判定（`--judge` なし）")
        else:
            lines += [
                f"- (i) macro accuracy（judge, 規則ラベル・種別語あり, n={j['n_rule']}）: {j['macro_accuracy']:.4f}",
                f"- (ii) precision @min_margin {chosen if chosen is not None else '-（dev で決まらない）'}"
                f"（judge, 種別語なし）: {_fmt(j['precision'])}"
                f"（被覆 {j['coverage']:.3f} = {j['n_kept']}/{j['n_ntw']}）",
                f"- 受け入れ条件（judge: (i) macro accuracy ≥ {macro_min:.2f} かつ (ii) precision ≥ {target:.2f}）: "
                f"{'○' if m['accepted'] else '×'}",
            ]
        lines.append("")

    if "accepted_kinds" in manifest:
        body = (', '.join(manifest['accepted_kinds']) or '（なし）') if judged else '（未判定: `--judge` なし）'
        lines += ["## 受け入れた kind（judge での条件のみ。実文書での確認は別に行う）", "", f"- {body}", ""]
    if "saved_kinds" in manifest:
        where = "（--install: モデルフォルダ）" if manifest.get("install") else "（候補）"
        lines += [f"## 保存した kind{where}", "",
                  f"- {', '.join(manifest['saved_kinds']) or '（なし: ヘッドのファイルは書いていない）'}", ""]
    if manifest["merged"]:
        lines += ["## 適用した統合（--merge）", ""]
        for kind, mapping in manifest["merged"].items():
            for frm, to in mapping.items():
                lines.append(f"- {kind}: {frm} → {to}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="本番 NER モデルのフォルダ（model.safetensors・tokenizer を含む）")
    ap.add_argument("--spans", required=True, help="tools/category_head_data.py --out のディレクトリ")
    ap.add_argument("--out", default=None,
                    help=f"候補の書き出し先（既定は <--spans>/{CANDIDATE_DIRNAME}。モデルフォルダは指定できない）")
    ap.add_argument("--install", action="store_true",
                    help="受け入れ条件（accepted）を満たした kind だけをモデルフォルダに書く（--judge が必須、"
                         "--out とは併用不可）。実文書で確かめてから使う")
    ap.add_argument("--judge", action="store_true",
                    help="spans_judge.jsonl を読み、dev で選んだヘッドと min_margin の受け入れを kind ごとに判定する"
                         "（付けなければ judge は一切読まない）")
    ap.add_argument("--features", choices=FEATURES, default="span_mean",
                    help="スパンベクトルの特徴量（span_mean = スパンの平均、span_ctx = スパン + 左右の文脈）")
    ap.add_argument("--head", choices=HEAD_TYPES, default="linear", help="ヘッドの種類")
    ap.add_argument("--hidden", type=int, default=128, help="mlp の中間次元")
    ap.add_argument("--dropout", type=float, default=0.2, help="mlp の dropout")
    ap.add_argument("--class-weight", choices=CLASS_WEIGHT_MODES, default="sqrt",
                    help="クラス重み（sqrt = 頻度の平方根の逆数、inv = 頻度の逆数）。訓練データに無いクラスも含む"
                         f"全カテゴリ数で割って平均 1 に正規化し、上限 {CLASS_WEIGHT_CAP:g} で切る"
                         f"（inv は上限を超える重みが無い限り v1 と同じ重み）")
    ap.add_argument("--patience", type=int, default=5,
                    help=f"mlp の早期終了: dev の macro accuracy を {EARLY_STOP_EVERY} epoch ごとに見て、"
                         "この回数続けて改善しなければ止め、最良の重みに戻す（macro accuracy が同点なら dev の"
                         "平均 NLL が下がったときも改善とみなす）")
    ap.add_argument("--report", default=None, help="Markdown の報告を書くパス")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None,
                    help=f"学習率（既定は linear {DEFAULT_LR['linear']:g} / mlp {DEFAULT_LR['mlp']:g}）")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--merge", action="append", type=_parse_merge, default=[], metavar="kind:from=to",
                    help="学習前にラベルを統合する（繰り返し指定可。例: org:party=association）")
    ap.add_argument("--kinds", default=None,
                    help="学習・保存する kind をカンマ区切りで限定する（例: org。受け入れた kind だけを保存する用）")
    ap.add_argument("--cache-dir", default=None,
                    help="ベクトルキャッシュ vectors_*.npz の置き場所（既定は --spans のディレクトリ。モデルフォルダには置かない）")
    ap.add_argument("--force", action="store_true", help="ベクトルキャッシュを無視して再計算する")
    ap.add_argument("--limit", type=int, default=None, help="動作確認用: 分割ごとに読み込む行数の上限（--install とは併用不可）")
    args = ap.parse_args(argv)

    if args.install and not args.judge:
        ap.error("--install requires --judge (acceptance is decided on the judge split)")
    if args.install and args.out is not None:
        ap.error("--install writes into --model; do not combine it with --out")
    if args.install and args.limit is not None:
        # --limit は動作確認用の部分データ。部分データで選んだ・判定したヘッドを本番のモデルフォルダへ書かない
        ap.error("--install must not be combined with --limit (partial data)")
    if args.install and args.target_precision < ACCEPT_TARGET_PRECISION:
        ap.error(f"--install requires --target-precision >= {ACCEPT_TARGET_PRECISION} "
                 f"(got {args.target_precision}); a lower target would install a head that fails the acceptance gate")
    if not args.install and args.out is not None and _same_dir(args.out, args.model):
        ap.error("--out must not be the model directory (the detector auto-loads a category head from it); "
                 "use --install to write the accepted kinds into it")
    if args.hidden < 1:
        ap.error(f"--hidden must be >= 1 (got {args.hidden})")
    if not 0.0 <= args.dropout < 1.0:
        ap.error(f"--dropout must be in [0, 1) (got {args.dropout})")
    if args.patience < 1:
        ap.error(f"--patience must be >= 1 (got {args.patience})")
    out_dir = args.model if args.install else (args.out or os.path.join(args.spans, CANDIDATE_DIRNAME))
    cache_dir = args.cache_dir or args.spans
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.head]
    hidden = args.hidden if args.head == "mlp" else None
    only_kinds: list[str] | None = None
    if args.kinds:
        only_kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
        unknown = [k for k in only_kinds if k not in CATEGORIES or k in ("person", "address")]
        if unknown or not only_kinds:
            ap.error(f"--kinds: unknown kinds {unknown!r} (use org, facility, location)")
    # judge は --judge のときだけ（無いときはファイルの存在も見ない。選択に使っていないことを構造で保つ）
    paths = {split: os.path.join(args.spans, f"spans_{split}.jsonl") for split in ("train", "dev")}
    if args.judge:
        paths["judge"] = os.path.join(args.spans, "spans_judge.jsonl")
    for p in paths.values():
        if not os.path.exists(p):
            raise SystemExit(f"input not found: {p}")

    from transformers import AutoTokenizer

    model = load_char_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    enc_sha = encoder_sha256(args.model)

    os.makedirs(out_dir, exist_ok=True)
    datasets = {
        split: _load_or_build_dataset(path, _cache_path(cache_dir, split, args.features), model, tokenizer,
                                      device=args.device, batch_size=args.batch_size, force=args.force,
                                      limit=args.limit, encoder_sha=enc_sha, features=args.features)
        for split, path in paths.items()
    }
    train_ds, dev_ds = datasets["train"], datasets["dev"]

    merges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for kind, frm, to in args.merge:
        merges[kind].append((frm, to))
    for ds in datasets.values():
        _apply_merges(ds, merges)

    kinds = sorted(set(train_ds["kinds"]) | set(dev_ds["kinds"]))       # judge は kind の決定にも使わない
    if only_kinds is not None:
        kinds = [k for k in kinds if k in only_kinds]
    categories = _effective_categories(kinds, merges)

    X_train, y_train = _by_kind(train_ds)
    X_train = {k: v for k, v in X_train.items() if k in kinds}      # --kinds で外した kind は学習しない
    y_train = {k: v for k, v in y_train.items() if k in kinds}
    missing = [k for k in kinds if k not in X_train]
    if missing:
        # --limit を小さくした動作確認などで、ある kind の訓練行がたまたま 0 件になった場合。
        # 学習できない kind は落とす（他の kind の学習・保存は続ける）。
        print(f"skipping kinds with no training vectors: {missing}", file=sys.stderr)
        kinds = [k for k in kinds if k not in missing]
        categories = {k: categories[k] for k in kinds}

    # mlp の早期終了は dev の全行（規則 + Qwen + 種別語削除写し）の macro accuracy で見る
    X_dev_all, y_dev_all = _by_kind(dev_ds)
    heads = train_heads(X_train, y_train, categories, head=args.head, hidden=args.hidden, dropout=args.dropout,
                        epochs=args.epochs, lr=lr, weight_decay=args.weight_decay, seed=args.seed,
                        class_weight=args.class_weight, X_dev=X_dev_all, y_dev=y_dev_all, patience=args.patience)

    # (i) は規則ラベル付きスパン（種別語あり）、min_margin と (ii) は種別語の無いスパン（Qwen + 削除写し）で見る
    X_dev_rule, y_dev_rule = _by_kind(dev_ds, label_source="rule", no_type_word=False)
    X_dev_ntw, y_dev_ntw = _by_kind(dev_ds, no_type_word=True)
    if args.judge:
        X_j_rule, y_j_rule = _by_kind(datasets["judge"], label_source="rule", no_type_word=False)
        X_j_ntw, y_j_ntw = _by_kind(datasets["judge"], no_type_word=True)

    metrics: dict[str, dict] = {}
    min_margin: dict[str, float | None] = {}
    for kind in kinds:
        cats = categories[kind]
        head = heads[kind]
        probs_dev_rule = predict_probs(args.head, head, X_dev_rule.get(kind, []))
        dev_rule = evaluate_probs(probs_dev_rule, y_dev_rule.get(kind, []), cats)
        probs_dev_ntw = predict_probs(args.head, head, X_dev_ntw.get(kind, []))
        y_idx = [cats.index(lab) for lab in y_dev_ntw.get(kind, [])]
        chosen, table = choose_min_margin(probs_dev_ntw, y_idx, target_precision=args.target_precision)
        min_margin[kind] = chosen

        judge: dict | None = None
        if args.judge:
            judge_rule = evaluate_probs(predict_probs(args.head, head, X_j_rule.get(kind, [])),
                                        y_j_rule.get(kind, []), cats)
            probs_j = predict_probs(args.head, head, X_j_ntw.get(kind, []))
            if chosen is not None:
                pm = precision_at_margin(probs_j, [cats.index(lab) for lab in y_j_ntw.get(kind, [])], chosen)
            else:       # dev で min_margin が決まらない kind は judge で margin を探さない（judge で選ばない）
                pm = {"precision": None, "coverage": 0.0, "n": 0, "n_total": len(probs_j)}
            judge = {"macro_accuracy": judge_rule["macro_accuracy"], "n_rule": len(X_j_rule.get(kind, [])),
                     "per_class": judge_rule["per_class"], "support": judge_rule["support"],
                     "precision": pm["precision"], "coverage": pm["coverage"], "n_kept": pm["n"], "n_ntw": pm["n_total"]}
        metrics[kind] = {
            "dev": {
                "macro_accuracy": dev_rule["macro_accuracy"],
                "n_rule": len(X_dev_rule.get(kind, [])),
                "ntw_accuracy": evaluate_probs(probs_dev_ntw, y_dev_ntw.get(kind, []), cats)["accuracy"],
                "n_ntw": len(probs_dev_ntw),
                "margin_table": table,
                "per_class": dev_rule["per_class"],
                "support": dev_rule["support"],
                "merge_suggestions": sorted(c for c, r in dev_rule["per_class"].items() if r is not None and r < 0.5),
            },
            "min_margin": chosen,
            "judge": judge,
            "accepted": (is_accepted(judge["macro_accuracy"], judge["precision"], chosen, args.target_precision)
                         if judge is not None else None),
            "n_train": len(X_train.get(kind, [])),
        }

    merged_record = {k: {frm: to for frm, to in v} for k, v in merges.items()}
    # min_margin が見つからなかった kind（dev の種別語なしで precision が target に届く margin が grid に無い）は
    # 保存しない: 採用値の無いヘッドは使えないうえ、ランタイムの HeadCategorizer.load は
    # min_margin=None を読めない。--install はさらに judge の受け入れ条件（accepted is True）を満たした
    # kind だけに絞る（accepted=None = 未判定は入れない）。指標は manifest / --report に全 kind 分を残す。
    accepted_kinds = [k for k in kinds if metrics[k]["accepted"] is True]
    dropped = [k for k in kinds if min_margin[k] is None]
    if dropped:
        print(f"not saving kinds without a min_margin reaching precision {args.target_precision} on dev: {dropped}",
              file=sys.stderr)
    if args.install:
        saved_kinds = accepted_kinds
        rejected = [k for k in kinds if min_margin[k] is not None and metrics[k]["accepted"] is not True]
        if rejected:
            print(f"not installing kinds that failed the judge acceptance gate: {rejected}", file=sys.stderr)
    else:
        saved_kinds = [k for k in kinds if min_margin[k] is not None]
        if args.judge:
            not_accepted = [k for k in saved_kinds if metrics[k]["accepted"] is not True]
            if not_accepted:
                print(f"candidate includes kinds that are not accepted (see metrics.accepted): {not_accepted}",
                      file=sys.stderr)
        elif saved_kinds:
            print(f"candidate kinds are not judged (accepted is null; run with --judge to decide): {saved_kinds}",
                  file=sys.stderr)
    data_sha = {split: _dataset_key(path, args.limit) for split, path in paths.items()}
    meta = {
        "encoder_sha256": enc_sha,
        "encoder_dir": os.path.basename(os.path.normpath(args.model)),
        "data_sha256": data_sha,
        "metrics": {k: metrics[k] for k in saved_kinds},
        "merged": {k: v for k, v in merged_record.items() if k in saved_kinds},
    }
    if saved_kinds:
        save_head(out_dir, {k: heads[k] for k in saved_kinds}, {k: categories[k] for k in saved_kinds},
                  {k: min_margin[k] for k in saved_kinds}, meta,
                  head_type=args.head, features=args.features, hidden=hidden)
    else:
        print("no kind to save: category head files were not written", file=sys.stderr)
        stale = [n for n in (CATEGORY_HEAD_WEIGHTS, CATEGORY_HEAD_CONFIG) if os.path.exists(os.path.join(out_dir, n))]
        if stale:   # 以前の実行で書いたヘッドが残っている（自動では消さない。不採用なら手で外す）
            print(f"warning: existing {stale} in {out_dir} were left untouched and do not match this run",
                  file=sys.stderr)

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "spans": args.spans,
        "out": out_dir,
        "install": bool(args.install),
        "judge": bool(args.judge),
        "features": args.features,
        "head_type": args.head,
        "hidden": hidden,
        "dropout": args.dropout if args.head == "mlp" else None,
        "class_weight": args.class_weight,
        "lr": lr,
        "patience": args.patience,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "encoder_sha256": enc_sha,
        "data_sha256": data_sha,
        "acceptance": {"macro_accuracy_min": ACCEPT_MACRO_ACCURACY, "target_precision": args.target_precision},
        "kinds": kinds,
        "accepted_kinds": accepted_kinds,
        "saved_kinds": saved_kinds,
        "vectors": {split: {"total": ds["total"], "skipped": ds["skipped"]} for split, ds in datasets.items()},
        "metrics": metrics,
        "merged": merged_record,
    }
    with open(os.path.join(out_dir, TRAIN_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(_build_report(manifest))

    summary = {
        "kinds": kinds,
        "vectors": manifest["vectors"],
        "metrics": {k: dict(m, dev={kk: vv for kk, vv in m["dev"].items() if kk != "margin_table"})
                    for k, m in metrics.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
