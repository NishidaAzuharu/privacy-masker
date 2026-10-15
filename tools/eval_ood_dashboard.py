"""ジャンル別OOD評価ダッシュボード。

モデルが「どのジャンルの文書でどれだけ汎化しているか」を1枚の表にする。
指標:
  - cov   : 文字レベルcoverage（gold PII文字のうちマスクされた割合。安全KPI）
  - FP字  : gold外を検出した文字数（過剰マスク量。中立領域は除外）
  - 危険  : coverage 50%未満のgoldエンティティ数（実質的な見逃し）

実行（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/eval_ood_dashboard.py [<model_dir または Hub ID> ...]

モデルはローカルのモデルフォルダか Hugging Face Hub のリポジトリ ID（owner/name）。引数を省略すると、
データフォルダ（PRIVACY_MASKER_DRIVE_DIR）内の MODELS に登録したモデルのうち存在するものを評価する。ジャンルデータはデータフォルダの eval_ood_v1.jsonl（自作8ジャンル）
+ eval_ood_news_v1.jsonl（第三者: wikinews）+ 既存 eval_real/eval_hard を自動で読む。
"""
import json
import os
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from privacy_masker.detectors import CharHeadNerDetector, TransformersNerDetector  # noqa: E402
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402

GD = drive_dir(require=False)   # import 時は止めない（tests が import する）。存在確認は main() の冒頭で行う

MODELS = {
    "30m(v5)": f"{GD}/outputs_ruri-v3-30m-v5data/best",
    "70m(v5)": f"{GD}/outputs_ruri-v3-70m-v5data/best",
    "30m(v4)": f"{GD}/outputs_ruri-v3-30m-v4data/best",
    "70m(v4)": f"{GD}/outputs_ruri-v3-70m-v4data/best",
    "30m(v3)": f"{GD}/outputs_ruri-v3-30m-v3data/best",
    "70m(v2)": f"{GD}/outputs_ruri-v3-70m/best",
    "30m(v2)": f"{GD}/outputs_ruri-v3-30m/best",
}


def load_docs():
    docs = []
    for path, default_genre in [
        (f"{GD}/eval_ood_v1.jsonl", None),          # 自作8ジャンル（genreフィールド持ち）
        (f"{GD}/eval_ood_news_v1.jsonl", "news"),   # 第三者: wikinews
        (f"{GD}/eval_real_v1.jsonl", "business"),   # 既存: 実文風
        (f"{GD}/eval_hard_v1.jsonl", "hard"),       # 既存: 難例
    ]:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        for line in open(path):
            d = json.loads(line)
            d.setdefault("genre", default_genre)
            d.setdefault("neutral", [])
            docs.append(d)
    return docs


def resolve_models(specs):
    """引数のモデル指定（フォルダパス / Hub ID / "表示名=パス"）を {表示名: フォルダ} にする。空なら MODELS。"""
    if not specs:
        return dict(MODELS)
    from privacy_masker.hub import resolve_model_dir
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            path = name
        out[name] = MODELS.get(path) or resolve_model_dir(path)
    return out


def build_detector(path, device="cpu"):
    if os.path.exists(os.path.join(path, "char_head_config.json")):
        return CharHeadNerDetector(model=path, device=device)
    return TransformersNerDetector(model=path, device=device)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__); return
    drive_dir()   # データフォルダが無ければ分かりやすく止める（import ではなく実行時に）
    models = resolve_models(argv)
    docs = load_docs()
    genres = sorted({d["genre"] for d in docs})
    print(f"評価文書: {len(docs)}件 / ジャンル: {genres}\n")

    ref = MorphologicalRefiner()
    for mname, mpath in models.items():
        if not os.path.isdir(mpath):
            print(f"[skip] {mname}: モデルなし")
            continue
        det = build_detector(mpath, device="cpu")  # CPU固定で再現性を確保
        preds = det.detect_batch([d["text"] for d in docs])
        stat = defaultdict(lambda: [0, 0, 0, 0, 0])  # gold字, hit字, FP字, 危険ent, gold ent
        for d, raw in zip(docs, preds):
            ents = ref.refine(d["text"], raw)
            pred = set()
            for e in ents:
                pred.update(range(e.start, e.end))
            neutral = set()
            for s, e in d["neutral"]:
                neutral.update(range(s, e))
            gold_all = set()
            row = stat[d["genre"]]
            for s, e, _ in d["entities"]:
                span = set(range(s, e))
                gold_all |= span
                row[4] += 1
                if len(span & pred) / len(span) < 0.5:
                    row[3] += 1
            row[0] += len(gold_all)
            row[1] += len(gold_all & pred)
            row[2] += len(pred - gold_all - neutral)

        print(f"===== {mname} =====")
        print(f"{'ジャンル':<12} {'cov':>7} {'FP字':>6} {'危険/ent':>10}")
        worst = []
        for g in genres:
            gold, hit, fp, danger, n_ent = stat[g]
            cov = hit / gold * 100 if gold else 100.0
            worst.append((cov, g))
            print(f"{g:<12} {cov:6.1f}% {fp:6d} {danger:>5}/{n_ent}")
        total = [sum(stat[g][i] for g in genres) for i in range(5)]
        print(f"{'全体':<12} {total[1]/total[0]*100:6.1f}% {total[2]:6d} {total[3]:>5}/{total[4]}")
        print(f"  最弱ジャンル: {min(worst)[1]} ({min(worst)[0]:.1f}%)\n")


if __name__ == "__main__":
    main()
