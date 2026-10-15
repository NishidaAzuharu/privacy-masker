"""2モデル間で「危険（coverage<50%）の gold エンティティ」の差分を出す。
usage: PYTHONPATH=. .venv/bin/python tools/diff_danger.py <model_dir または Hub ID> <model_dir または Hub ID>

モデルの指定はローカルのモデルフォルダか Hugging Face Hub のリポジトリ ID（owner/name）。
データフォルダ（PRIVACY_MASKER_DRIVE_DIR）に学習済みモデルを持っている場合は PATHS の登録名も使える。
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # tools/eval_ood_dashboard.py
sys.path.insert(0, os.getcwd())
import eval_ood_dashboard as d
from privacy_masker.detectors import CharHeadNerDetector, TransformersNerDetector
from privacy_masker.refiner_janome import MorphologicalRefiner

GD = d.GD
# データフォルダ内の学習済みモデルの登録名（開発用の短縮名 → フォルダ）。第三者はフォルダパスか Hub ID を渡す。
PATHS = {
    "30m(v4)": f"{GD}/outputs_ruri-v3-30m-v4data/best",
    "30m(v5)": f"{GD}/outputs_ruri-v3-30m-v5data/best",
    "30m(v5qwen)": f"{GD}/outputs_ruri-v3-30m-v5qwen/best",
    "70m(v4)": f"{GD}/outputs_ruri-v3-70m-v4data/best",
    "70m(v5)": f"{GD}/outputs_ruri-v3-70m-v5data/best",
    "70m(v5qwen)": f"{GD}/outputs_ruri-v3-70m-v5qwen/best",
    "30m(v45b)": f"{GD}/outputs_ruri-v3-30m-v45b/best",
    "70m(v45b)": f"{GD}/outputs_ruri-v3-70m-v45b/best",
    "30m(v45bqwen)": f"{GD}/outputs_ruri-v3-30m-v45bqwen/best",
    "70m(v45bqwen)": f"{GD}/outputs_ruri-v3-70m-v45bqwen/best",
    "30m(v45c)": f"{GD}/outputs_ruri-v3-30m-v45c/best",
    "70m(v45c)": f"{GD}/outputs_ruri-v3-70m-v45c/best",
    "30m(v45cq)": f"{GD}/outputs_ruri-v3-30m-v45cq/best",
    "70m(v45cq)": f"{GD}/outputs_ruri-v3-70m-v45cq/best",
    "30m(v45d)": f"{GD}/outputs_ruri-v3-30m-v45d/best",
    "70m(v45d)": f"{GD}/outputs_ruri-v3-70m-v45d/best",
    "30m(v45e)": f"{GD}/outputs_ruri-v3-30m-v45e/best",
    "70m(v45e)": f"{GD}/outputs_ruri-v3-70m-v45e/best",
}


def model_path(spec: str, registry: dict | None = None) -> str:
    """モデルの指定をローカルのフォルダにする。

    登録名（PATHS、または registry）→ 既存のフォルダ → Hugging Face Hub のリポジトリ ID（snapshot_download で取得）の順に解決する。
    """
    for reg in (registry or {}, PATHS):
        if spec in reg:
            return reg[spec]
    if os.path.isdir(spec):
        return spec
    from privacy_masker.hub import resolve_model_dir
    from privacy_masker.exceptions import ConfigurationError
    try:
        return resolve_model_dir(spec)
    except ConfigurationError as e:
        raise SystemExit(f"unknown model: {spec}\n  {e}") from e


def build_detector(path: str, device: str = "cpu"):
    """モデルフォルダの中身で文字ヘッド版と token 版を出し分ける（char_head_config.json があれば文字ヘッド付き）。"""
    if os.path.exists(os.path.join(path, "char_head_config.json")):
        return CharHeadNerDetector(model=path, device=device)
    return TransformersNerDetector(model=path, device=device)


def danger_set(tag, docs, ref):
    det = build_detector(model_path(tag), device="cpu")
    preds = det.detect_batch([x["text"] for x in docs])
    out = {}
    fp_by_genre = defaultdict(int)
    for i, (x, raw) in enumerate(zip(docs, preds)):
        ents = ref.refine(x["text"], raw)
        pred = set()
        for e in ents:
            pred.update(range(e.start, e.end))
        neutral = set()
        for s, e in x.get("neutral", []):
            neutral.update(range(s, e))
        gold_all = set()
        for s, e, lab in x["entities"]:
            span = set(range(s, e)); gold_all |= span
            cov = len(span & pred) / len(span)
            if cov < 0.5:
                out[(i, s, e)] = (x["genre"], lab, x["text"][s:e], x["text"][max(0, s-15):e+10].replace("\n", " "), cov)
        fp_by_genre[x["genre"]] += len(pred - gold_all - neutral)
    return out, fp_by_genre

def main():
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); raise SystemExit(0 if "-h" in sys.argv or "--help" in sys.argv else 2)
    base, cand = sys.argv[1], sys.argv[2]
    docs = d.load_docs(); ref = MorphologicalRefiner()
    db, fpb = danger_set(base, docs, ref)
    dc, fpc = danger_set(cand, docs, ref)
    print(f"危険gold: {base}={len(db)}  {cand}={len(dc)}  共通={len(set(db)&set(dc))}")
    print(f"\n--- {cand} で新たに危険になった gold（{base} では守れていた）: {len(set(dc)-set(db))}件 ---")
    for k in sorted(set(dc) - set(db), key=lambda k: dc[k][0]):
        g, lab, surf, ctx, cov = dc[k]; print(f"  [{g:<10}] {lab:<9} {surf!r:<22} cov={cov:.2f} ｟{ctx}｠")
    print(f"\n--- {cand} で解消した危険 gold（{base} では危険だった）: {len(set(db)-set(dc))}件 ---")
    for k in sorted(set(db) - set(dc), key=lambda k: db[k][0]):
        g, lab, surf, ctx, cov = db[k]; print(f"  [{g:<10}] {lab:<9} {surf!r:<22} ｟{ctx}｠")
    print(f"\n--- 過剰マスク文字数（ジャンル別） {base} → {cand} ---")
    for g in sorted(set(fpb) | set(fpc)):
        print(f"  {g:<12} {fpb[g]:5d} → {fpc[g]:5d}")

if __name__ == "__main__":
    main()
