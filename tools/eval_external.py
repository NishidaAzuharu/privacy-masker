"""外部評価（開発に未使用）: UD Japanese GSD-NE test（543文）と OpenPII 日本語 holdout（1,000文）で
本番設定（NER + 正規表現 + MorphologicalRefiner）の coverage / 危険 / 完全見逃し / 1文字以上漏れ をラベル別に出す。
gold と neutral は変換済みの *_predictions.jsonl（PRIVACY_MASKER_EXTERNAL_EVAL、既定はデータフォルダの external_eval/）を使う。
usage: PYTHONPATH=. .venv/bin/python tools/eval_external.py <model_dir または Hub ID> [...] [--raw]
（データフォルダに学習済みモデルを持っている場合は diff_danger.PATHS の登録名も使える）
"""
import ast, json, os, sys
from collections import Counter
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_danger as dd
from privacy_masker.detectors import RegexDetector
from privacy_masker.refiner_janome import MorphologicalRefiner

EXT = os.environ.get("PRIVACY_MASKER_EXTERNAL_EVAL", os.path.join(dd.GD, "external_eval"))   # 変換済み gold（<name>_predictions.jsonl）の置き場
SETS = {"gsd_ne_test": f"{EXT}/gsd_ne_test_predictions.jsonl", "openpii_ja_holdout": f"{EXT}/openpii_ja_holdout_predictions.jsonl"}
PATHS = dict(dd.PATHS); PATHS.update({
    "30m(v45cq)": f"{dd.GD}/outputs_ruri-v3-30m-v45cq/best", "70m(v45cq)": f"{dd.GD}/outputs_ruri-v3-70m-v45cq/best"})

def parse(v):
    return v if isinstance(v, list) else ast.literal_eval(v) if isinstance(v, str) else []

def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__); return
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]; raw_mode = "--raw" in sys.argv
    ref = MorphologicalRefiner(); rx = RegexDetector.default_ja()
    data = {}
    for name, path in SETS.items():
        docs = [json.loads(l) for l in open(path)]
        for d in docs:
            d["gold"] = parse(d["gold"]); d["neutral"] = parse(d.get("neutral", []))
        data[name] = docs
    for tag in tags:
        det = dd.build_detector(dd.model_path(tag, registry=PATHS), device="cpu")
        for name, docs in data.items():
            preds = det.detect_batch([d["text"] for d in docs])
            gold_c = cov_c = fp_c = 0; total = Counter(); danger = Counter(); miss = Counter(); anyleak = Counter(); ex = []
            for d, raw in zip(docs, preds):
                t = d["text"]
                ents = raw if raw_mode else ref.refine(t, raw + rx.detect(t))
                pc = set()
                for e in ents: pc.update(range(e.start, e.end))
                neutral = set()
                for s, e_ in d["neutral"]: neutral.update(range(s, e_))
                gold_all = set()
                for s, e_, lab in d["gold"]:
                    span = set(range(s, e_)); gold_all |= span; total[lab] += 1
                    c = len(span & pc) / len(span)
                    if c < 0.5: danger[lab] += 1
                    if c == 0: miss[lab] += 1
                    if c < 1: anyleak[lab] += 1
                    if lab in ("PERSON", "ADDRESS") and c < 0.5 and len(ex) < 8:
                        ex.append(f"{lab}:{t[s:e_]}｟{t[max(0,s-8):e_+5]}｠")
                gold_c += len(gold_all); cov_c += len(gold_all & pc); fp_c += len(pc - gold_all - neutral)
            n = sum(total.values())
            print(f"=== {tag}{' [raw NER]' if raw_mode else ' [本番設定]'} / {name}: 文書{len(docs)} gold entity {n}")
            print(f"  文字coverage {cov_c/gold_c*100:.1f}%  危険(cov<50%) {sum(danger.values())}  完全見逃し {sum(miss.values())}  1文字以上漏れ {sum(anyleak.values())}  過剰マスク字 {fp_c}")
            print("  ラベル別 危険/完全見逃し/1字以上漏れ/総数: " + "  ".join(f"{l}={danger[l]}/{miss[l]}/{anyleak[l]}/{total[l]}" for l in sorted(total)))
            print("  PERSON/ADDRESS 危険例:", ex)

if __name__ == "__main__":
    main()
