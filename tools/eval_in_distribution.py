"""分布内評価: 当時の学習ノートブックと同じ手順（ソース別 dedup → seed 42 で shuffle → 90/10）で検証分割を再構成し、
本番設定（NER + 正規表現 + MorphologicalRefiner）で ソース別・ラベル別 の coverage / 危険 / 完全見逃し / 1字以上漏れ を出す。
usage: PYTHONPATH=. .venv/bin/python tools/eval_in_distribution.py <model_dir または Hub ID> [...]
（データフォルダに学習済みモデルを持っている場合は diff_danger.PATHS の登録名も使える）
"""
import json, os, random, sys
from collections import Counter
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_danger as dd
from privacy_masker.detectors import RegexDetector
from privacy_masker.refiner_janome import MorphologicalRefiner

GD = dd.GD; SEED = 42
LABEL_MAP = {"人名": "PERSON", "法人名": "ORG", "政治的組織名": "ORG", "その他の組織名": "ORG", "地名": "LOCATION", "施設名": "FACILITY"}
BASE = {"ADDRESS", "FACILITY", "LOCATION", "ORG", "PERSON"}
SOURCES = {"wiki(out_ner)": "out_ner.jsonl", "合成(手テンプレ)": "synthetic_pii_v4.jsonl", "合成(LLMテンプレ)": "synthetic_pii_v5c.jsonl"}
PATHS = dict(dd.PATHS); PATHS.update({"30m(v45cq)": f"{GD}/outputs_ruri-v3-30m-v45cq/best", "70m(v45cq)": f"{GD}/outputs_ruri-v3-70m-v45cq/best"})

def normalize(records):
    out = []
    for rec in records:
        text = rec.get("text", ""); ents = []
        for it in rec.get("entities", []) or []:
            s, e, lab = it[0], it[1], str(it[2]).strip()
            lab = LABEL_MAP.get(lab, lab if lab in BASE else None)
            if lab is None or not (0 <= s < e <= len(text)): continue
            ents.append([s, e, lab])
        out.append({"text": text, "entities": ents})
    return out

def valid_split(path):
    recs = normalize([json.loads(l) for l in open(path) if l.strip()])
    seen, uniq = set(), []
    for r in recs:
        if r["text"] in seen: continue
        seen.add(r["text"]); uniq.append(r)
    rnd = random.Random(SEED); rnd.shuffle(uniq)
    return uniq[max(1, int(0.9 * len(uniq))):]

def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__); return
    ref = MorphologicalRefiner(); rx = RegexDetector.default_ja()
    splits = {name: valid_split(f"{GD}/{f}") for name, f in SOURCES.items()}
    for tag in sys.argv[1:]:
        det = dd.build_detector(dd.model_path(tag, registry=PATHS), device="cpu")
        print(f"===== {tag} =====")
        for name, docs in splits.items():
            preds = det.detect_batch([d["text"] for d in docs])
            gold_c = cov_c = fp_c = 0; total = Counter(); danger = Counter(); miss = Counter(); anyleak = Counter(); ex = []
            for d, raw in zip(docs, preds):
                t = d["text"]; ents = ref.refine(t, raw + rx.detect(t)); pc = set()
                for e in ents: pc.update(range(e.start, e.end))
                gold_all = set()
                for s, e_, lab in d["entities"]:
                    span = set(range(s, e_)); gold_all |= span; total[lab] += 1
                    c = len(span & pc) / len(span)
                    if c < 0.5: danger[lab] += 1
                    if c == 0: miss[lab] += 1
                    if c < 1: anyleak[lab] += 1
                    if lab in ("PERSON", "ADDRESS") and c < 1 and len(ex) < 6:
                        ex.append(f"{lab}:{t[s:e_]}→残{''.join(t[i] for i in sorted(span - pc))!r}")
                gold_c += len(gold_all); cov_c += len(gold_all & pc); fp_c += len(pc - gold_all)
            n = sum(total.values())
            print(f"  {name}: 文書{len(docs)} gold {n}  文字cov {cov_c/gold_c*100:.2f}%  危険 {sum(danger.values())}  完全見逃し {sum(miss.values())}  1字以上漏れ {sum(anyleak.values())}  gold外マスク字 {fp_c}")
            print("    ラベル別 危険/完全見逃し/1字以上漏れ/総数: " + "  ".join(f"{l}={danger[l]}/{miss[l]}/{anyleak[l]}/{total[l]}" for l in sorted(total)))
            print("    PERSON/ADDRESS 漏れ例:", ex)

if __name__ == "__main__":
    main()
