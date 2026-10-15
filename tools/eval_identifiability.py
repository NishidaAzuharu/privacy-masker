"""識別性ベース匿名化の安全コスト測定（ライブラリ実装 IdentifiabilityAnnotator を使用）。

各評価セットの gold エンティティのうち、standard レベルで「保持」されるものを数える。
  私人文書での保持 = 見逃し（安全コスト）。ニュースでの保持 = 望ましい（公知を残す）。
実行（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python -W ignore tools/eval_identifiability.py
"""
import json
import os
import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from privacy_masker import IdentifiabilityAnnotator, MorphologicalRefiner, PublicKnowledge  # noqa: E402
from privacy_masker.detectors import CharHeadNerDetector, RegexDetector, TransformersNerDetector  # noqa: E402
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir  # noqa: E402

GD = drive_dir()
MODEL = os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID)   # モデルフォルダか Hub ID（第 2 引数でも指定可）
SETS = [
    ("私人: eval_ood_v1(8ジャンル)", [f"{GD}/eval_ood_v1.jsonl"]),
    ("私人: eval_real+hard", [f"{GD}/eval_real_v1.jsonl", f"{GD}/eval_hard_v1.jsonl"]),
    ("私人: 合成文書(282)", [f"{GD}/synthetic_pii_claude_v1.jsonl"]),
    ("公知: wikinews(178)", [f"{GD}/eval_ood_news_v1.jsonl"]),
]


def main() -> None:
    level = sys.argv[1] if len(sys.argv) > 1 else "standard"
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    knowledge = PublicKnowledge.load(f"{GD}/jawiki_titles_ns0.pmk")
    ann = IdentifiabilityAnnotator(level=level, knowledge=knowledge)
    model = resolve_model_dir(model)
    det = (CharHeadNerDetector if os.path.exists(os.path.join(model, "char_head_config.json")) else TransformersNerDetector)(model=model, device="cpu")
    rx = RegexDetector.default_ja()
    ref = MorphologicalRefiner()
    print(f"level={level}  model={model}  knowledge={knowledge}\n")
    for name, paths in SETS:
        docs = [json.loads(l) for p in paths for l in open(p)]
        preds = det.detect_batch([d["text"] for d in docs])
        kept = defaultdict(int)
        total = defaultdict(int)
        ex = defaultdict(list)
        fp_old = fp_new = 0
        for d, raw in zip(docs, preds):
            t = d["text"]
            ents = ann.annotate(t, ref.refine(t, raw + rx.detect(t)))
            masked_old, masked_new = set(), set()
            for e in ents:
                masked_old.update(range(e.start, e.end))
                if not e.entity_type.endswith("_PUBLIC"):
                    masked_new.update(range(e.start, e.end))
            gold_all = set()
            for s, e_, lab in d["entities"]:
                span = set(range(s, e_))
                gold_all |= span
                total[lab] += 1
                cov_old = len(span & masked_old) / len(span)
                cov_new = len(span & masked_new) / len(span)
                if cov_old >= 0.5 and cov_new < 0.5:  # strict では守れていたが standard では保持される
                    kept[lab] += 1
                    if len(ex[lab]) < 4:
                        ctx = t[max(0, s - 12):e_ + 6].replace("\n", " ")
                        ex[lab].append(f"{t[s:e_]}｟{ctx}｠")
            neutral = set()
            for s, e_ in d.get("neutral", []):
                neutral.update(range(s, e_))
            fp_old += len(masked_old - gold_all - neutral)
            fp_new += len(masked_new - gold_all - neutral)
        print(f"=== {name}: 文書{len(docs)}件 ===")
        print(f"  過剰マスク文字数: strict {fp_old} → {level} {fp_new}")
        for lab in sorted(total, key=lambda k: -kept[k]):
            if kept[lab]:
                print(f"  {lab:<9} {level}で保持されるgold: {kept[lab]:>3}/{total[lab]:<4}  例: {ex[lab][:3]}")
        print()


if __name__ == "__main__":
    main()
