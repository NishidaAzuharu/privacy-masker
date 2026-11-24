"""識別性ベース匿名化の安全コスト測定（ライブラリ実装 IdentifiabilityAnnotator を使用）。

各評価セットの gold エンティティのうち、standard レベルで「保持」されるものを数える。
  私人文書での保持 = 見逃し（安全コスト）。ニュースでの保持 = 望ましい（公知を残す）。
実行（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python -W ignore tools/eval_identifiability.py
"""
import argparse
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

GD = drive_dir(require=False)
MODEL = os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID)   # モデルフォルダか Hub ID（第 2 引数でも指定可）
SETS = [
    ("私人: eval_ood_v1(8ジャンル)", [f"{GD}/eval_ood_v1.jsonl"]),
    ("私人: eval_real+hard", [f"{GD}/eval_real_v1.jsonl", f"{GD}/eval_hard_v1.jsonl"]),
    ("私人: 合成文書(282)", [f"{GD}/synthetic_pii_claude_v1.jsonl"]),
    ("公知: wikinews(178)", [f"{GD}/eval_ood_news_v1.jsonl"]),
]


def _masked_chars(entities):
    all_chars, masked = set(), set()
    for entity in entities:
        span = range(entity.start, entity.end)
        all_chars.update(span)
        if not entity.entity_type.endswith("_PUBLIC"):
            masked.update(span)
    return all_chars, masked


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="識別性判定の保持件数を比較（本文・表層は出力しない）")
    parser.add_argument("level", nargs="?", default="standard", choices=("strict", "standard", "minimal"))
    parser.add_argument("model", nargs="?", default=MODEL)
    parser.add_argument("--person-role-head", action=argparse.BooleanOptionalAction, default=False,
                        help="役割ヘッドを必須ロードし、規則だけと規則 + ヘッドの保持件数を比較")
    parser.add_argument("--person-role-threshold", type=float, default=None, help="役割ヘッドの閾値を上書き")
    args = parser.parse_args(argv)
    if args.person_role_head and args.level != "standard":
        parser.error("--person-role-head は standard でのみ使用できます")
    if args.person_role_threshold is not None:
        if not args.person_role_head or not 0 <= args.person_role_threshold <= 1:
            parser.error("--person-role-threshold は --person-role-head と [0, 1] の値が必要です")
    model = resolve_model_dir(args.model)
    is_char_head = os.path.exists(os.path.join(model, "char_head_config.json"))
    if args.person_role_head and not is_char_head:
        parser.error("--person-role-head は char_head モデルが必要です")
    knowledge = PublicKnowledge.load(f"{GD}/jawiki_titles_ns0.pmk")
    ann = IdentifiabilityAnnotator(level=args.level, knowledge=knowledge)
    det = (CharHeadNerDetector(model=model, device="cpu", person_role_head=args.person_role_head)
           if is_char_head else TransformersNerDetector(model=model, device="cpu"))
    role_ann = (IdentifiabilityAnnotator(level=args.level, knowledge=knowledge,
                                       person_role=det.person_role_head,
                                       person_role_threshold=args.person_role_threshold)
                if args.person_role_head else None)
    rx = RegexDetector.default_ja()
    ref = MorphologicalRefiner()
    for name, paths in SETS:
        docs = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                docs.extend(json.loads(line) for line in f)
        preds = det.detect_batch([d["text"] for d in docs])
        kept, kept_role, total = defaultdict(int), defaultdict(int), defaultdict(int)
        fp_old = fp_new = fp_role = 0
        for d, raw in zip(docs, preds):
            text = d["text"]
            refined = ref.refine(text, raw + rx.detect(text))
            masked_old, masked_new = _masked_chars(ann.annotate(text, refined))
            masked_role = _masked_chars(role_ann.annotate(text, refined))[1] if role_ann else set()
            gold_all = set()
            for start, end, label in d["entities"]:
                span = set(range(start, end))
                gold_all |= span
                total[label] += 1
                if len(span & masked_old) / len(span) >= 0.5:
                    if len(span & masked_new) / len(span) < 0.5:
                        kept[label] += 1
                    if role_ann and len(span & masked_role) / len(span) < 0.5:
                        kept_role[label] += 1
            neutral = set()
            for start, end in d.get("neutral", []):
                neutral.update(range(start, end))
            fp_old += len(masked_old - gold_all - neutral)
            fp_new += len(masked_new - gold_all - neutral)
            fp_role += len(masked_role - gold_all - neutral)
        print(f"=== {name}: 文書{len(docs)}件 ===")
        comparison = f" / 規則 + ヘッド {fp_role}" if role_ann else ""
        print(f"  過剰マスク文字数: strict {fp_old} / 規則だけ {fp_new}{comparison}")
        for label in sorted(total, key=lambda k: -kept[k]):
            comparison = f" / 規則 + ヘッド: {kept_role[label]}/{total[label]}" if role_ann else ""
            print(f"  {label:<9} 保持gold 規則だけ: {kept[label]}/{total[label]}{comparison}")
        print()


if __name__ == "__main__":
    main()
