"""semantic=True の hmac 出力をスナップショットに凍結する（ja_pools / 生成器 / フィルタの変更検知）。

usage: PYTHONPATH=. .venv/bin/python tools/freeze_surrogate_golden.py [--out tests/golden/surrogate_semantic_v0_10.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from privacy_masker.encoding import hmac_token_id  # noqa: E402
from privacy_masker.exceptions import StoreError  # noqa: E402
from privacy_masker.operators import Surrogate  # noqa: E402
from privacy_masker.types import Entity  # noqa: E402

KEY = b"golden-surrogate-key"
CASES = [
    ("person", "山田太郎"), ("person", "山田"), ("person", "佐藤 花子"), ("person", "ミゲル・サントス"), ("person", "ながしま"),
    ("person", "John Smith"),
    # 裸姓の文字種保存（Important I4）: 平仮名・ローマ字・半角カナは漢字の姓リンク経路に入れない
    ("person", "たなか"), ("person", "Ito"), ("person", "ｻﾄｳ"),
    ("org", "株式会社大和技研"), ("org", "西日本あおば信用金庫"), ("org", "アオバシステムズ株式会社"),
    ("org", "営業二課"), ("org", "医療法人伊織会"), ("facility", "汐見クリニック"), ("facility", "市立若草小学校"), ("facility", "汐見駅"),
    ("location", "大分県"), ("location", "札幌市"), ("location", "パリ"), ("location", "西新宿2丁目"),
    ("address", "東京都新宿区西新宿1-2-3"), ("address", "神奈川県横浜市中区太田町５丁目８番地 サンハイツ301"),
    # v0.9 保留 Minor 群の解消（Task 1〜5）で変わった経路を覆う追加ケース
    ("person", "スズキ タロウ"),  # カナ氏名フルネーム・名が男性寄り（性別プールの偶奇＋GENDER_OVERRIDES）
    ("person", "マイケルジョンソン"),  # 区切り無しカタカナ外国人名（structure=full として公知照合の対象になる）
    ("location", "函館市"),  # 北海道の市（PREF_CITIES_EXTRA、semantic 経路専用の拡張プール）
    # v0.10.0: 規則カテゴリ判定の癖の修正（設計は開発記録、非公開）
    ("org", "青葉電気鉄道"), ("org", "みなと商事"), ("org", "若草の会"), ("org", "青葉開発機構"), ("org", "汐見航空局"),
    ("org", "若葉ホークス"), ("org", "青葉通信社"), ("org", "若葉県教育委員会"), ("org", "青葉祭実行委員会"),
    ("location", "若葉地区"), ("location", "青葉1区"), ("location", "ミナト市"), ("location", "イラク"), ("location", "上海"),
]


class Ctx:
    scope_id = "golden"
    text = None

    def __init__(self):
        self.tokens = {}

    def next_seq(self, t):
        raise AssertionError

    def get_token(self, t, n):
        return self.tokens.get((t, n))

    def save_token(self, t, n, token, original_value=None):
        if token in self.tokens.values():
            raise StoreError("dup")
        self.tokens[(t, n)] = token

    def hmac_token(self, t, n, key=None, length=8):
        return hmac_token_id(key or KEY, self.scope_id, t, n, length)


TYPE = {"person": "PERSON", "org": "ORG", "facility": "FACILITY", "location": "LOCATION", "address": "ADDRESS"}


def compute() -> list[dict]:
    out = []
    for kind, value in CASES:
        op = Surrogate(kind=kind, strategy="hmac", key=KEY)
        out.append({"kind": kind, "value": value, "expected": op.apply(value, Entity(TYPE[kind], 0, len(value), 1.0), Ctx())})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("tests", "golden", "surrogate_semantic_v0_10.json"))
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {"version": "0.10", "key": KEY.decode(), "cases": compute()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for c in data["cases"]:
        print(f"{c['kind']:9s} {c['value']!r:32} -> {c['expected']!r}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
