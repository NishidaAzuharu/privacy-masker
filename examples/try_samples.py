"""任意のサンプル文で検出と変換を試す。サンプルはこのファイルの SAMPLES に書く（引数で差し替えも可）。

  PYTHONPATH=. .venv/bin/python examples/try_samples.py                       # SAMPLES を全部
  PYTHONPATH=. .venv/bin/python examples/try_samples.py "任意の文" "もう 1 文"   # 引数の文だけ
  PYTHONPATH=. .venv/bin/python examples/try_samples.py --explain --threshold 0.6
  PYTHONPATH=. .venv/bin/python examples/try_samples.py --no-ner               # 正規表現 + ハンドル名だけ（モデル不要）

構成は README の本番構成（NER + RegexDetector.default_ja() + HandleDetector + 既定の SpanRefiner）。
NER モデルは Hub の本番モデル（privacy_masker.hub.DEFAULT_MODEL_ID）。環境変数 PRIVACY_MASKER_MODEL か --model で
ローカルのフォルダか Hub のリポジトリ ID に差し替えられる。char_head.safetensors があるフォルダは CharHeadNerDetector、
無ければ TransformersNerDetector（ラベルの対応は --label-map "PER=PERSON,ORG=ORG,LOC=LOCATION" の形で指定）。
依存: pip install 'privacy-masker[transformers]'（--no-ner なら core だけで動く）。
"""
from __future__ import annotations

import argparse
import os
import sys

from privacy_masker import Policy, PrivacyEngine
from privacy_masker.detectors import HandleDetector, RegexDetector

DEFAULT_MODEL = os.environ.get("PRIVACY_MASKER_MODEL")   # None なら Hub の本番モデル

# ---- ここにサンプルを書く。1 要素 = 1 文書（複数行可）。コメントは何を見たいか ----
SAMPLES = [
    # 人名・組織。同じ姓の再出現（山田太郎 / 山田）が同じトークンになるか
    "株式会社大和電機 営業部の山田太郎です。先日の件、弊社の山田までご連絡ください。",
    # 番地付き住所・組織・姓・括弧内のハンドル名
    "納品先は埼玉県川口市領家3-7-12 ハイツ領家302、担当は東雲物流の三輪（yuki_t）です。",
    # 正規表現: 電話・メール・郵便番号
    "TEL 048-222-7815 / taro.yamada@example.jp / 〒343-0824 まで。",
    # 文脈語あり（個人番号・登録番号の T・カード）→ score 1.0
    "個人番号は 1234 5678 9018、登録番号は T9234567890123、カードは 4111-1111-1111-1111 です。",
    # 文脈語なしで検査数字だけ合う数字列 → score 0.5（--threshold 0.6 で外れる）
    "伝票番号 123456789018 と受付 4111 1111 1111 1111 を確認。",
    # URL・メールの内側の数字は分断されず URL / EMAIL として一括
    "詳細は https://example.com/?card=4111111111111111 を参照。送付先は 123456789018@example.com。",
    # 敬称（さん）の切り分け。tokenizer が「田中さん」を 1 ピースにする文
    "お世話になっております。田中さんに確認します。佐藤さんと鈴木さんにも連絡しました。",
    # チャット: 発言者名・@ハンドル・ひらがなの名前
    "[09:12] けんた: 青葉駅の改札に集合で！ @りさ 遅れるなら連絡して",
    # 全角の電話番号と登録番号
    "次回連絡は０９０－１２３４－５６７８へ。登録番号 Ｔ９２３４５６７８９０１２３。",
    # ログ形式: key=value の中の人名・ハンドル・IP
    "operator=真壁 action=login user=yuki_t ip=192.168.1.10 result=ok",
    # 個人情報を含まない文（誤検出が無いことの確認）
    "本日は晴天なり。会議は 14 時から第 2 会議室で行います。議題は来期の予算です。",
]


def parse_label_map(spec: str | None) -> dict[str, str]:
    """"PER=PERSON,ORG=ORG" → {"PER": "PERSON", "ORG": "ORG"}"""
    out: dict[str, str] = {}
    for item in (spec or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_ner(model: str | None, device: str, label_map: dict[str, str]):
    from privacy_masker.detectors import CharHeadNerDetector, TransformersNerDetector
    from privacy_masker.exceptions import ConfigurationError
    from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir
    model = model or DEFAULT_MODEL_ID
    try:
        path = resolve_model_dir(model)      # Hub のリポジトリ ID ならここで取得（キャッシュ済みなら再取得しない）
    except ConfigurationError as e:
        sys.exit(f"{e}\n--model か環境変数 PRIVACY_MASKER_MODEL で NER モデルのフォルダか Hub のリポジトリ ID を指定してください")
    if os.path.exists(os.path.join(path, "char_head.safetensors")):
        print(f"NER: CharHeadNerDetector({model})")
        return CharHeadNerDetector(model=path, device=device)
    print(f"NER: TransformersNerDetector({model}) label_map={label_map or '{}'}")
    return TransformersNerDetector(model=path, label_map=label_map, device=None if device == "cpu" else device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("texts", nargs="*", help="試す文。省略時は SAMPLES")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="NER モデルのフォルダか Hub のリポジトリ ID（既定: Hub の本番モデル）")
    ap.add_argument("--label-map", default="PER=PERSON,PERSON=PERSON,ORG=ORG,LOC=LOCATION,LOCATION=LOCATION",
                    help="TransformersNerDetector 用のラベル対応（CharHead モデルでは不要）")
    ap.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    ap.add_argument("--preset", default="llm_input", help="Policy.preset の名前（llm_input / strict / dataset）")
    ap.add_argument("--threshold", type=float, default=None, help="score_threshold（例 0.6: 文脈語の無い番号を外す）")
    ap.add_argument("--no-ner", action="store_true", help="NER を使わず正規表現 + ハンドル名だけ")
    ap.add_argument("--no-refiner", action="store_true", help="SpanRefiner を外す")
    ap.add_argument("--explain", action="store_true", help="各エンティティの判定の経緯（検出器・スコア・Refiner・重複解決）を出す")
    args = ap.parse_args()

    detectors = []
    if not args.no_ner:
        detectors.append(load_ner(args.model, args.device, parse_label_map(args.label_map)))
    detectors += [RegexDetector.default_ja(), HandleDetector()]
    kw = {"refiner": None} if args.no_refiner else {}
    engine = PrivacyEngine(detectors=detectors, policy=Policy.preset(args.preset), **kw)

    texts = args.texts or SAMPLES
    for i, text in enumerate(texts, 1):
        result = engine.transform(text, score_threshold=args.threshold, explain=args.explain)
        print(f"\n[{i}] {text}")
        print(f"    → {result.text}")
        if not result.entities:
            print("    （検出なし）")
        for e in result.entities:
            surface = text[e.original_start:e.original_end]
            line = f"    {e.entity_type:<16} {surface!r:<34} -> {e.replacement!r}  score={e.score:.2f} ({e.operator})"
            if args.explain and e.explanation:
                line += f"\n        {e.explanation}"
            print(line)


if __name__ == "__main__":
    main()
