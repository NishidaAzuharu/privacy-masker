"""クイックスタート: README の最小構成を NER モデル付きで動かして、主な機能を順に試す。

  PYTHONPATH=. .venv/bin/python examples/quickstart.py
  PYTHONPATH=. .venv/bin/python examples/quickstart.py --model <NER モデルのフォルダか Hub ID> "任意のテキスト"

既定のモデルは Hugging Face Hub で配布している本番モデル（privacy_masker.hub.DEFAULT_MODEL_ID。初回に取得してキャッシュ）。
環境変数 PRIVACY_MASKER_MODEL か --model で差し替えられる（ローカルのフォルダか Hub のリポジトリ ID）。
char_head.safetensors があるフォルダは CharHeadNerDetector、無ければ TransformersNerDetector で読む。
standard レベル（4.）だけ公知データ .pmk が要る（PRIVACY_MASKER_KNOWLEDGE か、データフォルダ PRIVACY_MASKER_DRIVE_DIR 配下の
jawiki_titles_ns0.pmk。無ければその段を飛ばす）。
依存: pip install 'privacy-masker[transformers,crypto]'（vault の往復に crypto が要る。無ければその段だけ飛ばす）。
"""
from __future__ import annotations

import argparse
import os
import sys

from privacy_masker import IdentifiabilityAnnotator, Policy, PrivacyEngine, PublicKnowledge
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector, TransformersNerDetector
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir
from privacy_masker.operators import Surrogate, Tokenize

# 公知データ .pmk の置き場（standard レベル用）。環境変数 PRIVACY_MASKER_DRIVE_DIR で差し替える
DATA_DIR = os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", "~/privacy-masker-data"))
DEFAULT_MODEL = os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID)   # Hub の本番モデル（フォルダか Hub ID で差し替え可）
DEFAULT_KNOWLEDGE = os.environ.get("PRIVACY_MASKER_KNOWLEDGE", f"{DATA_DIR}/jawiki_titles_ns0.pmk")
KEY = b"quickstart-demo-key-please-rotate"   # デモ用。実運用では環境変数などから渡す

DOC = (
    "株式会社大和電機 営業部の山田太郎です。先日の件、弊社の山田までご連絡ください。\n"
    "納品先は埼玉県川口市領家3-7-12 ハイツ領家302、担当は東雲物流の三輪（yuki_t）です。\n"
    "打合せは大江戸記念病院の近くの喫茶店で、来週奈良市へ出張します。\n"
    "TEL 048-222-7815 / taro.yamada@example.jp"
)


def hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def show(result, original: str) -> None:
    print(result.text)
    print("--- entities ---")
    for e in result.entities:
        print(f"  {e.entity_type:<13} {original[e.original_start:e.original_end]!r:<30} -> {e.replacement!r}  ({e.operator})")


def load_ner(model: str):
    try:
        model = resolve_model_dir(model)      # Hub のリポジトリ ID ならここで取得（キャッシュ済みなら再取得しない）
    except ConfigurationError as e:
        sys.exit(f"{e}\n--model か環境変数 PRIVACY_MASKER_MODEL で NER モデルのフォルダか Hub のリポジトリ ID を指定してください")
    if os.path.exists(os.path.join(model, "char_head.safetensors")):
        print(f"CharHeadNerDetector: {model}")
        return CharHeadNerDetector(model=model, device="cpu")
    print(f"TransformersNerDetector: {model}")
    return TransformersNerDetector(model=model, label_map={"PER": "PERSON", "ORG": "ORG", "LOC": "ADDRESS"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default=None, help="処理するテキスト（省略時はデモ文書）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="NER モデルのフォルダか Hugging Face Hub のリポジトリ ID（既定: 環境変数 PRIVACY_MASKER_MODEL、無ければ Hub の本番モデル）")
    ap.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE, help="standard レベル用の公知データ .pmk（無ければその段を飛ばす）")
    args = ap.parse_args()
    doc = args.text or DOC

    detectors = [load_ner(args.model), RegexDetector.default_ja(), HandleDetector()]

    # 1. README のクイックスタート: すべて <TYPE_001> 形式のトークンに
    hr("1. llm_input プリセット（連番トークン）")
    engine = PrivacyEngine(detectors=detectors, policy=Policy.preset("llm_input"))
    show(engine.transform(doc), doc)

    # 2. 意味を保つ偽名への置換（既定 semantic=True）。電話・メール・ハンドル名はトークン
    hr("2. Surrogate（同じカテゴリの実在風の別名。姓リンク: 山田太郎と山田は同じ偽姓）")
    policy = Policy(rules={
        "PERSON":   Surrogate(kind="person", key=KEY),
        "ORG":      Surrogate(kind="org", key=KEY),
        "FACILITY": Surrogate(kind="facility", key=KEY),
        "LOCATION": Surrogate(kind="location", key=KEY),
        "ADDRESS":  Surrogate(kind="address", key=KEY, keep="city"),   # 市区町村は残し、番地以下を置換
    }, default=Tokenize(strategy="sequential"))
    engine = PrivacyEngine(detectors=detectors, policy=policy)
    show(engine.transform(doc), doc)
    print("※ 同じ鍵なら別の文書でも同じ値は同じ偽名になります（hmac）。候補が本文に出ているときだけ引き直します")

    # 3. vault: 偽名から元の値へ復元（LLM に渡して応答を戻す用途）
    hr("3. vault 戦略 + restore()（偽名 → 元の値）")
    try:
        from privacy_masker.crypto import FernetProvider
    except ImportError:
        print("cryptography が無いので飛ばします: pip install 'privacy-masker[crypto]'")
    else:
        policy_v = Policy(rules={
            "PERSON": Surrogate(kind="person", strategy="vault"),
            "ORG":    Surrogate(kind="org", strategy="vault"),
        }, default=Tokenize(strategy="vault"))
        engine_v = PrivacyEngine(detectors=detectors, policy=policy_v, secret_key=KEY,
                                 crypto=FernetProvider(FernetProvider.generate_key()))
        masked = engine_v.transform(doc, scope_id="case_1")
        print(masked.text)
        # LLM の応答を模した文（偽名をそのまま含む）を元に戻す
        reply = f"承知しました。{masked.entities[0].replacement}様の件は担当に共有します。"
        back = engine_v.restore(reply, scope_id="case_1")
        print("--- LLM の応答（模擬） ---\n" + reply)
        print(f"--- restore() ---\n{back.text}   （復元 {back.restored} 件）")

    # 4. standard レベル: 公人・公知の地名/施設は残し、私人の情報だけ匿名化
    hr("4. standard レベル（識別性ベース。公知データが要る）")
    if not os.path.exists(args.knowledge):
        print(f"公知データが無いので飛ばします: {args.knowledge}（tools/build_knowledge.py で生成）")
    else:
        knowledge = PublicKnowledge.load(args.knowledge)
        policy_s = Policy.preset("standard")
        policy_s.rules.update({"PERSON": Surrogate(kind="person", key=KEY),
                               "ADDRESS": Surrogate(kind="address", key=KEY, keep="city")})
        engine_s = PrivacyEngine(detectors=detectors, policy=policy_s,
                                 annotators=[IdentifiabilityAnnotator(level="standard", knowledge=knowledge)])
        doc_s = "安倍元総理は奈良市で演説し、田中さんは新宿駅で待った。"
        show(engine_s.transform(doc_s), doc_s)

    # 5. 検出だけ（自前の後処理をしたいとき）
    hr("5. detect()（変換せず検出結果だけ）")
    for e in engine.detect(doc):
        print(f"  {e.entity_type:<13} [{e.start}:{e.end}] {doc[e.start:e.end]!r}  score={e.score:.2f}  source={e.source}")


if __name__ == "__main__":
    main()
