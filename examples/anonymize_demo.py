"""privacy-masker をライブラリとして使うときの一通り（本文はこのファイル内に書く）。

    cd <repo>
    PYTHONPATH=. .venv/bin/python examples/anonymize_demo.py            # 全部
    PYTHONPATH=. .venv/bin/python examples/anonymize_demo.py 1 3 5      # 節を選んで実行

匿名化したい本文は下の TEXT / RECORDS を書き換えてください。


■ NER モデルと device

既定の NER は Hugging Face Hub の文字ヘッドモデル（privacy_masker.hub.DEFAULT_MODEL_ID。初回に取得してキャッシュ）を
`CharHeadNerDetector` で読みます。環境変数 PRIVACY_MASKER_MODEL でローカルのフォルダか別の Hub ID に差し替えられ、
char_head.safetensors が無いフォルダは `TransformersNerDetector`（token 分類モデル）で読みます。
`CharHeadNerDetector` の device は既定 "cpu"（Apple Silicon なら "mps"、NVIDIA なら "cuda" を指定すると速い）。
`TransformersNerDetector` は device を省略すると transformers が自動で選びます（CUDA → MPS → CPU）。

■ 公知データ（3. standard レベルだけ必要）

環境変数 PRIVACY_MASKER_KNOWLEDGE か、データフォルダ PRIVACY_MASKER_DRIVE_DIR 配下の jawiki_titles_ns0.pmk を読みます
（tools/build_knowledge.py で生成）。無ければ 3. を飛ばします。

■ Refiner（スパン境界の補正）

既定の SpanRefiner は依存ゼロで +0.5 ms/doc、精度も形態素版とほぼ同じなので、そのままで構いません。
形態素情報（サブワード断片のスナップ、社内固有名のユーザー辞書）が必要なら
`pip install 'privacy-masker[mecab]'` を入れて `MorphologicalRefiner()` を渡します（fugashi が自動で使われます）。
"""
from __future__ import annotations

import os
import sys
import time

from privacy_masker import (
    ContextEnhancer, IdentifiabilityAnnotator, Policy, PrivacyEngine, PublicKnowledge, SpanRefiner,
)
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector, TransformersNerDetector
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir
from privacy_masker.operators import (
    Bucketize, Custom, DateShift, Encrypt, FormatPreserving, Hash, Keep, Mask, Surrogate, Tokenize,
    decrypt_text,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
# 公知データ .pmk の置き場。環境変数 PRIVACY_MASKER_DRIVE_DIR で差し替える
DATA_DIR = os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", "~/privacy-masker-data"))
MODEL = os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID)   # Hub の本番モデル（フォルダか Hub ID で差し替え可）
KNOWLEDGE = os.environ.get("PRIVACY_MASKER_KNOWLEDGE", f"{DATA_DIR}/jawiki_titles_ns0.pmk")   # standard レベル用（Wikipedia 見出し語）

SECRET_KEY = b"demo-secret-key-please-rotate-in-production"   # 本番は環境変数や秘密管理から
ENCRYPT_KEY = Encrypt.generate_key()                          # 本番は保管して再利用（毎回変えると復号できない）

# ---------------------------------------------------------------------------
# 匿名化したい本文（ここを書き換える）
# ---------------------------------------------------------------------------
TEXT = """
株式会社ミナセ電装の本社は新宿区西新宿にあり、代表取締役は三輪亮介です。問い合わせは048-222-7815まで。
"""

TEXT_PUBLIC = "岸田首相は新宿駅で演説した。同日、田中さんは西新宿五丁目駅の近くでトヨタ自動車の販売店を訪れた。"

RECORDS = [
    {"id": 1, "name": "三輪 亮介", "tel": "048-222-7815", "memo": "配送先変更の連絡。担当は広瀬あかね。", "age": 34},
    {"id": 2, "name": "桐谷 洸", "tel": "048-640-2200", "memo": "見積の返信待ち", "age": 51},
    {"id": 3, "name": "轟 与一", "tel": "0263-47-2210", "memo": "", "age": 88},
    {"id": 4, "name": "百瀬 いずみ", "tel": "090-4471-2085", "memo": None, "age": 42},
]


# ---------------------------------------------------------------------------
# 共通の部品
# ---------------------------------------------------------------------------
def build_detectors(model: str = MODEL):
    """本番構成の検出器。モデル読み込みが一番重いので、作ったら使い回す。

    model はフォルダか Hub のリポジトリ ID。char_head.safetensors があれば文字ヘッド（CharHeadNerDetector、cpu）、
    無ければ token 分類モデル（TransformersNerDetector、device は自動選択）。
    """
    path = resolve_model_dir(model)
    if os.path.exists(os.path.join(path, "char_head.safetensors")):
        ner = CharHeadNerDetector(model=path, device="cpu", batch_size=16)
        print(f"  [検出器] CharHeadNerDetector model={model} device={ner.device}")
    else:
        ner = TransformersNerDetector(model=path, batch_size=16)   # ← device 省略で自動選択
        print(f"  [検出器] TransformersNerDetector model={model} device={ner._pipeline.device}（自動選択）")
    return [ner, RegexDetector.default_ja(dates=True), HandleDetector()]
    # dates=True で DATE / AGE も拾う。日付が大量に出る文書では False（既定）に


def show(title: str, result, original: str, limit: int = 24) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(result.text)
    print(f"--- 検出・置換 {len(result.entities)} 件 ---")
    for e in result.entities[:limit]:
        src = original[e.original_start:e.original_end].replace("\n", "⏎")
        print(f"  {e.entity_type:<17} {src!r:<38} → {e.replacement!r:<22} [{e.operator}]")
    if len(result.entities) > limit:
        print(f"  … 他 {len(result.entities) - limit} 件")


# ---------------------------------------------------------------------------
# 1. 最短の使い方（全部トークン化）— LLM への入力を作るときの既定
# ---------------------------------------------------------------------------
def demo1_strict(detectors) -> None:
    engine = PrivacyEngine(
        detectors=detectors,
        policy=Policy.preset("strict"),   # 未知の型も default でトークン化される（素通しさせない）
        refiner=SpanRefiner(),            # 既定なので省略可
    )
    t0 = time.perf_counter()
    result = engine.transform(TEXT)
    show(f"1. strict: 全部トークン化（{(time.perf_counter() - t0) * 1000:.0f} ms）", result, TEXT)


# ---------------------------------------------------------------------------
# 2. 自然な仮名に置換（分布を保ったまま匿名化したいとき）
# ---------------------------------------------------------------------------
def demo2_surrogate(detectors) -> None:
    policy = Policy(
        rules={
            "PERSON":   Surrogate(kind="person"),               # 実在風の別人名（姓のみ／フルネームの連携あり）
            "ORG":      Surrogate(kind="org"),
            "FACILITY": Surrogate(kind="org"),
            "ADDRESS":  Surrogate(kind="address", keep="city"),  # 市区町村までは残す
            "LOCATION": Keep(),                                  # 地名はそのまま
            "DATE":     DateShift(max_days=365),                 # 全日付を同じ日数ずらす（順序と間隔は保たれる）
            "AGE":      Bucketize(10),                           # 34歳 → 30-39歳
            "PHONE_NUMBER": FormatPreserving(),                  # 桁数と形式を保った別番号
            "EMAIL":    Hash(length=12, template="user_{hash}@example.com"),
        },
        default=Tokenize(strategy="sequential"),                 # 上に無い型（URL・法人番号等）はトークン
    )
    engine = PrivacyEngine(detectors=detectors, policy=policy, secret_key=SECRET_KEY)
    show("2. surrogate: 自然な仮名（同じ鍵なら毎回同じ置換）", engine.transform(TEXT), TEXT)


# ---------------------------------------------------------------------------
# 3. 識別性ベース（公人・公知の施設は残し、私人に紐づく情報だけ匿名化）
# ---------------------------------------------------------------------------
def demo3_standard(detectors) -> None:
    if not os.path.exists(KNOWLEDGE):
        print(f"\n[3 をスキップ] 公知データが無い: {KNOWLEDGE}\n"
              f"  作り方: PYTHONPATH=. .venv/bin/python tools/build_knowledge.py")
        return
    print("\n  [公知データ] 読み込み中…", end="", flush=True)
    knowledge = PublicKnowledge.load(KNOWLEDGE)
    print(f" {len(knowledge):,} 件")
    engine = PrivacyEngine(
        detectors=detectors,
        policy=Policy.preset("standard"),
        annotators=[IdentifiabilityAnnotator(level="standard", knowledge=knowledge)],
    )
    show("3. standard: 公知は保持、私人と私人に帰属するものは匿名化",
         engine.transform(TEXT_PUBLIC), TEXT_PUBLIC)
    # level="minimal" なら PERSON だけ匿名化して地名・組織は全部残す
    # level="strict"（既定）は Annotator を無効化＝全部匿名化


# ---------------------------------------------------------------------------
# 4. 呼び出し時オプション（対象型の限定・閾値・残す語・必ず消す語・判定理由）
# ---------------------------------------------------------------------------
def demo4_options(detectors) -> None:
    engine = PrivacyEngine(detectors=detectors, policy=Policy.preset("strict"))

    r = engine.transform(TEXT, entities=["PERSON", "ADDRESS", "PHONE_NUMBER"])
    print(f"\n{'=' * 78}\n4-a. entities= で対象型を限定\n{'=' * 78}")
    print(f"  検出された型: {sorted({e.entity_type for e in r.entities})}")

    r = engine.transform(
        TEXT,
        allow_list=["株式会社ミナセ電装", "東雲物流株式会社"],   # 取引先名は残す
        deny_list={"MD-220": "PRODUCT_CODE"},                    # 型番は必ず消す
    )
    print(f"\n4-b. allow_list / deny_list")
    print(f"  「株式会社ミナセ電装」は残った: {'株式会社ミナセ電装' in r.text}")
    print(f"  「MD-220」は消えた: {'MD-220' not in r.text}")
    print(f"  PRODUCT_CODE として置換: {[e.replacement for e in r.entities if e.entity_type == 'PRODUCT_CODE']}")

    # 文脈語でスコアを上げてから閾値で切る（過剰マスクを減らす）
    strict_engine = PrivacyEngine(
        detectors=detectors, policy=Policy.preset("strict"),
        enhancer=ContextEnhancer(),          # 「氏名」「TEL」等が近くにあれば +0.3
        score_threshold={"DEFAULT": 0.5, "PERSON": 0.7},
    )
    print(f"\n4-c. ContextEnhancer + score_threshold: 検出 {len(strict_engine.transform(TEXT).entities)} 件 "
          f"(閾値なし: {len(engine.transform(TEXT).entities)} 件)")

    print(f"\n4-d. explain=True で判定理由（原文の文字は含まれない）")
    for e in engine.transform(TEXT, explain=True).entities[:6]:
        print(f"  {e.entity_type:<15} {e.explanation}")


# ---------------------------------------------------------------------------
# 5. 復元できる匿名化（2 通り）
# ---------------------------------------------------------------------------
def demo5_reversible(detectors) -> None:
    from privacy_masker.crypto import FernetProvider
    from privacy_masker.stores import MemoryStore, SQLiteStore   # noqa: F401（永続化は SQLiteStore(path)）

    # (a) vault: トークンを置き、原文は暗号化して Store に保存
    engine = PrivacyEngine(
        detectors=detectors, policy=Policy(default=Tokenize(strategy="vault")),
        secret_key=SECRET_KEY, crypto=FernetProvider(ENCRYPT_KEY), store=MemoryStore(),
    )
    masked = engine.transform(TEXT, scope_id="case-001")
    restored = engine.restore(masked.text, scope_id="case-001")
    print(f"\n{'=' * 78}\n5. 復元できる匿名化\n{'=' * 78}")
    print(f"  (a) vault  : 匿名化 → restore で完全一致 {restored.text == TEXT}（未解決 {len(restored.unresolved)} 件）")

    # (b) Encrypt: 暗号文を本文に埋める（Store 不要、鍵だけで戻せる）
    engine2 = PrivacyEngine(detectors=detectors, policy=Policy(default=Encrypt(ENCRYPT_KEY)))
    enc = engine2.transform(TEXT)
    print(f"  (b) Encrypt: 匿名化 → decrypt_text で完全一致 {decrypt_text(enc.text, ENCRYPT_KEY) == TEXT}")


# ---------------------------------------------------------------------------
# 6. まとめて処理（バッチ）と構造化データ
# ---------------------------------------------------------------------------
def demo6_batch_and_records(detectors) -> None:
    from privacy_masker.structured import transform_records

    engine = PrivacyEngine(
        detectors=detectors,
        policy=Policy(rules={"PERSON": Surrogate(kind="person"), "PHONE_NUMBER": Mask(keep_suffix=4)},
                      default=Tokenize()),
        secret_key=SECRET_KEY,
    )
    print(f"\n{'=' * 78}\n6. バッチと構造化データ\n{'=' * 78}")

    texts = [TEXT, TEXT_PUBLIC, "問い合わせ: 百瀬いずみ（izumi.momose@example.jp）"]
    t0 = time.perf_counter()
    results = engine.transform_batch(texts, batch_size=32)   # 大量処理はこちら（逐次より約 2 倍速い）
    print(f"  transform_batch: {len(results)} 件を {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"  3 件目 → {results[2].text}")

    rows = transform_records(engine, RECORDS)   # 列ごとに型を推定し、氏名列などは列全体を置換
    print(f"\n  transform_records（{len(rows)} 行）:")
    for before, after in zip(RECORDS, rows):
        print(f"    {before['name']:<12} → {after['name']:<12} | {before['tel']} → {after['tel']} | memo: {after['memo']!r}")
    # pandas なら: from privacy_masker.structured import transform_dataframe; transform_dataframe(engine, df)


# ---------------------------------------------------------------------------
# 7. 変換せず検出だけ（自前の後処理をしたいとき）
# ---------------------------------------------------------------------------
def demo7_detect_only(detectors) -> None:
    engine = PrivacyEngine(detectors=detectors, policy=Policy.preset("strict"))
    print(f"\n{'=' * 78}\n7. detect() / explain()：検出だけ\n{'=' * 78}")
    for e in engine.detect(TEXT)[:8]:
        print(f"  {e.entity_type:<17} [{e.start:>4}:{e.end:>4}] score={e.score:.2f} source={e.source:<17} {TEXT[e.start:e.end]!r}")
    print(f"  … 全 {len(engine.detect(TEXT))} 件")


DEMOS = {
    "1": ("strict（全部トークン化）", demo1_strict),
    "2": ("surrogate（自然な仮名）", demo2_surrogate),
    "3": ("standard（公知は保持）", demo3_standard),
    "4": ("呼び出しオプション", demo4_options),
    "5": ("復元できる匿名化", demo5_reversible),
    "6": ("バッチ・構造化データ", demo6_batch_and_records),
    "7": ("検出だけ", demo7_detect_only),
}


def main() -> None:
    want = sys.argv[1:] or list(DEMOS)
    unknown = [w for w in want if w not in DEMOS]
    if unknown:
        print(f"不明な節: {unknown}\n使える節: " + ", ".join(f"{k}={v[0]}" for k, v in DEMOS.items()))
        raise SystemExit(2)
    print("privacy-masker デモ（本文はこのファイルの TEXT / RECORDS）")
    detectors = build_detectors()
    for key in want:
        _title, fn = DEMOS[key]
        fn(detectors)


if __name__ == "__main__":
    main()
