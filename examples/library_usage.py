"""ライブラリとして使うときの最小構成（本番設定）を、公開 API だけで組んで試す。

  PYTHONPATH=. .venv/bin/python examples/library_usage.py

構成 = CharHeadNerDetector（Hugging Face Hub の本番モデル。初回に取得してキャッシュ）+ RegexDetector.default_ja() + HandleDetector()
       + MorphologicalRefiner()（境界補正）+ 任意で IdentifiabilityAnnotator（公知エンティティの保持。公知データ .pmk が要る）
NER モデルは環境変数 PRIVACY_MASKER_MODEL（ローカルのフォルダか Hub のリポジトリ ID）で差し替えられる。
char_head.safetensors があるフォルダは CharHeadNerDetector、無ければ TransformersNerDetector で読む。
公知データ .pmk は PRIVACY_MASKER_KNOWLEDGE か、データフォルダ PRIVACY_MASKER_DRIVE_DIR 配下の jawiki_titles_ns0.pmk（無ければ D. を飛ばす）。
"""
from __future__ import annotations

import os
import time

from privacy_masker import IdentifiabilityAnnotator, MorphologicalRefiner, Policy, PrivacyEngine, PublicKnowledge
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector, TransformersNerDetector
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir
from privacy_masker.operators import Surrogate, Tokenize

# 公知データ .pmk の置き場。環境変数 PRIVACY_MASKER_DRIVE_DIR で差し替える
DATA_DIR = os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", "~/privacy-masker-data"))
MODEL = os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID)   # 本番 NER（Hub の文字ヘッドモデル）
KNOWLEDGE = os.environ.get("PRIVACY_MASKER_KNOWLEDGE", f"{DATA_DIR}/jawiki_titles_ns0.pmk")   # standard レベル用の公知データ（Wikipedia 見出し語）


def load_ner(model: str):
    """フォルダか Hub のリポジトリ ID を受け取り、char_head.safetensors の有無で検出器を選ぶ。"""
    path = resolve_model_dir(model)
    if os.path.exists(os.path.join(path, "char_head.safetensors")):
        return CharHeadNerDetector(model=path, device="cpu")
    return TransformersNerDetector(model=path, device="cpu", label_map={"PER": "PERSON", "ORG": "ORG", "LOC": "ADDRESS"})


# 検出器は 1 回作って使い回す（モデル読み込みが一番重い）
ner = load_ner(MODEL)
detectors = [ner, RegexDetector.default_ja(), HandleDetector()]
refiner = MorphologicalRefiner()

DOC = """件名: 見積依頼（MD-220 300台）
株式会社ミナセ電装 営業部 桐谷 様

東雲物流株式会社 調達課の三輪です。納品先は埼玉県川口市領家3-7-12 ハイツ領家 302、
希望納期は10月末です。担当 yuki_t にも共有済みです。
TEL 048-222-7815 / r.miwa@shinonome-lg.example.jp
"""


def show(title, result, original):
    print(f"\n===== {title} =====")
    print(result.text)
    print("--- entities ---")
    for e in result.entities:
        print(f"  {e.entity_type:<13} {original[e.original_start:e.original_end]!r:<34} -> {e.replacement!r}  ({e.operator})")


# A. strict: 全部トークン化（LLM 入力・既定の推奨）
engine = PrivacyEngine(detectors=detectors, policy=Policy.preset("strict"), refiner=refiner)
t0 = time.perf_counter(); r = engine.transform(DOC); dt = time.perf_counter() - t0
show(f"A. strict（Tokenize sequential） {dt*1000:.0f} ms", r, DOC)

# B. dataset: HMAC トークン。同じ鍵なら文書をまたいで同一人物が同一トークン
engine_b = PrivacyEngine(detectors=detectors, policy=Policy.preset("dataset"), refiner=refiner,
                         secret_key=b"demo-secret-key-please-rotate")
r1 = engine_b.transform("三輪さんから連絡があった。")
r2 = engine_b.transform("先方の担当は三輪です。折り返しは桐谷まで。")
print("\n===== B. dataset（Tokenize hmac、同じ鍵なら文書をまたいで同じ表層 = 同じトークン） =====")
print(r1.text); print(r2.text, "  ← 「三輪」は別文書でも同じトークン。表層単位なので「三輪 亮介」は別トークンになる")

# C. surrogate: 実在風の別名に置換（分布保存）。住所は市区町村まで残す
policy_c = Policy(rules={
    "PERSON":   Surrogate(kind="person"),
    "ORG":      Surrogate(kind="org"),
    "FACILITY": Surrogate(kind="org"),
    "ADDRESS":  Surrogate(kind="address", keep="city"),
}, default=Tokenize(strategy="sequential"))   # 電話・メール・ハンドル等はトークン
engine_c = PrivacyEngine(detectors=detectors, policy=policy_c, refiner=refiner, secret_key=b"demo-secret-key-please-rotate")
r = engine_c.transform(DOC)
show("C. surrogate（自然な仮名、住所は市まで保持）", r, DOC)

# D. standard: 識別性ベース。公知のエンティティ（公人・有名施設・地名）は残し、私人に帰属するものだけ匿名化
if os.path.exists(KNOWLEDGE):
    knowledge = PublicKnowledge.load(KNOWLEDGE)
    annot = IdentifiabilityAnnotator(level="standard", knowledge=knowledge)
    engine_d = PrivacyEngine(detectors=detectors, policy=Policy.preset("standard"), refiner=refiner, annotators=[annot])
    DOC_D = "岸田首相は新宿駅前で演説した。同日、田中さんは西新宿五丁目駅の近くのトヨタ自動車の販売店で契約した。"
    r = engine_d.transform(DOC_D)
    show("D. standard（公知は保持、私人の準識別子は匿名化）", r, DOC_D)
else:
    print(f"\n===== D. standard =====\n公知データが無いので飛ばします: {KNOWLEDGE}（tools/build_knowledge.py で生成、PRIVACY_MASKER_KNOWLEDGE で指定）")

# E. detect() だけ使う（自前の後処理をしたい場合）
print("\n===== E. detect()（変換せず検出結果だけ） =====")
for e in engine.detect("担当: 桐谷 洸（070-2233-8109）")[:6]:
    print(f"  {e.entity_type:<13} [{e.start}:{e.end}] score={e.score:.2f} source={e.source}")

# F. vault: 復元可能なトークン化（scope_id 単位）。暗号化した原文を Store に保存するので CryptoProvider と Store が必要
from privacy_masker.crypto import FernetProvider     # pip install 'privacy-masker[crypto]'
from privacy_masker.stores import MemoryStore        # 永続化するなら SQLiteStore(path)

engine_f = PrivacyEngine(detectors=detectors, policy=Policy(default=Tokenize(strategy="vault")), refiner=refiner,
                         secret_key=b"demo-secret-key-please-rotate",            # トークン生成用
                         crypto=FernetProvider(FernetProvider.generate_key()),  # 原文の暗号化用（本番は鍵を保管）
                         store=MemoryStore())
r = engine_f.transform("連絡先は三輪 亮介、048-222-7815。", scope_id="case-001")
back = engine_f.restore(r.text, scope_id="case-001")
print("\n===== F. vault（復元可能） =====")
print(r.text); print(back.text, "  unresolved:", back.unresolved)
