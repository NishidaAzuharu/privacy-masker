"""privacy-masker お試しスクリプト（主要機能のデモ + コマンドラインからのマスク実行）。

実行方法（リポジトリ直下で）:
    PYTHONPATH=. .venv/bin/python examples/try_privacy_masker.py                       # 引数なし → デモ5本
    PYTHONPATH=. .venv/bin/python examples/try_privacy_masker.py "大和技研の山田です"   # テキストを直接マスク
    PYTHONPATH=. .venv/bin/python examples/try_privacy_masker.py --file input.txt      # ファイル（1行=1文書）をバッチでマスク
    echo "山田です" | PYTHONPATH=. .venv/bin/python examples/try_privacy_masker.py -   # 標準入力からマスク

オプション:
    --model PATH      NERモデルのフォルダか Hugging Face Hub のリポジトリ ID（既定: 環境変数 PRIVACY_MASKER_MODEL、
                      無ければ Hub で配布している本番モデル privacy_masker.hub.DEFAULT_MODEL_ID。初回に取得してキャッシュ）。
                      char_head.safetensors があるディレクトリは CharHeadNerDetector、なければ TransformersNerDetector で読む。
    --device DEV      cpu / mps / cuda（既定: 文字ヘッドモデルは cpu、token モデルは transformers の自動選択）
    --key STR         hmacの鍵（既定: デモ用固定鍵。同じ鍵なら同一人物=同一偽名）
    --show-entities   検出内訳（位置・スコア・operator）も表示
    --level LV        strict（既定・全エンティティを匿名化）/ standard（公人・公知は残す）/ minimal
    --knowledge PATH  公知データ .pmk（既定: データフォルダ PRIVACY_MASKER_DRIVE_DIR 配下の jawiki_titles_ns0.pmk。standard/minimal で使用）
    --out PATH        マスク結果をファイルにも保存
    --no-bench        （デモ時）バッチスループット計測をスキップ

デモ内容:
    1. Surrogate仮名化（実在風の日本人名・企業名・住所へ置換、姓リンク付き）
    2. hmac決定性（別々の呼び出しでも同一人物は同一偽名）
    3. Tokenize(vault) + restore（可逆トークン化と復元）
    4. 正規表現検出の併用（メール・電話 → トークン化）
    5. バッチ処理のスループット（transform逐次 vs transform_batch）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# インストールせずリポジトリ直下から実行できるようにするための保険
try:
    import privacy_masker  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_masker import IdentifiabilityAnnotator, PrivacyEngine, Policy, PublicKnowledge
from privacy_masker.detectors import (
    CharHeadNerDetector, HandleDetector, RegexDetector, TransformersNerDetector,
)
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.hub import DEFAULT_MODEL_ID, resolve_model_dir
from privacy_masker.operators import Surrogate, Tokenize

# 形態素解析器（fugashi か janome）があれば語境界スナップ Refiner を使う: 形態素境界へのスパン拡張・
# 断片誤検出の棄却により、医療・帳票等の構造化テキストで境界品質が上がる
try:
    from privacy_masker import MorphologicalRefiner
    REFINER = MorphologicalRefiner()
except ImportError:
    REFINER = None  # 既定のSpanRefinerにフォールバック

# データフォルダ（公知データ .pmk の置き場）。環境変数 PRIVACY_MASKER_DRIVE_DIR で差し替える
DATA_DIR = os.path.expanduser(os.environ.get("PRIVACY_MASKER_DRIVE_DIR", "~/privacy-masker-data"))

KEY = b"try-privacy-masker-demo-key"  # デモ用。実運用では環境変数などから渡すこと


def hr(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def show(label: str, before: str, after: str) -> None:
    print(f"--- {label} ---")
    print(f"  原文  : {before}")
    print(f"  変換後: {after}")


def read_inputs(args) -> list[str] | None:
    """マスク対象の文書リストを返す。入力指定がなければNone（デモモード）。"""
    docs: list[str] = []
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        docs.extend(line for line in text.splitlines() if line.strip())
    if args.text == ["-"] or (not args.text and not args.file and not sys.stdin.isatty()):
        docs.extend(line for line in sys.stdin.read().splitlines() if line.strip())
    elif args.text:
        docs.extend(args.text)  # 引数1つ=1文書
    return docs if docs else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*",
                    help="マスクするテキスト（複数可、1引数=1文書。'-'で標準入力）")
    ap.add_argument("--file", default=None, help="マスクするファイル（1行=1文書）")
    ap.add_argument("--out", default=None, help="マスク結果の保存先ファイル")
    ap.add_argument("--show-entities", action="store_true", help="検出内訳も表示")
    ap.add_argument("--key", default=None, help="hmac鍵（既定: デモ用固定鍵）")
    # 既定は Hub で配布している本番モデル（privacy_masker.hub.DEFAULT_MODEL_ID）。環境変数 PRIVACY_MASKER_MODEL で差し替え可
    ap.add_argument("--model", default=os.environ.get("PRIVACY_MASKER_MODEL", DEFAULT_MODEL_ID),
                    help="NER モデルのフォルダか Hugging Face Hub のリポジトリ ID（既定: 環境変数 PRIVACY_MASKER_MODEL、無ければ Hub の本番モデル）")
    ap.add_argument("--device", default=None,
                    help="cpu / mps / cuda（既定: 文字ヘッドモデルは cpu、token モデルは自動選択）")
    ap.add_argument("--level", default="strict", choices=["strict", "standard", "minimal"],
                    help="匿名化レベル（既定 strict=現行。standard は公人・公知を残す）")
    ap.add_argument("--knowledge", default=os.environ.get("PRIVACY_MASKER_KNOWLEDGE", f"{DATA_DIR}/jawiki_titles_ns0.pmk"),
                    help="公知データ .pmk（既定: PRIVACY_MASKER_KNOWLEDGE、無ければデータフォルダの jawiki_titles_ns0.pmk。tools/build_knowledge.py で生成）")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    global KEY
    if args.key:
        KEY = args.key.encode("utf-8")

    try:
        model = resolve_model_dir(args.model)   # Hub のリポジトリ ID ならここで取得（キャッシュ済みなら再取得しない）
    except ConfigurationError as e:
        sys.exit(f"{e}\n--model か環境変数 PRIVACY_MASKER_MODEL でフォルダか Hub のリポジトリ ID を指定してください")

    print(f"モデル読み込み中: {model}")
    # 文字ヘッドモデル（char_head.safetensors あり）と token モデルを自動で見分ける
    if os.path.exists(os.path.join(model, "char_head.safetensors")):
        device = args.device or "cpu"
        print(f"detector: CharHeadNerDetector  device: {device}")
        ner = CharHeadNerDetector(model=model, device=device)
    else:
        print(f"detector: TransformersNerDetector  device: {args.device or '自動選択'}")
        ner = TransformersNerDetector(model=model, device=args.device)
    # 分類ヘッド: 文字ヘッドモデルのフォルダに category_head.json があれば自動で読み込まれている
    # （CharHeadNerDetector(category_head="auto")、既定）。token モデルには属性が無いので None になる
    head = getattr(ner, "category_head", None)
    if head is not None:
        print(f"分類ヘッド: あり（min_margin={head.min_margin}）")
    else:
        print("分類ヘッド: なし")
    regex = RegexDetector.default_ja()
    handles = HandleDetector()   # チャット・ログのハンドル名（v0.8: USERNAME 型）

    # PERSON/ORG/ADDRESS/FACILITY/LOCATIONは実在風surrogate、それ以外（EMAIL等）はトークン化
    # ORG/FACILITY/LOCATIONは分類ヘッドがあれば規則で決まらないカテゴリの決定に使う（なければ規則のみ）
    rules = {
        "PERSON": Surrogate(kind="person", key=KEY),
        "ORG": Surrogate(kind="org", key=KEY, category_head=head),
        "ADDRESS": Surrogate(kind="address", key=KEY),
        "FACILITY": Surrogate(kind="facility", key=KEY, category_head=head),
        "LOCATION": Surrogate(kind="location", key=KEY, category_head=head),
    }
    engine_kw = {} if REFINER is None else {"refiner": REFINER}
    if args.level != "strict":
        # 識別性ベース匿名化（v0.8）: 公人・公知は *_PUBLIC サブタイプになり Keep で残る。
        # 私人・非公知・私人に帰属する準識別子はサブタイプが付かず、上の rules / default で匿名化
        if not os.path.exists(args.knowledge):
            sys.exit(f"公知データが見つかりません: {args.knowledge}\n"
                     "PYTHONPATH=. python tools/build_knowledge.py で生成するか --knowledge で指定してください")
        knowledge = PublicKnowledge.load(args.knowledge)
        print(f"level: {args.level}  公知データ: {knowledge}")
        rules.update(Policy.preset(args.level).rules)
        rules["ADDRESS"] = Surrogate(kind="address", key=KEY, keep="city")  # 市区町村まで残す
        engine_kw["annotators"] = [IdentifiabilityAnnotator(level=args.level, knowledge=knowledge)]
    else:
        print("level: strict（全エンティティを匿名化。--level standard で公人・公知を残す）")
    surrogate_policy = Policy(rules=rules, default=Tokenize(strategy="sequential"))
    engine = PrivacyEngine(detectors=[ner, regex, handles], policy=surrogate_policy, **engine_kw)

    # ---- マスクモード: コマンドラインから渡された入力を処理して終了 ----
    docs = read_inputs(args)
    if docs is not None:
        results = engine.transform_batch(docs)
        out_lines = [r.text for r in results]
        print()
        for src, r in zip(docs, results):
            print(r.text)
            if args.show_entities:
                for e in r.entities:
                    print(f"    {e.entity_type:<12} {src[e.original_start:e.original_end]!r}"
                          f" -> {e.replacement!r} (score={e.score:.2f}, op={e.operator})",
                          file=sys.stderr)
        if args.out:
            Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"\n保存しました: {args.out}", file=sys.stderr)
        return

    # ---------------------------------------------------------------
    hr("デモ1: Surrogate仮名化（姓リンク付き）")
    text = (
        "お世話になっております。大和技研の山田です。\n"
        "先日の件、弊社の山田太郎までご連絡ください。\n"
        "同席する山田花子も承知しております。\n"
        "住所は東京都新宿区西新宿1-2-3です。"
    )
    r = engine.transform(text)
    show("姓のみ言及もフルネームと同じ偽姓にリンクされる", text, r.text)
    print("  検出内訳:")
    for e in r.entities:
        print(f"    {e.entity_type:<10} [{e.original_start}:{e.original_end}]"
              f" -> {e.replacement} (score={e.score:.2f}, op={e.operator})")

    # ---------------------------------------------------------------
    hr("デモ2: hmac決定性（別々の呼び出しでも同一人物=同一偽名）")
    a = engine.transform("担当は山田太郎です。")
    b = engine.transform("本件の窓口は山田太郎になります。")
    show("呼び出し1", "担当は山田太郎です。", a.text)
    show("呼び出し2", "本件の窓口は山田太郎になります。", b.text)
    print("  → 同じ鍵の下で決定的（Store不要・プロセスを跨いでも同じ結果）")

    # ---------------------------------------------------------------
    hr("デモ3: Tokenize(vault) + restore（可逆トークン化）")
    try:
        from privacy_masker.crypto.fernet import FernetProvider
    except ImportError:
        print("  cryptography未導入のためスキップ（pip install 'privacy-masker[crypto]'）")
    else:
        vault_engine = PrivacyEngine(
            detectors=[ner, regex, handles],
            policy=Policy(default=Tokenize(strategy="vault")), **engine_kw,
            secret_key=KEY,
            crypto=FernetProvider(FernetProvider.generate_key()),
        )
        text3 = "田中一郎さん（メール: ichiro.tanaka@example.co.jp）から入電。"
        masked = vault_engine.transform(text3, scope_id="case_001")
        restored = vault_engine.restore(masked.text, scope_id="case_001")
        show("トークン化", text3, masked.text)
        print(f"  復元    : {restored.text}")
        print(f"  復元一致: {restored.text == text3}")

    # ---------------------------------------------------------------
    hr("デモ4: 正規表現検出の併用（メール・電話・郵便番号）")
    text4 = "連絡先: 03-1234-5678 / suzuki@example.com / 〒160-0023"
    r4 = engine.transform(text4)
    show("NERと正規表現の検出を統合", text4, r4.text)

    # ---------------------------------------------------------------
    if not args.no_bench:
        hr("デモ5: バッチ処理スループット（LLM学習データ想定）")
        names = ["佐藤健一", "鈴木美咲", "高橋大輔", "伊藤さくら", "渡辺翔太", "小林優子"]
        orgs = ["株式会社青空製作所", "みらい物流株式会社", "ひかり銀行"]
        docs = [
            f"{orgs[i % 3]}の{names[i % 6]}です。会議は東京都千代田区丸の内1-1-{i + 1}の"
            f"本社で行います。詳細は {'taro'}{i}@example.com までご連絡ください。"
            for i in range(48)
        ]
        engine.transform_batch(docs[:4])  # ウォームアップ
        t0 = time.perf_counter()
        for d in docs:
            engine.transform(d)
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        results = engine.transform_batch(docs)
        t3 = time.perf_counter()
        n = len(docs)
        print(f"  transform 逐次    : {(t1 - t0) / n * 1000:6.1f} ms/doc ({n / (t1 - t0):6.1f} docs/s)")
        print(f"  transform_batch   : {(t3 - t2) / n * 1000:6.1f} ms/doc ({n / (t3 - t2):6.1f} docs/s)")
        print(f"  高速化率          : {(t1 - t0) / (t3 - t2):.1f}x")
        print(f"  出力例: {results[0].text}")

    print("\n完了。オプションは --help を参照してください。")


if __name__ == "__main__":
    main()
