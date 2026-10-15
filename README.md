# privacy-masker

日本語テキストの個人情報（PII）を検出し、トークン化・仮名化・匿名化する Python ライブラリです。
Detector（検出）/ Policy（方針）/ Operator（変換）を分離した構成で、LLM への入力の保護、学習データの匿名化、
ログや帳票の仮名化に使えます。

*privacy-masker is a Japanese PII detection and de-identification / pseudonymization library with a
Detector / Policy / Operator architecture: deterministic tokens, reversible vault tokens, and
realistic same-category surrogate names (people, organizations, facilities, places, addresses) with
exact round-trip restore. MIT licensed. Documentation is in Japanese.*

## 特徴

- **検出器を選べる**: Hugging Face の token-classification モデル、自前学習の文字単位 NER、正規表現（電話番号・マイナンバー・法人番号・在留カード番号など日本固有の識別子）、ハンドル名検出器を組み合わせられます。
- **3 種類のトークン化**: 文書内連番 `<PERSON_001>`、鍵付き HMAC で文書をまたいで同じ値になる `<PERSON_MFZWK4TB>`、`restore()` で元に戻せる vault。
- **意味を保つ偽名への置換（Surrogate）**: 人名は性別傾向と姓リンク（同じ姓は同じ偽姓）を保ち、組織・施設・地名・住所は同じ粗いカテゴリの実在風の別名に置き換えます（病院 → 別の病院名、製造業 → 別の製造業の社名、東北の市 → 東北の別の市）。偽名は元の値の一部を含まず、文書本文にある語や公知の実体名とも衝突しません。vault 戦略なら偽名から元の値へ完全一致で復元できます。
- **識別性ベースの匿名化**: 公人名や公知の地名・施設・組織を残し、私人を識別する情報だけを匿名化する `standard` レベル（既定は全部マスクする `strict`）。
- **日本語向けの境界補正**: NER が巻き込んだ敬称・助詞を削り、落とした種別語尾・法人格（〜病院・株式会社〜）を取り込みます。形態素解析（fugashi / Janome）による補正も選べます。
- **運用向けの機能**: 型別の信頼度閾値、allow / deny リスト、判定経緯の説明（原文を含まない）、dict・DataFrame の列単位変換、バッチ推論、YAML / JSON 設定、gold 付き文書での評価。
- **依存ゼロの core**: 正規表現・トークン化・Surrogate は追加依存なしで動きます。NER は `[transformers]`、復元可能な暗号化は `[crypto]`、形態素解析は `[mecab]` または `[janome]` を追加します。

## インストール

```bash
pip install privacy-masker                   # core（依存ゼロ）
pip install 'privacy-masker[transformers]'   # NER（transformers + torch）
pip install 'privacy-masker[crypto]'         # vault 戦略・Encrypt（cryptography）
pip install 'privacy-masker[mecab]'          # MorphologicalRefiner の高速バックエンド（fugashi + IPADIC）
pip install 'privacy-masker[janome]'         # 同・純 Python バックエンド
```

Python 3.10 以上。

## クイックスタート

```python
from privacy_masker import PrivacyEngine, Policy
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector

engine = PrivacyEngine(
    detectors=[
        CharHeadNerDetector(model="az225/privacy-masker-ja-pii-ner-30m"),   # 当方の文字単位 NER（Hub から取得。後述）
        RegexDetector.default_ja(),   # 電話番号・メール・マイナンバー・法人番号など
        HandleDetector(),             # チャット・ログのハンドル名 → USERNAME
    ],
    policy=Policy.preset("llm_input"),   # すべて <TYPE_001> 形式のトークンに
)

result = engine.transform("山田太郎さんは株式会社ABCに勤務しており、東京都新宿区に住んでいる。")
print(result.text)
# <PERSON_001>さんは<ORG_001>に勤務しており、<LOCATION_001>に住んでいる。
```

手元で一通り試すには `examples/quickstart.py` を実行してください（連番トークン → 意味を保つ偽名 → vault の往復 → standard レベル → `detect()` の順に出力します。NER モデルは既定で上記の Hub のモデル（`az225/privacy-masker-ja-pii-ner-30m`）を取得します。`--model` でフォルダか Hub のリポジトリ ID を指定できます）。

`engine.detect(text)` で検出結果だけを取り出せます。大量の文書は `engine.transform_batch(texts)` でまとめて処理してください（推論をバッチ化し、結果は逐次処理と同一です）。

### NER モデルについて

ライブラリ本体はモデルを同梱しません。次のどちらかを使います。

- **Hugging Face の token-classification モデル**: `TransformersNerDetector(model=..., label_map=...)`。ラベル名を `PERSON` / `ORG` / `ADDRESS` / `LOCATION` / `FACILITY` に対応付けます。
- **文字単位ヘッド付きの自前モデル**: `CharHeadNerDetector(model=<フォルダ または Hugging Face Hub のリポジトリ ID>)`。日本語のサブワード境界に左右されない文字単位のスパンを出します。Hub のリポジトリ ID（`owner/name`）を渡すと推論に必要なファイルだけを取得してキャッシュします（`revision=` で固定、`local_files_only=True` でオフライン）。当方が `cl-nagoya/ruri-v3-30m` を基に学習したモデルは Hugging Face Hub の [`az225/privacy-masker-ja-pii-ner-30m`](https://huggingface.co/az225/privacy-masker-ja-pii-ner-30m)（Apache-2.0）で配布しています: `CharHeadNerDetector(model="az225/privacy-masker-ja-pii-ner-30m")`。構成・学習データ・評価はモデルカードに、学習・評価のスクリプトは `tools/` にあります。

どのモデルでも、導入前に自分の文書で評価することを勧めます（後述「評価」）。

## 仕組み

```
テキスト → Detector（NER・正規表現・ハンドル名）→ Refiner（境界補正）→ Resolver（重複解決）
        → Annotator（識別性の判定、任意）→ Policy（型ごとの Operator）→ 変換後テキスト
```

- **Refiner**: 既定の `SpanRefiner` は文字種の規則で端の空白・敬称・助詞を削り、隣接する種別語尾・法人格・住所末尾を取り込みます（PII を落とす方向の変更はしません）。`MorphologicalRefiner` は品詞で判定し、スパンを形態素境界に揃えます。
- **Resolver**: 重なった検出結果は型の優先順位（MY_NUMBER > CREDIT_CARD > EMAIL > PHONE_NUMBER > PERSON > ADDRESS > …）で決め、同じ型ならスコア、同点なら長い方を採ります。
- **Policy**: 型ごとに Operator を割り当てます。プリセットは `llm_input` / `strict`（連番トークン）、`dataset`（HMAC トークン）、`standard` / `minimal`（公知の実体は残す）。

## トークン化の 3 戦略

| strategy | 出力例 | 決定性 | 復元 |
|---|---|---|---|
| `sequential` | `<PERSON_001>` | 文書（scope）内のみ | 不可 |
| `hmac` | `<PERSON_MFZWK4TB>` | 同じ鍵（+ scope）で常に同じ値 | 鍵の秘匿が前提。鍵保持者は全数照合できる |
| `vault` | `<PERSON_…>`（乱数） | scope 内のみ | `engine.restore(text, scope_id=...)` |

## 意味を保つ偽名への置換（Surrogate）

```python
from privacy_masker.operators import Surrogate

policy = Policy(rules={
    "PERSON":   Surrogate(kind="person"),    # 山田太郎 → 佐藤健一（性別傾向を保存。署名の「山田」も「佐藤」に）
    "ORG":      Surrogate(kind="org"),       # 株式会社大和電機 → 株式会社◯◯精機（同じ製造業）
    "FACILITY": Surrogate(kind="facility"),  # 大江戸記念病院 → ◯◯中央クリニック
    "LOCATION": Surrogate(kind="location"),  # 奈良市 → 近畿の別の市
    "ADDRESS":  Surrogate(kind="address"),   # 実在風の別住所（丁目表記を保存）
})
# Surrogate(kind="auto") は entity_type から人名 / 組織 / 施設 / 地名 / 住所を自動で選びます
```

**カテゴリの決め方**: 種別語の規則で決めます。組織は 医療法人・学校法人・官公庁（省庁・局・議会・公的な委員会など）・メディア（新聞社・放送・球団名）・政党・団体（協会・組合・〜の会）・部署（〜部・〜課）・運輸・製造・IT・金融・小売・建設、それ以外は一般の会社。施設は 医療・教育・交通・商業・公共・建物・文化。地名は 国（国名一覧と略称）・都道府県・地区（〜地区・選挙区）・外国（カタカナの地名、漢字の外国都市）・市区町村・町名・自然地名。規則で決まらないものは型ごとの既定の生成器に落ちます。

**安全性**: 偽名が元の値に依存するのは、カテゴリと表層特徴（文字種・法人格・住所の粒度・地方ブロックなど、どれも母集団が数千件以上の粗い属性）と鍵付き HMAC のシードを通してだけです。候補は次の条件をすべて満たすものだけ採用します。

- 元の値と同じでない。元の姓・名、組織・施設のコア語、住所の市区町村・町名、地名の語幹を含まない。元の値に含まれる国名や、略称に対する同じ国の標準名（米国 → アメリカ）も使わない。
- 変換中の文書本文に既に出ている語でない（検出漏れの実名との同一化を防ぐ）。
- 同梱の禁止リスト（生成器が出しうる候補と Wikipedia 見出し語の衝突分）、実行時に渡した公知データ `knowledge=`、利用者の `denylist=` に無い。

候補が尽きたときは `<TYPE_xxxxxxxx>` トークンに退避し、**元の値をそのまま返す経路はありません**。
「元の値の埋め込みに近い候補を選ぶ」方式は、同じモデルと候補集合を持つ攻撃者が写像を再現できるため採用していません。

**往復**: `Surrogate(kind=..., strategy="vault")` と `engine.restore()` で偽名から元の値へ戻せます。LLM に偽名入りの文書を渡し、応答の偽名を元に戻す用途向けです。

```python
policy = Policy(rules={"PERSON": Surrogate(kind="person", strategy="vault")}, ...)
result = engine.transform(text, scope_id="case_1")          # 偽名に変換
back = engine.restore(llm_response, scope_id="case_1")       # 偽名 → 元の値（完全一致のみ）
```

復元は文字列の完全一致で行い、文脈や共参照は見ません。LLM の応答が偶然同じ表記の名前を書いた場合、そこも元の値に置き換わります。
`semantic=False` を指定すると、カテゴリを見ない旧来の偽名生成（型に合った偽名・表層特徴の保存・姓リンク）になります。

## 識別性ベースの匿名化（strict / standard / minimal）

既定の `strict` は検出した全エンティティを匿名化するため、「安倍元総理」が別の人名に、「東京都庁」が架空の企業名に変わります。
`standard` は **私人を識別する情報だけを匿名化し、公人名や公知の地名・施設・組織はそのまま残します**。

```python
from privacy_masker import IdentifiabilityAnnotator, MorphologicalRefiner, PrivacyEngine, Policy, PublicKnowledge
from privacy_masker.detectors import CharHeadNerDetector, RegexDetector
from privacy_masker.operators import Surrogate

ner = CharHeadNerDetector(model="az225/privacy-masker-ja-pii-ner-30m")
KEY = b"replace-with-your-own-secret-key"      # Surrogate の決定性に使う鍵（運用では環境変数などから）

knowledge = PublicKnowledge.load("jawiki_titles_ns0.pmk")   # tools/build_knowledge.py で Wikipedia 見出し語から生成
policy = Policy.preset("standard")                           # *_PUBLIC → Keep、それ以外はトークン化
policy.rules.update({
    "PERSON":  Surrogate(kind="person", key=KEY),
    "ADDRESS": Surrogate(kind="address", key=KEY, keep="city"),   # 市区町村は残し、番地以下を置換
})
engine = PrivacyEngine(
    detectors=[ner, RegexDetector.default_ja()], refiner=MorphologicalRefiner(), policy=policy,
    annotators=[IdentifiabilityAnnotator(level="standard", knowledge=knowledge)],
)
engine.transform("安倍元総理は奈良市で演説し、田中さんは新宿駅で待った。").text
# → 安倍元総理・奈良市・新宿駅はそのまま、田中だけが偽姓に
```

- 公人と判定するのは、公職の肩書き（総理・大臣・知事・議員など）が隣接するか、フルネームが公知データに一致するときだけです。企業の肩書きや姓だけでは公人にしません。
- 公知の地名・施設・組織は、Wikipedia 見出し語に一致し、かつ私人に帰属していないときだけ残します（「田中さんの最寄りは◯◯駅」の駅は匿名化）。
- **保持は正の証拠があるときだけ**です。Policy に `*_PUBLIC` の規則が無ければ従来どおりマスクされます。
- 既知の限界: 著名人と同姓同名の私人は `standard` で保持され得ます。公人との共起自体が手掛かりになる文は、私人名は匿名化されても文脈からの絞り込みは残ります。許容できない用途では `strict` を使ってください。

## 検出できる型

- **正規表現** `RegexDetector.default_ja()`: EMAIL、PHONE_NUMBER（全角・国際番号・空白区切り）、POSTAL_CODE、CREDIT_CARD（Luhn）、IP_ADDRESS（IPv4 / IPv6）、MAC_ADDRESS、URL、MY_NUMBER（検査用数字）、CORPORATE_NUMBER（法人番号）、PASSPORT、RESIDENCE_CARD、文脈語が必須の DRIVERS_LICENSE・PENSION_NUMBER・BANK_ACCOUNT。`default_ja(dates=True)` で DATE と AGE も検出します。全角の入力は長さを変えない写像で半角化して照合するので、`０４８－２２２－７８１５` のような表記もそのまま拾えます。
- **NER**: PERSON / ADDRESS / ORG / FACILITY / LOCATION。
- **ハンドル名** `HandleDetector`: USERNAME（`yuki_t`、`@handle`、`user=…`）。
- 自前の型は `RegexPattern("PROJECT", re.compile(r"PJ-\d{4}"), score=0.6, context=("案件",))` のように追加できます。

## 呼び出し時オプション

```python
engine.transform(
    text,
    entities=["PERSON", "ADDRESS"],                   # この型だけ対象にする
    score_threshold={"DEFAULT": 0.7, "PERSON": 0.5},  # 型別の信頼度閾値
    allow_list=["株式会社アオバ"],                     # 検出されても必ず残す語（allow_list_match="regex" も可）
    deny_list={"ハヤブサ計画": "PROJECT"},             # 必ずマスクする語 → 指定した型で Policy に渡る
    explain=True,                                     # 各エンティティに判定の経緯を付ける
)
result.entities[0].explanation
# 'detector=transformers_ner(0.98); refined 18-22→18-20; annotator PERSON→PERSON_PUBLIC; operator=keep'
```

`explain` の文字列は検出器名・スコア・位置・型だけで、原文の文字を含みません。`PrivacyEngine(...)` に同名の引数を渡すと既定値になります。
`ContextEnhancer()` を加えると、スパンの前後に型ごとの手掛かり語（氏名・住所・電話 …）があるとき score を上げられ、`score_threshold` と組み合わせて「文脈のない低スコア検出だけを落とす」運用ができます。

## オペレータ一覧

| Operator | 出力 | 可逆 | 決定性 |
|---|---|---|---|
| `Tokenize(strategy="sequential" / "hmac" / "vault")` | `<PERSON_001>` / `<PERSON_MFZWK4TB>` / 乱数トークン | vault のみ `restore()` | 戦略による |
| `Surrogate(kind=...)` | 実在風の別名（意味を保つ置換・姓リンク） | vault なら `restore()` | 同じ鍵（+ scope）で常に同値 |
| `Hash(algorithm, salt, length)` | 16 進ダイジェスト | 不可 | 常に同値 |
| `Encrypt(key)` | Fernet 暗号文 | `decrypt_text(text, key)` | 毎回異なる |
| `Mask(char, keep_prefix, keep_suffix)` | `03-****-5678` | 不可 | — |
| `Replace(text)` / `Redact()` / `Keep()` | 固定文字列 / 削除 / そのまま | — | — |
| `Custom(fn)` | `fn(value, entity)` の戻り値 | fn 次第 | fn 次第 |
| `DateShift(days= / max_days=)` | 日付を一定日数ずらす（順序と間隔を保存） | 不可 | 同じ鍵で常に同値 |
| `Bucketize(size)` | 数値を区間に一般化（34歳 → 30-39歳） | 不可 | — |
| `FormatPreserving(key)` | 桁数と文字種を保った仮名（03-1234-5678 → 07-9302-4416） | 不可 | 同じ鍵で常に同値 |

## 構造化データ

```python
from privacy_masker.structured import transform_records, transform_dataframe
rows = transform_records(engine, records)   # list[dict] → list[dict]（文字列フィールドだけ変換）
df2  = transform_dataframe(engine, df)      # pandas DataFrame → 新しい DataFrame
```

`column_types="auto"`（既定）は列ごとに標本を検出し、値のほぼ全体が単一の型に一致する列（氏名列・住所列）は列全体をその型として扱います。NER が短い値を落としても列単位で守られます。`column_types={"name": "PERSON"}` で明示指定もできます。

## 設定ファイルから組み立てる

```python
from privacy_masker.config import load_engine
engine = load_engine("privacy_masker.yaml")   # .yaml は PyYAML が必要。.json も可。書式は privacy_masker/config.py の docstring
```

検出器（`char_head` で Hub の文字ヘッドモデルも指定可）・Refiner・Enhancer・Annotator・Policy（preset + rules）・呼び出しオプション・鍵の環境変数名を宣言的に書けます。

## 評価 — 自分のデータで測る

精度は文書種・文体で大きく変わります。導入前に自分の文書で評価してください。gold 付き文書は 1 行 1 文書の JSONL で、`text`・`entities`（`[start, end, label]`、文字オフセット）、任意の `neutral`（採点しない範囲）を持ちます。

```python
from privacy_masker.evaluation import evaluate, load_jsonl
report = evaluate(load_jsonl("gold.jsonl"), engine)
print(report.to_markdown())        # 文字 coverage / 危険(被覆 50% 未満) / 完全見逃し / 1 字以上漏れ / 過剰マスク字
report.by_label["PERSON"].danger   # ラベル別。漏れの一覧は位置のみで原文を含みません
```

見る順番は、PERSON / ADDRESS の **完全見逃し** と **危険**（漏れる方向）、次に **過剰マスク字**（消しすぎる方向）です。span F1 は境界のずれと漏れを区別できないので補助に使います。

参考として、当方の文字単位モデル（`ruri-v3-30m` 系、正規表現・ハンドル名・Refiner 込みの構成）の結果を示します。いずれも学習に使っていない文書で、「危険」は被覆 50% 未満、「完全見逃し」は被覆ゼロの gold エンティティ数（型不問）です。

| 評価セット | 危険 | 完全見逃し | 過剰マスク字 |
|---|---|---|---|
| 固定 OOD（327 文書・11 ジャンル。帳票・チャット・メールを含む） | 11 | 11 | 285 |
| UD Japanese GSD テスト（543 文書） | 12 | 11 | 127 |
| OpenPII 日本語（1,000 文書） | 9 | 4 | — |

このほか、開発中に一切参照しない盲検セットで最終判定を行っています。見逃しはゼロではないので、人手レビューの併用を前提にしてください。

## 性能の目安

NER が所要時間のほぼ全てを占めます（Apple Silicon、平均 337 字の文書、`transform_batch`）。

| 構成 | CPU | MPS |
|---|---|---|
| 30m モデル + 正規表現 + SpanRefiner | 13.8 ms/doc | 7.5 ms/doc |
| 同 + fugashi（`[mecab]`） | 14.2 ms/doc | 7.7 ms/doc |
| 同 + Janome（`[janome]`） | 19.1 ms/doc | 12.4 ms/doc |
| 正規表現 + ハンドル名のみ | 0.5 ms/doc | — |

- `CharHeadNerDetector` の `device` の既定は `cpu` です。Apple Silicon なら `device="mps"`、NVIDIA なら `"cuda"` を明示してください。`TransformersNerDetector` は `device` を省くと transformers が CUDA → MPS → CPU の順で自動選択します。精度型の既定は float32 で、MPS と CPU では bf16 / fp16 は速くなりません。
- 形態素解析が必要なら `[mecab]`（fugashi）を推奨します。Janome は純 Python で約 5 ms/doc 余計にかかります。
- detector 側の `batch_size` は CPU で 8〜16、GPU で 32 以上が目安です。

## Presidio との対応

| Presidio | privacy-masker |
|---|---|
| `AnalyzerEngine.analyze()` / `AnonymizerEngine.anonymize()` | `engine.detect()` / `engine.transform()` |
| `DeanonymizeEngine.deanonymize()` | `engine.restore()`（vault）/ `decrypt_text()` |
| `OperatorConfig("replace" / "mask" / "hash" / "encrypt" / "custom" / "keep")` | `Replace` / `Mask` / `Hash` / `Encrypt` / `Custom` / `Keep` |
| `PatternRecognizer(patterns=, deny_list=)` | `RegexDetector([RegexPattern(...)])`、`deny_list=` |
| `analyze(entities=, score_threshold=, allow_list=, return_decision_process=)` | `entities=`, `score_threshold=`, `allow_list=`, `explain=True` |
| `BatchAnalyzerEngine` / presidio-structured | `transform_batch()` / `structured.transform_records()` |
| `LemmaContextAwareEnhancer` | `ContextEnhancer` |
| YAML 設定 | `config.load_engine()` |

Presidio に無いもの: 形態素境界の補正、種別語尾の拡張、識別性ベースの保持、日本語の意味を保つ Surrogate と姓リンク、日本固有の識別子。privacy-masker に無いもの: 画像の墨消し、REST API、多言語 NLP エンジンの差し替え。

## 注意（脅威モデル）

- `hmac` 戦略は **鍵の秘匿を前提とする仮名化** です。鍵保持者は低エントロピーの値（電話番号など）を全数照合で復元できます。個人情報保護法の匿名加工情報の基準への適合を保証するものではありません。
- 対応表（Store）・secret_key・HMAC 鍵・Fernet 鍵は「復元につながる情報」として安全管理措置の対象です。
- NER・正規表現の検出漏れは起こり得ます。`detect()` による事前確認と人手レビューを併用してください。
- Surrogate の偽名は実在の個人・組織と偶然一致しないよう禁止リストで衝突を避けていますが、網羅を保証するものではありません。

## 開発

```bash
pip install -e '.[transformers,crypto,janome,dev]'
PYTHONPATH=. python -m pytest -q
```

設計文書は `docs/` にあります（アーキテクチャ `docs/superpowers/specs/2026-08-08-privacy-masker-design.md`、識別性ベースの匿名化 `docs/design_identifiability.md`、意味を保つ置換 `docs/superpowers/specs/2026-09-24-semantic-surrogate-design.md`）。変更履歴は `CHANGELOG.md`。

## ライセンス

MIT License

同梱の `privacy_masker/data/surrogate_denylist.pmk`（偽名が公知の実体名と衝突しないための禁止リスト）は、Wikipedia 日本語版の見出し語（ns0、2026-09-02 取得のダンプ。[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）と生成候補の衝突分を 64 bit ハッシュにしたもので、見出し語の本文は含みません。
