# privacy-masker 設計仕様書

- 日付: 2026-08-08
- ステータス: 承認済み
- パッケージ名: `privacy-masker`（import名: `privacy_masker`、PyPI空き確認済み 2026-08-08）

## 1. 概要と位置付け

任意のDetectorが検出したSensitive Entityに対して、Policyに基づいて
De-identification / Pseudonymization Operatorを適用する汎用Pythonライブラリ。

「個人情報NERライブラリ」ではない。NER・正規表現・その他任意の検出器を
差し替え可能なコンポーネントとして扱い、検出（Detector）と変換（Operator）を
分離する。検出と変換を分離する一般的な設計思想を参考にしつつ、
日本語NER（HuggingFace Transformers token classification）をfirst-classに扱う
軽量ライブラリとして設計する。

### 決定事項（要件ヒアリング結果）

| 項目 | 決定 |
|---|---|
| 位置付け | 汎用PII匿名化SDKとしてPyPI公開を目指す |
| 保有NER | HuggingFace Transformers token classificationモデル（日本語: 人名・住所・会社名） |
| v0.1範囲 | sequential / hmac / vault の3トークン戦略すべて。Store（Memory/SQLite）とCrypto（Fernet）を含む |
| 構成 | 単一パッケージ + optional extras（案A）。coreはstdlibのみに依存 |
| Python | 3.10以上 |

## 2. アーキテクチャ

```
text ─→ Detector群 ─→ EntityResolver ─→ Policy ─→ Operator ─→ TransformResult
        (検出)        (重複解決)      (方針決定)  (変換)
```

- 各段はProtocolで疎結合。NERDetectorとOperatorは直接結合しない。
- 置換はentityの `start` 降順で適用し、offsetずれを構造的に排除する。

### パッケージ構成

```
privacy_masker/
├── __init__.py          # PrivacyEngine, Policy, Entity など公開APIをre-export
├── engine.py            # PrivacyEngine
├── types.py             # Entity, TransformResult, TransformedEntity, RestoreResult
├── entity_types.py      # PERSON, ORG などの定数
├── policy.py            # Policy, presets
├── resolver.py          # EntityResolver
├── detectors/
│   ├── base.py          # Detector / BatchDetector Protocol
│   ├── regex.py         # RegexDetector + default_ja() プリセット
│   └── transformers.py  # TransformersNerDetector（[transformers] extra）
├── operators/
│   ├── base.py          # Operator Protocol, TransformContext
│   ├── replace.py       # Replace
│   ├── redact.py        # Redact
│   ├── mask.py          # Mask
│   ├── keep.py          # Keep（明示的な素通し）
│   └── tokenize.py      # Tokenize（sequential / hmac / vault）
├── scopes.py            # scope管理（document / scope_id）
├── stores/
│   ├── base.py          # PseudonymStore Protocol
│   ├── memory.py        # MemoryStore
│   └── sqlite.py        # SQLiteStore
├── crypto/
│   ├── base.py          # CryptoProvider Protocol
│   └── fernet.py        # FernetProvider（[crypto] extra）
├── secrets_types.py     # SecretBytes（reprを伏せる鍵ラッパー）
├── normalize.py         # NFKC等の正規化
├── encoding.py          # フィールドの単射エンコード（HMAC入力・vault平文の共通処理）
└── exceptions.py
```

### 依存関係

- core: 標準ライブラリのみ（`re`, `hmac`/`hashlib`, `secrets`, `sqlite3`, `base64`, `dataclasses`, `typing`）
- extras:
  - `privacy-masker[transformers]` → `transformers`（TransformersNerDetector用）
  - `privacy-masker[crypto]` → `cryptography`（FernetProvider用）
  - `privacy-masker[pandas]` → v0.2で追加予定
- 「extras無しでcoreがimportできる」ことをCIの独立ジョブで検証する。

## 3. 公開API

公開APIは `PrivacyEngine` の4メソッドに絞る。

```python
from privacy_masker import PrivacyEngine, Policy
from privacy_masker.detectors import RegexDetector, TransformersNerDetector

engine = PrivacyEngine(
    detectors=[TransformersNerDetector(model=...), RegexDetector.default_ja()],
    policy=Policy.preset("llm_input"),
    secret_key_env="PM_SECRET_KEY",   # Store永続化・hmac戦略を使う場合に必須（§7・§9）
    # store=..., crypto=..., resolver=..., on_detector_error="raise" は省略可（既定あり）
)

result   = engine.transform(text)                          # 変換（document scope）
result   = engine.transform(text, scope_id="patient_123")  # 変換（永続scope）
dets     = engine.detect(text)                             # 検出のみ
results  = engine.transform_batch(texts, batch_size=32)    # バッチ変換
restored = engine.restore(masked_text, scope_id="patient_123")  # 復元（vault戦略のみ）
```

シグネチャ:

```python
def transform(self, text: str, *, scope_id: str | None = None) -> TransformResult: ...
def detect(self, text: str) -> list[Entity]: ...
def transform_batch(self, texts: Sequence[str], *,
                    scope_id: str | None = None,
                    batch_size: int = 32) -> list[TransformResult]: ...
def restore(self, text: str, *, scope_id: str) -> RestoreResult: ...
```

- `transform_batch` は既定（`scope_id=None`）で**各文書を独立したdocument scope**として
  処理する。バッチ全体（および呼び出しを跨いで）トークンを一貫させたい場合は
  `scope_id` を渡す。`per_document` のような追加パラメータは設けない
  （scope_idの有無ですべて表現できる）。
- `restore` の `scope_id` は必須。

設計判断:

- `dry_run=True` フラグは設けない。「置換しないtransform」=「検出」なので
  `detect()` に一本化する（NER評価・UIハイライト・投入前チェック用途）。
- エンジン名は `Pseudonymizer` ではなく `PrivacyEngine`。redact/maskなど
  仮名化以外のOperatorも担うため中立な名前にする。
- 構成の妥当性検証は可能な限り `PrivacyEngine.__init__` で行う（構築時に失敗させる）。
  呼び出し引数と構成の組み合わせで初めて判明する不整合は呼び出し時に
  `ConfigurationError` を送出する（§12）。

構築時検証の規則:

| 構成 | 検証 |
|---|---|
| Policyにvault戦略を含む | CryptoProvider必須・secret_key必須。欠けたら `ConfigurationError` |
| Policyにhmac戦略を含む | operator個別のkey指定 or engineのsecret_keyのどちらかが必須 |
| `secret_key` と `secret_key_env` の同時指定 | `ConfigurationError` |

呼び出し時検証の規則:

| 呼び出し | 検証 |
|---|---|
| vault戦略を含むPolicyで `scope_id` 未指定のtransform | `ConfigurationError`（vaultは永続scopeでのみ意味を持つ） |
| `scope_id` 指定のtransformでsecret_key未設定 | `ConfigurationError`（value_hash計算に必要。§9） |

## 4. コア型

### Entity（検出結果の共通フォーマット）

```python
@dataclass(frozen=True, slots=True)
class Entity:
    entity_type: str      # "PERSON" など。openな文字列（enumにしない）
    start: int            # 元テキストのcodepoint offset（Python文字列のインデックス）
    end: int
    score: float          # 0.0〜1.0
    source: str = ""      # "transformers_ner" / "regex" など検出器名
```

- **原文テキストをフィールドに持たない**。必要なら `text[e.start:e.end]` で取り出す。
  結果オブジェクト・ログ・例外からのPII漏洩経路を型設計で塞ぐ。
- engineはDetectorが返した各Entityを検証する:
  `0 <= start < end <= len(text)` かつ `0.0 <= score <= 1.0` を満たさないEntityが
  あれば `DetectorError`（違反したdetector名のみをメッセージに含める）。
- `entity_types.py` に定数を提供:
  `PERSON, ORG, LOCATION, ADDRESS, EMAIL, PHONE_NUMBER, POSTAL_CODE,
  CREDIT_CARD, IP_ADDRESS, MY_NUMBER, DATE`
  （entity_typeは自由文字列なので利用者が独自typeを追加できる）

### TransformResult / RestoreResult

```python
@dataclass(frozen=True)
class TransformedEntity:
    entity_type: str
    original_start: int
    original_end: int
    replacement: str
    score: float
    operator: str          # 適用されたoperator名

@dataclass(frozen=True)
class TransformResult:
    text: str
    entities: tuple[TransformedEntity, ...]
    scope_id: str | None
    failed_detectors: tuple[str, ...] = ()   # on_detector_error="skip" 時のみ非空

@dataclass(frozen=True)
class RestoreResult:
    text: str
    unresolved: tuple[str, ...]   # vaultに見つからなかったトークン
```

- 原文値はどこにも含めない（`include_original` のようなオプトインも設けない）。
- `TransformedEntity.replacement` はMask等で原文由来の文字を部分的に含み得るため、
  **`__repr__` / `__str__` にはreplacementを含めない**（フィールドアクセスは可能）。
- `failed_detectors`: 部分成功モード（§12）で失敗したdetector名を機械可読に返す。
  「NERが全滅して素通しした結果」を正常結果と区別できるようにする。
- 検出精度のデバッグは `engine.detect(text)` と `text[e.start:e.end]` の
  突き合わせを利用者側で行う（ドキュメントで案内）。

## 5. Detector

### Protocol

```python
class Detector(Protocol):
    name: str
    def detect(self, text: str) -> list[Entity]: ...

class BatchDetector(Detector, Protocol):   # 実装は任意
    def detect_batch(self, texts: Sequence[str]) -> list[list[Entity]]: ...
```

### RegexDetector（core同梱）

- パターン集合（entity_type → compiled pattern + 検証関数）を渡す汎用regex検出器。
- `RegexDetector.default_ja()` で日本語向けプリセットを提供:

| entity_type | 内容 |
|---|---|
| EMAIL | RFC準拠は狙わず実用的なパターン |
| PHONE_NUMBER | 固定・携帯・ハイフン有無・市外局番括弧（例: 03-1234-5678, 090-1234-5678, (03)1234-5678） |
| POSTAL_CODE | 〒マーク有無両対応（123-4567, 〒123-4567） |
| CREDIT_CARD | 13〜19桁 + Luhnチェック検証（誤検出削減） |
| IP_ADDRESS | IPv4 |
| MY_NUMBER | 12桁 + 個人番号チェックデジット検証 |

- regexマッチのscoreは1.0固定。ただし検証関数（Luhn等）を通らないものは検出しない。

### TransformersNerDetector（[transformers] extra）

- HuggingFace token classificationモデルのAdapter。
- コンストラクタでモデル（pipelineまたはmodel+tokenizer）とラベルマッピングを受け取る:
  `label_map={"PER": "PERSON", "ORG": "ORG", "LOC": "ADDRESS"}`
- **label_mapに無いラベルは、ラベル名をそのままentity_typeとして通す**（既定）。
  破棄するとPolicyのdefault operatorすら適用されず素通しになるため、
  フェイルセーフ原則（§13）に従いpass-throughを既定とする。
  明示的に捨てたい場合は `drop_unmapped=True`。
- pipelineのaggregation結果（char offset付き）を `Entity` に変換する。
  offsetがcodepoint基準であることをテストで保証（tokenizer由来のbyte offset混入を防ぐ）。
- **長文はオーバーラップ付きスライディングウィンドウで分割して推論し、
  offsetを再結合する**（外側は`chunk_chars` / `chunk_overlap`、HF token上限超過時の
  内側strideは安全条件を確認して自動設定。§26参照）。
  オーバーラップ領域の重複検出はスパン一致でdedupする。
  モデルの最大系列長によるサイレント切り捨ては構造的に禁止する
  （切り捨てが起きる実装は「後半のPIIが素通しする」ため§13の最悪の故障モードに当たる）。
- `detect_batch` を実装し、GPUバッチ推論に対応（`batch_size` 指定可）。

## 6. EntityResolver（重複解決）

```python
resolver = EntityResolver(
    priority=["MY_NUMBER", "CREDIT_CARD", "EMAIL", "PHONE_NUMBER",
              "PERSON", "ADDRESS", "POSTAL_CODE", "LOCATION", "ORG"],
    min_score=0.0,
)
```

アルゴリズム（決定的・貪欲）:

1. `min_score`（**既定0.0** = 何も捨てない）未満のentityを除外する。
   閾値を上げると「低信頼度の検出が素通しする」リスクを利用者が引き受けることになる。
   この注意はAPIドキュメントに明記する。
2. 全entityを次のキーでソートする:
   `(priorityの順位, -score, -(end - start), start, 登録順)`
   - priorityの順位: priorityリストのindex。リストに無いtypeはリスト末尾扱いとし、
     検出結果に最初に現れた順で相対順位を付ける
   - 登録順: `detectors` リストの順序、同一detector内は `detect()` の返却順。
     これが最終タイブレークとなり、結果は常に決定的
3. 先頭から順に採用する。採用済みentityとスパンが重なる候補は、
   **重なる部分を切り落とし、非空の残余スパンを同じentity（同type・同score）として
   候補列に再挿入する**。残余が空（完全包含）なら破棄する。

- 残余の再挿入により「部分重複で負けたentityの非重複部分が平文で素通しする」
  漏洩経路を塞ぐ（例: PERSON(0-5)がADDRESS(3-20)に勝った場合、ADDRESS(5-20)が
  残余として再評価され、マスク対象になる）。切り詰められた残余は元の文字列として
  意味をなさないことがあるが、フェイルセーフ側（変換される側）に倒れる。
- 既定のresolverは上記設定で提供し、`PrivacyEngine(resolver=...)` で差し替え可能。

## 7. Policy と Operator

### 設計判断: modeの廃止

ドラフト案にあった「engine全体のmode（local/stable/reversible）」と
「entity_typeごとのoperator指定」の二重構造を廃止し、
トークン化戦略を `Tokenize` Operatorのパラメータに統一する。
グローバル設定とentity別設定の優先関係という問題を根本から消す。

```python
from privacy_masker.operators import Tokenize, Mask, Redact, Replace, Keep

policy = Policy(
    rules={
        "PERSON":  Tokenize(strategy="vault"),        # 復元可能
        "ORG":     Tokenize(strategy="hmac"),         # 決定的・鍵秘匿前提で復元困難
        "ADDRESS": Tokenize(strategy="sequential"),   # 連番
        "PHONE_NUMBER": Mask(char="*", keep_suffix=4),  # 03-1234-5678 → ********5678
        "EMAIL":   Redact(),                          # 完全削除
        "DATE":    Replace("<DATE>"),                 # 固定文字列
    },
    default=Tokenize(strategy="sequential"),
)
```

### フェイルセーフ原則

**未知のentity_typeは素通しさせない。** Detectorが検出したのにPolicyにルールが
ない場合、`default` Operator（未指定時の既定は `Tokenize(strategy="sequential")`）
を適用する。プライバシーライブラリにおいて「設定漏れ → PII素通し」は
最悪の故障モードであり、フェイルセーフ側に倒す。
素通しさせたい場合は明示的に `Keep()` operatorをルールに書く。

### Operator Protocol と TransformContext

```python
class Operator(Protocol):
    name: str
    def apply(self, value: str, entity: Entity, ctx: TransformContext) -> str: ...
```

`TransformContext` はカスタムOperatorの実装者向けの公開APIであり、
scope・Store・CryptoProvider・連番カウンタへのアクセスを仲介する。
OperatorはStoreやCryptoProviderを直接importしない。

```python
class TransformContext(Protocol):
    scope_id: str | None                                  # document scopeではNone

    def next_seq(self, entity_type: str) -> int:
        """scope内のtype別連番を採番する。"""

    def get_token(self, entity_type: str, normalized_value: str) -> str | None:
        """同一scope内で既に発行済みのトークンを引く（value_hash経由）。"""

    def save_token(self, entity_type: str, normalized_value: str, token: str,
                   original_value: str | None = None) -> None:
        """トークンを登録する。original_valueを渡すと内部で暗号化しvaultに保存する
        （CryptoProvider未設定ならConfigurationError）。"""

    def hmac_token(self, entity_type: str, normalized_value: str,
                   key: bytes | None = None, length: int = 8) -> str:
        """HMACトークンのID部を計算する（keyがNoneならengineのsecret_key）。"""
```

### v0.1同梱Operator

| Operator | 動作 |
|---|---|
| `Replace(text)` | 固定文字列に置換 |
| `Redact()` | 削除（空文字列に置換） |
| `Mask(char="*", keep_prefix=0, keep_suffix=0)` | 単純な文字数ベースのマスク（下記） |
| `Tokenize(strategy=..., template="<{type}_{id}>")` | トークン化（下記3戦略） |
| `Keep()` | 明示的な素通し |

**Maskのセマンティクス**: 値の構造（メールのドメイン等）は一切解釈しない、
純粋な文字数ベースのマスク。先頭 `keep_prefix` 文字と末尾 `keep_suffix` 文字を
残し、間をすべて `char` に置き換える。
**境界規則**: `keep_prefix + keep_suffix >= len(value)` の場合はkeep指定を無視して
**全文字をマスクする**（実質素通しになる構成をフェイルセーフ側に倒す）。
構造を保持するマスク（メールのドメイン保持等）はv0.1では提供しない。

### Tokenizeの3戦略

| strategy | 出力例 | 決定性 | 復元 | 必要な構成 |
|---|---|---|---|---|
| `sequential` | `<PERSON_001>` | scope内のみ | 不可 | scope_id指定時のみStore + secret_key |
| `hmac` | `<PERSON_MFZWK4TB>` | 同一key（+scope_id）で常に同値 | 鍵秘匿前提で困難（§14） | key必須。Store不要 |
| `vault` | `<PERSON_（base32 24文字）>` | scope内のみ | `restore()` 可 | scope_id・Store・Crypto・secret_key必須 |

- トークン書式は `template="<{type}_{id}>"` で変更可能。
- `sequential`: scope内でtype別に1から連番。ゼロ埋め3桁（設定可）。
- `hmac`: `HMAC-SHA256(key, encode(scope_component, entity_type, normalize(value)))`
  の先頭をbase32（RFC 4648: A〜Z, 2〜7）エンコードして8文字
  （`token_length` で変更可）。
  - `scope_component` は `scope_id`（指定時）または空文字列（document scope時）。
    つまりscope_id無しのhmacは**同一keyの下でプロセス・呼び出しを跨いで決定的**。
  - `encode()` は各フィールドをUTF-8化し8バイトbig-endianの長さプレフィックスを
    付けて連結する単射エンコード（`encoding.py`）。区切りなし連結による
    フィールド境界の曖昧さ（`"case_1"+"PERSON"` と `"case_1P"+"ERSON"` の衝突）を防ぐ。
  - keyは `bytes` 直接指定または環境変数名で指定（`key_env="PM_HMAC_KEY"`）。
    未指定ならengineの `secret_key` を使う。
  - 切り詰めによる衝突確率はドキュメントに明記する
    （8文字base32 = 40bit、10万エンティティで衝突確率 約0.5%程度 → 用途に応じて長くする）。
- `vault`: `secrets.token_bytes(15)`（120bit）をbase32エンコードした24文字を
  ID部とする乱数トークン（`token_bytes` で変更可、下限8バイトを強制）。
  トークン → `encrypt(encode(scope_id, token, 原文))` の対応をStoreに保存する。
  - Storeは `(scope_id, token)` の一意性を保証する（§9）。衝突時は `StoreError` を
    受けてengineが再生成する（最大3回、それでも失敗なら送出）。
  - 復号時は平文内の `scope_id`・`token` がリクエストと一致することを検証し、
    不一致（Store改竄・取り違え）は `RestoreError`。

### 正規化（normalize.py）

- すべてのTokenize戦略で、同一scope内の同値判定・HMAC入力の前に値を正規化する。
  既定は NFKC + 前後空白除去 + 連続空白の単一化。
  「山田太郎」と「山田　太郎」が別トークンになるのを防ぐ。
- entity_type別に正規化関数を差し替え可能（`Tokenize(normalizer=...)`）。

### presets

- `Policy.preset("llm_input")`: 全typeを `Tokenize(strategy="sequential")`。
  LLM API投入前のマスキング用。
- `Policy.preset("dataset")`: 全typeを `Tokenize(strategy="hmac")`。
  文書横断で結合可能な分析用データセット作成用。

### v0.2に延期するOperator

- `Generalize`（住所の市区町村への丸め等）: 日本語住所パーサが必要で
  実装が重いため延期。
- 構造保持Mask（メールのドメイン保持等）。

## 8. Scope（トークンの一貫性境界）

scopeは2種類に絞る（ドラフト案にあった `session` scopeは、
「MemoryStore + 固定scope_id」で同じことが実現でき、APIの2軸化
（scope文字列とscope_id）を招くだけなので**v0.1では提供しない**。
レシピとしてドキュメントで案内する）。

```python
engine.transform(text)                          # document scope（既定）
engine.transform(text, scope_id="patient_123")  # 明示ID・Store管理
```

| scope | mapping保持場所 | 一貫性の範囲 |
|---|---|---|
| `document`（既定） | 呼び出し内ローカル | 1回のtransform内 |
| `scope_id="..."` | Store | 同一Storeを見るすべての呼び出し（永続性はStore実装に依存: MemoryStoreならプロセス内、SQLiteStoreならプロセスを跨ぐ） |

**戦略 × scope の要求マトリクス**（構築時・呼び出し時検証の根拠。§3参照）:

| | document（既定） | scope_id指定 |
|---|---|---|
| `sequential` | 可。Store不要（呼び出し内カウンタ） | 可。Store + secret_key必須 |
| `hmac` | 可。Store不要（key依存でグローバル決定的） | 可。Store不要（scope_idはHMAC入力に入るのみ） |
| `vault` | **不可**（呼び出し時 `ConfigurationError`） | 可。Store + Crypto + secret_key必須 |

- vaultをdocument scopeで許すと「対応表がどこにも残らず復元不能なvaultトークン」
  ができてしまうため、構成エラーとして弾く。
- hmacはkeyとscope成分から純関数的にトークンが決まるため、常にStore不要
  （この利点をドキュメントで明示する）。

## 9. Store

### PseudonymStore Protocol

```python
class PseudonymStore(Protocol):
    def get_token(self, scope_id: str, entity_type: str, value_hash: str) -> str | None: ...
    def put_token(self, scope_id: str, entity_type: str, value_hash: str,
                  token: str, encrypted_value: bytes | None) -> None: ...
    def get_value(self, scope_id: str, token: str) -> bytes | None: ...  # restore用
    def next_seq(self, scope_id: str, entity_type: str) -> int: ...      # 連番カウンタ
    def delete_scope(self, scope_id: str) -> None: ...                   # scope単位の破棄
```

- **value_hash**: `HMAC-SHA256(secret_key, encode(scope_id, entity_type, normalized_value))`
  のhex表現。`secret_key` は `PrivacyEngine(secret_key=... | secret_key_env=...)` で
  供給する（`SecretBytes` として保持。§13）。
  **Storeに原文を平文で置かず、かつ無鍵ハッシュも置かない。**
  無鍵ハッシュ（単純なSHA-256等）では電話番号・郵便番号・人名辞書などの
  低エントロピーPIIが全数探索で復元できてしまうため、秘密鍵付きHMACを必須とする。
  scope_id指定の呼び出しでsecret_key未設定なら `ConfigurationError`（§3）。
- `encrypted_value` はvault戦略のときのみ非None。
- `put_token` は `(scope_id, token)` の一意性を保証し、違反時は `StoreError` を送出する
  （vaultトークン衝突の検出。engineが再生成する。§7）。
- `delete_scope`: 案件終了時の対応表破棄・保持期限・消去請求対応のための最低限の
  削除API。対応表は「復元につながる情報」であり、作れるが消せないStoreは
  プライバシーSDKとして不適格。トークン単位の削除はv0.2で検討。
- v0.1同梱: `MemoryStore`（既定）、`SQLiteStore(path)`。
- `SQLiteStore` はWALモード + プロセス内ロックで同一プロセス多スレッドまで保証。
  マルチプロセス同時書き込みはv0.1のスコープ外（ドキュメント明記）。
- Postgres / Redis / DynamoDB はProtocolを満たす外部実装として利用者が書ける。
  公式提供はv0.2以降に検討。core packageにDB依存を入れない。

## 10. Crypto

### CryptoProvider Protocol

```python
class CryptoProvider(Protocol):
    def encrypt(self, value: bytes) -> bytes: ...
    def decrypt(self, value: bytes) -> bytes: ...
```

- vault戦略の対応表暗号化にのみ使用する。
- 暗号化する平文は原文単体ではなく `encode(scope_id, token, original_value)`
  （§7と同じ単射エンコード）。復号時にscope_id・tokenの一致を検証することで、
  Store上で暗号文を行間で入れ替える取り違えを検知する（FernetはAADを
  サポートしないため、平文へのコンテキスト内包で代替する）。
- v0.1同梱: `FernetProvider`（`[crypto]` extra、cryptographyパッケージ）。
  AES-GCMのnonce管理のような誤用しやすい低レベルAPIは公開しない。
- vault戦略を含むPolicyでCryptoProvider未指定なら `PrivacyEngine.__init__` で
  `ConfigurationError`（構築時検証）。
- 鍵ローテーションはv0.1スコープ外（ドキュメント明記）。

## 11. restore

```python
result = engine.transform(text, scope_id="case_42")
# ... LLMがresult.textを加工した出力を返す ...
restored = engine.restore(llm_output, scope_id="case_42")
```

動作仕様:

- engineはPolicy内の全Tokenizeルールの `template` からトークン検出用の正規表現を
  導出する（`{type}` → `[A-Z][A-Z0-9_]*`、`{id}` → `[A-Za-z0-9]+`）。
  複数のtemplateが併用されていれば全パターンを走査する。
- マッチした各トークンをvault（`store.get_value`）から引き、復号して置換する。
  vaultに無いトークン（sequential / hmac由来、本文中の偶然のトークン様文字列を含む）
  はそのまま残し、`RestoreResult.unresolved` に列挙する。
- 未知の `scope_id` はエラーにしない（Storeは「存在しないscope」と「空のscope」を
  区別しない）。この場合は全トークンがunresolvedになり正常終了する。
  `RestoreError` はStore I/O失敗・復号失敗・コンテキスト不一致（§10）のときのみ。
- **restoreできるのはvault戦略のトークンのみ。**

法的概念との対応（ドキュメントに記載）:

- 対応表を保持する仮名化は個人情報保護法上の「仮名加工情報」に相当し得る。
  対応表（Storeファイル）・secret_key・HMAC鍵・Fernet鍵はいずれも復元につながる
  情報として安全管理措置の対象になる。
- ライブラリとして法的適合性（匿名加工情報の基準への適合等）を保証するものでは
  ない旨を明記する。

## 12. エラー処理

```
PrivacyMaskerError            # 基底
├── ConfigurationError        # 構成・引数の組み合わせ不整合。構築時に検出できるものは
│                             #   構築時に、呼び出しで初めて判明するものは呼び出し時に送出
│                             #   （例: vault+Crypto無し、vault+scope_id無し、key未設定）
├── DetectorError             # 検出器の実行時失敗・不正なEntityの返却
├── StoreError                # Store I/O失敗・一意性制約違反
└── RestoreError              # 復元失敗（復号失敗・コンテキスト不一致 等）
```

- **例外メッセージ・ログに入力テキスト・検出値・鍵を一切含めない。**
  DetectorErrorは「どのdetectorが失敗したか」は伝えるが入力は含めない。
- Detectorが1つ失敗したときの既定動作は**全体失敗（例外送出）**。
  「NERが落ちたのでregex分だけマスクして返す」は静かなPII漏洩につながる。
  部分成功を許容する場合は利用者が明示的に
  `PrivacyEngine(on_detector_error="skip")` を指定する（既定は `"raise"`）。
  skip時は `TransformResult.failed_detectors` に失敗detector名が入り、
  劣化した結果であることを下流が機械的に検知できる（§4）。
- 空文字列・空白のみの入力はエラーではなく無変換で返す。

## 13. セキュリティ設計原則（横断）

1. 型がPIIを持たない: `Entity` / `TransformResult` に原文フィールドを設けない。
2. `__repr__` / `__str__` は原文由来の文字列を含まない。
   `TransformedEntity.replacement` は原文文字を含み得るためreprから除外する（§4）。
3. 例外・ログに入力テキストを埋め込まない。ライブラリ自身はロギングを行うが
   （`logging.getLogger("privacy_masker")`）、出力するのは件数・type・所要時間のみ。
4. **シークレットを漏らさない**: HMAC key・secret_key・Fernet keyは `SecretBytes`
   ラッパー型で保持し、reprは `SecretBytes(****)` に伏せる。例外メッセージ・ログにも
   含めない。トレースバック収集基盤（エラーレポーティングサービス）経由の漏洩を防ぐ。
5. Storeには秘密鍵付きHMACの `value_hash` と暗号化済みバイト列のみを置く（§9）。
6. 乱数トークンは `secrets` モジュールを使用（`random` は使わない）。
7. フェイルセーフ: ルール未定義entityは既定でトークン化（§7）、label_map未登録
   ラベルはpass-through（§5）、Mask境界超過は全マスク（§7）、Resolver残余は
   再評価（§6）、Detector失敗は既定で全体失敗（§12）。素通しは明示時のみ。
8. 鍵のハードコードをドキュメントで明確に非推奨とし、
   環境変数・シークレットマネージャ経由を案内する。

## 14. 脅威モデルと限界（ドキュメントに明記する内容）

- **hmac戦略は「鍵の秘匿を前提とする仮名化」である。** 鍵保持者（および鍵漏洩後の
  攻撃者）は候補値の全数HMAC計算で照合でき、電話番号・生年月日・人名辞書などの
  低エントロピーPIIは事実上復元可能。「復元不能」ではなく「鍵が守られている限り
  復元困難」と位置付け、匿名加工情報に相当すると誤認させない。
- **restoreはオラクルになり得る。** restoreを呼べる主体は実質的にscope内の全PIIに
  アクセスできる。restoreの認可・監査はライブラリの責務外であり、アプリケーション
  側で制御する。vaultトークンのID部を十分長くする（既定120bit）ことで、
  トークン総当たりによる列挙を非現実的にする。
- **Storeの防御は限定的。** 暗号文のコンテキスト内包検証（§10）で取り違えは検知
  するが、Storeへの書き込み権限を持つ攻撃者への包括的防御・Store持ち出し時の
  保護（value_hashへの辞書攻撃は秘密鍵が漏れない限り不可）はv0.1の脅威モデル外。
- **検出漏れは起きる。** NER・regexの再現率は100%ではなく、本ライブラリは
  匿名化の完全性を保証しない。`detect()` による事前確認・サンプリングによる
  人手レビューの併用を推奨する。

## 15. スレッドセーフティ

- `document` scope: 呼び出しローカルの状態しか持たないため並行呼び出し安全。
- `scope_id` + Store: `MemoryStore` はロックで保護。`SQLiteStore` は
  同一プロセス多スレッドまで保証（§9）。`next_seq` の採番はStore実装が
  アトミック性を保証する。
- `PrivacyEngine` 自体はimmutableな構成（detectors / policy / resolver）を持ち、
  構築後の変更APIは提供しない。

## 16. テスト戦略

- pytest。coreのユニットテストはstdlibのみで完結させる
  （FakeDetectorをProtocolで注入。transformersのモック不要）。
- 重点項目:
  1. offset整合性: 複数entity、絵文字・サロゲートペアを含む日本語テキストでの置換位置検証
  2. overlap解決の網羅: 包含 / 部分重複 / 連鎖重複 / 同点タイブレーク / 残余スパンの再評価
  3. HMAC決定性: 同一key+scope → 同値、異なるscope → 別値、正規化の効き、
     単射エンコードの境界（`"case_1"+"PERSON"` vs `"case_1P"+"ERSON"` が別トークン）
  4. vault round-trip: transform → restore → 原文一致。unresolved列挙。
     トークン衝突時の再生成。コンテキスト不一致（改竄）検知。未知scope_idの正常終了
  5. フェイルセーフ群: 未知typeのdefault落ち / label_map未登録のpass-through /
     Mask境界超過の全マスク / skip時のfailed_detectors
  6. 戦略×scopeマトリクス: 許容組み合わせと各ConfigurationErrorの発生
  7. extras境界: 「extras無しでcoreがimportできる」をCIの独立ジョブで検証
  8. TransformersNerDetector: 小型モデルまたは録画済み出力によるoffset変換と
     スライディングウィンドウ再結合の検証
  9. シークレット非漏洩: SecretBytesのrepr、例外メッセージに鍵・原文が含まれないこと
- CI: GitHub Actions。Python 3.10 / 3.11 / 3.12 / 3.13 マトリクス。
  ruff（lint + format）+ mypy strict。`py.typed` を同梱。

## 17. パッケージング

- `pyproject.toml`（hatchling想定）。
- `pip install privacy-masker` はcoreのみ（依存ゼロ）。
- extras: `[transformers]` `[crypto]`、v0.2で `[pandas]`。
- extra未導入で該当クラスをimportした場合は、必要なextraを案内する
  `ImportError` を送出する（例: `pip install 'privacy-masker[crypto]'`）。

## 18. v0.1スコープ（確定）

| 領域 | 含むもの |
|---|---|
| パイプライン | Detector → Resolver → Policy → Operator → Result の全段 |
| Detector | RegexDetector（日本語プリセット6種）、TransformersNerDetector（長文チャンク対応） |
| Operator | Tokenize（sequential / hmac / vault）、Mask、Redact、Replace、Keep |
| Policy | rules + default fallback、presets 2種（llm_input / dataset） |
| Scope | document / scope_id（sessionは提供しない） |
| Store | MemoryStore、SQLiteStore（delete_scope含む） |
| Crypto | FernetProvider（コンテキスト内包検証付き） |
| API | detect / transform / transform_batch / restore |

### v0.2以降のロードマップ

- Generalize Operator（住所の市区町村丸め。日本語住所パーサ）
- 構造保持Mask（メールのドメイン保持等）
- pandas DataFrame対応（`[pandas]` extra、column_policy）
- Postgres / Redis Store公式提供
- 鍵ローテーション
- トークン単位の削除API
- LLMベースDetector
- CLI（ファイル一括処理）

## 19. 参考

- Google Cloud Sensitive Data Protection: infoType（検出）とtransformation（変換）の分離、一方向 / 双方向トークンの使い分け
- 個人情報保護委員会: 仮名加工情報 / 匿名加工情報の区別、対応表の安全管理措置
- cryptography公式ドキュメント: 高レベルAPI（Fernet）推奨、AES-GCM nonce再利用の危険性

## 20. v0.2追補: SpanRefiner（境界リファインメント）— 2026-08-12

実モデル（ruri-v3ベースNER）の運用検証で、tokenizerの語彙結合（「田中さん」「日本の」が
1トークン）により検出スパンが敬称・助詞・空白を巻き込み、(1) vault/sequentialの同値判定が
表層ゆれで壊れる（同一人物・同一scope_idでも別トークンになる）、(2) スパン境界が不正確になる、
という課題が確認された。対処として `SpanRefiner` を追加する。

- 配置: `privacy_masker/refiner.py`。`PrivacyEngine(refiner=...)` で注入、**既定で有効**
  （`refiner=None` で無効化）。検出直後・Resolver適用前に働く
- 規則: ①スパン端の空白・改行トリム（全type）②末尾敬称トリム（既定はPERSONのみ。
  さん/さま/様/氏/くん/ちゃん/君/殿）③端の助詞runトリム（の・は・が等、累積で片側最大2文字のハードキャップ。
  **削った後に端に残る文字がひらがなの場合は削らない** — 「りの」のようなひらがな終わりの
  名前の誤削り防止）
- 安全性: 削る対象は非PII文字のみなので検出漏れ（危険側）は増えない。既知の限界として
  「田中さんの」型（敬称+助詞連続）は助詞側が残る（安全側）
- restoreとの関係: リファイン後のスパンが保存されるため、復元は各文書の表層形どおりに戻る

## 21. v0.3追補: Surrogate operator（自然な日本語surrogateへの仮名化）— 2026-08-13

`<PERSON_001>` 型トークンはLLM学習データの分布を歪めるため、実在風の日本人名・
企業名・住所へ置換する `Surrogate` operatorを追加する（`operators/surrogate.py`）。

- 種別: `kind="person" / "org" / "address"`。戦略: `random`（scope内一貫・Store経由）
  / `hmac`（鍵付きHMACをシードにプールから決定的選択。Store不要で文書横断一貫）。
  **vault非対応 = restoreは一方向**（復元要件は `Tokenize(vault)` と使い分け。
  自然文字列はトークン書式を持たず復元走査に乗らないため）
- 表層特徴の保存: 文字種（漢字/カタカナ・「・」区切り）、姓のみ（正規化後3文字以下）/
  フルネーム、空白区切り、法人格の位置（株式会社の前置き/後置き・（株））、
  既知種別語（銀行・信用金庫・大学等）、住所の丁目表記・建物名有無
- 同一性: 正規化同値 → 同一の偽名実体。表記（空白等）は各出現の原文に追従するため、
  同一実体の出現間で文字列は正規化後一致（完全一致ではない）
- 衝突対策: 生成値が原文と一致したら再抽選（最大10回）。random戦略はStoreの
  トークン一意性（DocumentContextにも同ガードを追加）で別実体との衝突を検知し
  再生成（最大5回）。hmac戦略は無状態のため別実体間の偶然の一致は許容（実世界の
  同姓同名と同じ性質としてドキュメントに明記）
- データ: `privacy_masker/data/ja_pools.py` に自作プールを同梱（姓140・名100・
  カタカナ名60・企業語幹・実在の都道府県+市区町村ペア66・汎用町域名）。
  生成器は `Surrogate(generator=...)` で差し替え可能
- 電話・メール等のkindはv0.3では提供しない（日本には安全な例示用番号帯がなく、
  実在番号との衝突リスクの設計判断が別途必要なため）

### §20補足: MorphologicalRefiner（v0.3.2、`[janome]` extra）

ルール式SpanRefinerの上位互換として、Janome（純Python形態素解析器）の品詞情報で
スパン端の付属語（助詞・助動詞・記号・人名接尾）を削る `MorphologicalRefiner` を追加。
MeCab事前分割方式（初期の学習ノートブック）と異なり検出後の後処理のみなので、モデルの再学習や
学習/推論の分割器一致は不要。ルール式のひらがなアンカー制約（みなと+まで、などでは連鎖）を
品詞判定で解決する。スパン端が形態素の途中に落ちる場合は削らない（安全側）。
全体が付属語のスパンは検出ノイズとして破棄する。既定はSpanRefinerのまま（依存ゼロ維持）。

## 22. v0.4追補: 姓リンク（Surrogateの同一人物連携）— 2026-08-14

Surrogateでフルネームと姓のみ言及が別偽名に割れる問題（山田太郎→山本洋平なのに
署名の山田→伊藤）への対処。`Surrogate(kind="person", link_surnames=True)`（既定有効）。

- **設計原則: 同じ原姓は偽名でも姓を共有する（同姓構造の保存）**。山田太郎と山田花子が
  同一scopeにいても曖昧性は生じず、両者とも同じ偽姓+各自の名になる
- **hmac戦略は無状態のまま成立**: 偽姓を「姓部分のみのHMAC決定値」から引くため、
  裸の「山田」とフルネーム「山田*」の偽姓が定義により一致する。前方・後方の出現順に
  依存せず、Store不要・決定性も維持
- random戦略はscopeの対応表（通常mappingと `\x00sei:` プレフィックスの別名エントリ）で
  前方リンクし、裸姓が先に割当済みの場合はその偽姓をフルネームが引き継ぐ（後方リンク）。
  エンジンの文末側からの処理順でも後方リンクにより文書内一貫性が成立する
- 対象: 漢字名かつ既定generator。原姓の分割は同梱姓プールの最長前方一致で行い、
  プール外の姓（勘解由小路等）・カタカナ名・カスタムgeneratorはリンク対象外
  （独立偽名のまま。安全側）
- **v0.4はSurrogateの人名生成規則の破壊的変更**（偽姓の導出が変わる）。ゴールデン値
  テストを更新済み

### §22補足: hmac+姓リンクの相関特性（レビュー指摘の受容判断）

偽姓は(鍵, scope, 原姓)のみで決まるため、同じ鍵・scopeでは無関係な人物でも同姓なら
同じ偽姓に収束する（等価類が姓プール約140種に縮小）。姓の分布の偏りにより、大量出力
への頻度分析で偽姓→原姓の推測性が上がる点をトレードオフとして受容し、ドキュメントに
明示する。緩和策: scope_id分割 / link_surnames=False / random戦略。

## 23. v0.5追補: バッチ推論によるCPU/GPUスループット改善 — 2026-08-15

LLM学習データ等の大量文書処理向け。従来のdetect_batchはチャンク毎にpipelineを
1回ずつ呼ぶ素朴実装で、detectorのbatch_sizeは未使用だった。

- `_run_pipeline`: チャンク列を長さ順にソートして1回のリスト呼び出し
  （`pipeline(chunks, batch_size=N)`）に集約し、結果を元順序に復元。
  パディング量が揃いend-to-end実測でCPU約1.9倍・MPS約3.8倍
- `detect_batch`: 全文書の全チャンクを(文書index, offset)台帳付きでフラット化し
  一括推論後に文書毎へ再集約。dedupは文書毎のseen集合で従来と同一のセマンティクス
- 後方互換: HF Pipelineインスタンスのみバッチ経路（isinstance判定、import失敗時は
  逐次）。str専用カスタムcallableは従来どおり逐次。`pipeline_batching`で明示上書き可
- `device`パラメータ追加（自前構築時のみ。カスタムpipeline併用はConfigurationError）。
  batch_size既定は8→16（CPU実測の最良点）
- 正しさの検証: 当時の 30m モデル・混合175文書で逐次とバッチのエンティティ完全一致
- int8動的量子化は不採用（ARM/qnnpackでfp32バッチより遅い実測。将来はONNX/optimumを
  カスタムpipeline注入で使う方針）

## 24. v0.6追補: 語境界スナップ（MorphologicalRefiner拡張） — 2026-08-21

医療文書での実運用比較で判明した知見への対応: 旧構成（stockmarkのみ学習+MeCab語単位
予測）の境界の綺麗さの核心はモデルでなく語境界スナップだった。一方で旧構成は裸姓
（松本・皆見等）や複合組織名の見逃しが多い（難例cov 74% vs 当時の 70m モデル 98.6%）。
新モデルのcoverageと旧構成の境界品質を両立するため、後処理として移植する。

- スパン端が形態素の途中 → 境界まで拡張（鮫|島病院→鮫島病院、青葉電力送→送配まで）
- スナップで重なった同typeスパンは統合。異typeの重なりはEntityResolverの責務
- 単一形態素の内部断片は、形態素が固有名詞なら拡張（山→山田）、それ以外は棄却
  （拡張の「拡」→PERSON等の誤検出パターン）
- 英字1〜2文字だけのスパンは拡張前に棄却（S・NS・di等。JR単独等の2字組織名と
  裸のイニシャルは犠牲になるトレードオフを許容、drop_ascii_len=0で無効化可）。
  **トリム後の再判定はしない**: 「Kさん」の敬称トリムで残るKを捨てると平文漏れになる
  （v0.6レビューのCritical指摘で実証・修正済み）
- 内部断片の棄却は非名詞系形態素に限定: 名詞,一般はIPADIC非収載の実名（東雲・琥珀）が
  該当し得るため棄却でなく拡張で守る（同レビューImportant対応）
- 拡張の結果、全体が付属語runとして空になる場合は拡張前スパンを無傷で残す
  （つじの等、誤解析ひらがな名の欠落防止。旧「途中の端は触らない」と同じ安全側）
- 実測（当時の 70m モデル+snap、CPU固定）: 実文書評価セット・難例セットのcoverage・FP字数は無劣化のまま、
  医療文書のS・NS・di誤検出と鮫島病院の分割が全て解消
- min_score（検出器の信頼度閾値）も追加したが既定0.0: 0.6でcoverageが約1pt落ちる
  （見逃し増）ため、精度優先ユーザー向けのオプトインに留める

## 25. v0.7追補: 自立語トリム（動詞句・接続詞） — 2026-08-25

レビュー指摘「松本駅から歩いて はNERとして問題」への対応。v0.6の語境界スナップは
付属語（助詞・助動詞・記号）しか削らないため、エンティティの外側にある動詞句が残っていた。

- 追加: 端の外側にある自立語も削る。閉クラス（接続詞・副詞・感動詞・フィラー・連体詞）は
  隣接していても削り、開クラス（動詞・形容詞）は付属語で隔てられている場合のみ削る
- **必須の安全制約: 削る自立語は漢字を含む形態素に限る**。実測でJanomeは仮名の名前を
  頻繁に自立語と誤解析する（みずき→み(動詞)+ず+き / イザベラ→イザ(副詞)+ベラ /
  あべ→フィラー+助詞 / おおた→感動詞 / あおき→形容詞 / つむぎ→動詞）。
  この制約なしで実装すると「みずき信用金庫」が「信用金庫」に、
  「イザベラ・フェレイラ」が「ベラ・フェレイラ」に欠けることを実測で確認した
- 開クラスに付属語の介在を要求する理由: 「皆見」が 皆+見(動詞) と解析されるため
- スパン全体が消える方向のトリムは付属語のみのrunに限る（ひらがな名の保護）
- 実測（11ジャンル878エンティティ、当時の 30m / 70m モデル）: coverageは完全に不変（PII欠け0件）、
  過剰マスクのみ減少（550→542字 / 469→464字）。**利得は小さいが一方向の改善**
- 根本原因の分析: 合成データ（テンプレート）ではエンティティ直後に助詞+動詞句が続く
  文脈が2.7%しかなく、wiki実テキストの6.3%より少ない。テンプレート方式では
  エンティティ後続文脈の多様性が作者の想像力に制限されるため、この種の境界エラーは
  データ側でも起きやすい（学習データでLLMパラフレーズを増やした動機）

### §25補足: v0.7レビューで撤回した範囲とPII漏れ既存バグの修正 — 2026-08-25

初版実装（接続詞・副詞も隣接トリム）はレビューで**Critical 2件のPII漏れ**が実証され撤回した。
教訓は「11ジャンル評価セットが該当表層を含んでいなかった」= 評価の網羅性不足である。

- 撤回1: 閉クラス（接続詞・副詞・感動詞・フィラー・連体詞）の隣接トリム。IPADICは
  尚/又/将/唯/只/猶/彌/宜/土台/一寸 等の**一般的な人名用字**をこれらに解析するため、
  「尚子」→「子」「将也」→「也」「又川産業」→「川産業」と漏れる（実証済み）
- 撤回2: 隔ての付属語に 記号 と 助詞,連体化 を認める設計。PII内部の記号・「の」で
  分断されるため（見の越駅→見の / 与々木→木 / 「田中␣走」→田中 / 佐藤・誠→佐藤・）
- 併せて修正した**v0.6以前からの既存PII漏れ**（新設の総当たり検査で発見）:
  - 漢字の助詞（「有之」の之）が末尾トリムで削られる → _TRIM_POS判定に「漢字を
    含まない」を追加（敬称の様・氏は _HONORIFIC_POS 経路なので従来どおり削れる）
  - 実名が動詞形態素の内部断片になる場合（「見越さ」内の「見越」）にスパン棄却
    → 1文字の断片のみ棄却に変更（「拡張」の拡 は引き続き棄却）
- 恒久的な検証手段: 名前2,529種×文脈15種=37,935通りの総当たりでPII欠けを検査する
  ストレススクリプトを整備（`tools/stress_truncation.py`）。0件を確認

### §25再改訂: 自立語トリムの撤回とPII漏れ既存バグ3件の修正 — 2026-08-25

2回のレビューを経て**自立語トリム自体を撤回**した。教訓を残す。

- 撤回理由: IPADICは 之・ノ・が を助詞,格助詞と解析するため、どんなPOS条件を足しても
  表記ゆれ経由でPIIが分断される（勝之進→勝 / 元湯ノ出→元湯 / 浅が野→野 / 田之見→田。
  いずれもend-to-endで平文残存を実証）。得られる利得は過剰マスク数文字で、
  匿名化ライブラリとしての損益が合わない。「松本駅から歩いて」型は**モデル側**
  （エンティティ直後に動詞句が続く学習データ。合成データは2.7%でwiki実テキストは6.3%）
  で対処すべき問題である
- 副産物として発見・修正したv0.3.2以来の既存PII漏れ3件:
  1. 漢字・カタカナの助詞表層（之・ノ・ヲ・ワ・デ）が付属語トリムで削られる
     → _TRIMMABLE_SURFACE_RE（ひらがな＋記号のみ）で表層を検査する方式に変更
  2. スパン全体が付属語と解析されると棄却され、実在のかな・カナ人名
     （チャン・かな・りか・よね・さえ・はな…）が**検出済みでもマスクされない**。
     既定のSpanRefinerでは正しくマスクされるため、形態素refinerが危険側に劣化していた
     → 1文字以下のみ棄却し、それ以外は原スパンを残す
  3. 実名が動詞形態素の内部断片になる場合（「見越さ」内の見越）にスパンごと棄却
     → 1文字の断片のみ棄却
- 恒久的な検証手段: 名前3,080種×文脈15種=46,200通りの総当たりPII欠け検査
  （`tools/stress_truncation.py`）。0件を確認。**評価セットのcoverageだけでは
  この種の漏れは検出できない**（該当表層が含まれないため）という教訓が最大の学び

## 26. 未リリース追補: batch対応付けとHF token上限のfail-closed化 — 2026-08-26

- `PrivacyEngine.transform_batch()` はbatch detectorの返却件数が入力件数と一致することを、
  Entityの検証・コミット前に確認する。不一致は`DetectorError`とし、
  `on_detector_error="skip"`でも該当チャンク全件をロールバックして文書との取り違えを防ぐ
- `TransformersNerDetector`の外側の400文字・50文字overlapは、互換性と推論コスト上限のため
  維持する。ただし文字数はtoken上限を保証しないため、genuine HF
  `TokenClassificationPipeline`では特殊token込みの実token長を推論前に確認する
- 上限超過時はfast tokenizer・aggregation有効・tokenizer/modelの最大長が整合する場合だけ
  HF標準strideを使用する。いずれかを満たさない、または安全な上限を特定できない場合は
  `ConfigurationError`で停止し、後半のサイレント切り捨てを禁止する
- `tokenizer`属性を偶然持つwrapperを含め、任意の非HF custom callableにはこの検査や
  `stride` keywordを適用せず、従来のstr/list・`batch_size`契約を維持する
- 回帰検証: 小型実HFモデルで上限後方の検出と絵文字を含むcodepoint offsetを確認。
  当時の 30m モデル・評価327文書では旧経路と新経路の1,041 entity tupleが完全一致した

## 27. v0.9追補: 意味を保つ Surrogate と往復 — 2026-09-24

設計文書 `docs/superpowers/specs/2026-09-24-semantic-surrogate-design.md`（実装計画は開発記録、非公開）。決定事項:

1. **逆算不能が must**。偽名が元の値に依存するのはカテゴリと表層特徴（母集団数千以上の粗い属性）と
   鍵付き HMAC のシードだけ。「元の値の埋め込みに最も近い候補を選ぶ」方式は攻撃者が同じモデル・
   候補集合で写像を再現できるため不採用（`Surrogate` docstring、`operators/surrogate.py`）
2. **NER encoder のスパンベクトルを流用**（別埋め込みモデルは使わない）。`CharHeadNerDetector(span_vectors=True)`
   が `Entity.vector` に付け、分類ヘッド（計画 B、未着手）が使う想定。現行は種別語の規則（`RuleCategorizer`）
   だけでカテゴリを決め、`category_head=None` なら規則のみ
3. **hmac は best effort**。文書内の既存表層を回避するため（`SurrogateFilter.present_in_document`）、
   同じ値・同じ鍵でも文書によって偽名が変わりうる。対象は `semantic=True` の全経路と、
   `semantic=False` でも **PERSON の姓リンク経路**（`_draw_sei` が `text` を渡す）。`semantic=False` の
   legacy `_draw`（ORG・ADDRESS、および姓リンクに入らない PERSON）は `text` を渡さないので文書に依存しない。
   したがって v0.8 とのバイト互換は「文書全文を渡さない場合」または「文書全文を渡しても legacy `_draw` だけを
   通る型」で成立する
4. **往復は完全一致のみ**。`Surrogate(strategy="vault")` + `Engine.restore()`。偽姓は 2 字以上のみ
   保存対象（`_restorable`）、`<TYPE_xxxxxxxx>` フォールバックトークンも vault では値付き保存され復元できる

破壊的変更: `semantic=True` が既定になり hmac の偽名が変わる（`semantic=False` で v0.8 互換）、
自作 `PseudonymStore` は `iter_tokens(scope_id)` が必須（vault 復元）、同梱禁止リストは
`semantic=False` でも適用、`Policy.has_strategy` が Tokenize 以外の operator も見る。
計画 B（`Entity.vector` を使う分類ヘッド）は本追補では未着手。

## 28. 2026-09-28 追補: 分類ヘッド（計画 B）の実装と不採用

設計 `docs/superpowers/specs/2026-09-24-semantic-surrogate-design.md` §8（計画と結果は開発記録、非公開）。§27 の 2. で「未着手」とした分類ヘッドを実装した。決定事項:

1. **読み込みは検出器側**。設計 §8 の「`load_char_model` が読む」ではなく、`CharHeadNerDetector(category_head="auto")` がモデルフォルダの
   `category_head.json` を `HeadCategorizer.load` で読む（`True` で必須、`False` で読まない）。`load_char_model` は NER モデルだけを返す。
   `privacy_masker.category_head` は torch / numpy / safetensors をトップレベルで import しない（推論は純 Python、safetensors は読み込み時だけ）
2. **`Surrogate(min_margin=None)` が既定**: `None` ならヘッドの推奨値（`category_head.json` の kind 別 `min_margin`）、ヘッドが無ければ 0.15。明示した値が優先
3. **ヘッドが無くても警告しない**（規則のみが v0.9.0 の正常な構成）。encoder SHA の不一致・壊れたファイル・min_margin の欠落は `ConfigurationError`
4. **判定: 全 kind 不採用、v0.9.0 のカテゴリは規則のみ**。ORG は valid で (i) 0.905・(ii) min_margin 0.55 で precision 0.967 を満たしたが、
   実文書で種別語を削ったスパンでは同じ margin で 0.741（n=27）。FACILITY・LOCATION は valid の (ii) にも届かない
5. **候補は本番モデルフォルダの外に置く**。学習ツールは既定で候補に書き、kind ごとの `accepted`
   （(i)(ii) の両方）を記録し、モデルフォルダへは `--install`（`accepted` の kind だけ）でしか書かない。`accepted` は valid での必要条件で、
   採用には実文書での確認が要る（min_margin は実文書の dev 分割で選び、固定 OOD・盲検では選ばない）

**2026-09-29 追補: 分類ヘッド v2（実文書の Qwen ラベル、第 1 段）**（設計・計画・結果は開発記録、非公開）。決定事項:

6. **特徴量 `features`**: `span_mean`（スパンに重なるサブワードの隠れ状態の平均、30m で 256 次元、従来どおり）と `span_ctx`（[span_mean; 直前 16 サブワードの平均; 直後 16 サブワードの平均]、
   区画ごとに L2 正規化して連結、768 次元。文脈は同じチャンクの内容トークンの範囲内だけで、足りなければある分、無ければゼロ）。`Entity.vector` は float16 LE のバイト列のまま
7. **ヘッドの種類 `head_type`**: `linear`（従来、純 Python 推論）と `mlp`（Linear → GELU(tanh) → Linear、numpy で推論。numpy は関数内で遅延 import し、
   `privacy_masker.category_head` のトップレベルの import 制約は据え置き）。`category_head.json`（version 2）が `head_type` / `features` / `hidden` を宣言する
8. **検出器はヘッドの `features` に従う**: `CharHeadNerDetector.vector_features` は読み込んだヘッドの `features`、ヘッドが無ければ `span_mean`（ゴールデン値は不変）。
   `HeadCategorizer.load(model_dir, head_dir=None)` は未知の `head_type` / `features`、重みの入力次元と `features` の不一致（encoder の `config.json` の `hidden_size` × 1 / × 3）を
   `ConfigurationError` にする。`head_dir` を渡すとヘッドはそこから読み、encoder の SHA は `model_dir` と照合する（候補フォルダのヘッドを本番 encoder で評価する用）
9. **選択と判定の分離・門**: 学習ツールは `head_dev` だけで選び（MLP の早期終了・min_margin・統合候補）、`--judge` を付けたときだけ `head_judge` を読んで kind ごとに 1 回判定する
   （(i) judge の規則ラベル付きスパンの macro accuracy ≥ 0.90、かつ (ii) judge の種別語の無いスパンで dev の min_margin の precision ≥ 0.95 → `accepted`）。
   `--install` は `--judge` 必須で `accepted` の kind だけをモデルフォルダに書く（`--target-precision` < 0.95 は拒否、従来どおり）
10. **判定: v2 も全 kind 不採用**。ORG は dev のどの変種（線形 / MLP × 2 特徴量、統合 1 回）でも min_margin が決まらず（judge (i) 0.869）、facility は judge (ii) 0.878、
    location は judge (i) 0.820・(ii) 0.833。`--install` は実行しておらず、**本番モデルフォルダは無変更**（カテゴリは規則のみ）。
    分類ヘッド v2 の設計（開発記録、非公開）で定めた第 2 段（encoder ごとの多タスク学習）の発動条件（ORG が judge で不合格）を満たし、実行は別計画とした

**2026-10-02 追補: 分類ヘッド 第 2 段（encoder ごと多タスク学習）**（設計・計画・結果は開発記録、非公開）。決定事項:

11. **学習専用の部品を `char_head.py` に足した**: `CategoryAuxHead`（kind 別の `nn.Linear(3H, k)`、`CharBioNerModel` の重みには含めない）、`category_loss`（batch 内の全カテゴリスパンの
    kind 別 cross entropy の平均、カテゴリスパンが無ければ 0）、`span_ctx_batch`（`span_vector(features="span_ctx")` と同じ規則の batched torch 実装。float16 への丸め後にバイト一致、
    L2 正規化は分母にも勾配を流す）、`encode_chunk(category_spans=)`（`category_spans` はチャンク相対の文字位置で渡す。呼び出し側 = ノートブックの `build_samples` がチャンクに完全に含まれるものだけを相対化して渡し、範囲外は `ValueError`、切れたスパンの件数はノートの `n_cat_cut`）、
    `CharNerCollator` の `cat_spans`（`(B, S_max, 4)` long）/ `cat_mask`、`sqrt_class_weights`、`category_spans_by_text`、カテゴリ語彙 `CATEGORY_KINDS` / `CATEGORY_NAMES`
    （`semantic.CATEGORIES` との一致をテストで固定）。`char_head.py` は学習ノートブックがデータフォルダ（`PRIVACY_MASKER_DRIVE_DIR`）から単体 import するので、パッケージ相対 import は足していない
12. **推論形式は不変**: `Entity.vector`・`span_mean` / `span_ctx` の値、`save_char_model` / `load_char_model` の保存形式、ゴールデン値は変わらない。
    `category_spans` を省略した `encode_chunk` は従来どおりで、collator の追加キーは `CharBioNerModel.forward(**_unused)` が無視する（既存の学習ノートもそのまま動く）。
    補助層の重みはエポックごとの `category_aux.safetensors`（記録用。読み込む側は無い）
13. **判定: 不採用、本番モデルは差し替えない**。本番と同じレシピ + `0.3 × cat_loss` で学習した encoder は
    盲検セットと 4 評価セットで NER の採用条件を満たさず、その上で作り直したヘッドも
    judge-B で受け入れた kind が無かった（judge に回した linear / span_mean は judge-A で 3 kind とも min_margin が決まらない）。本番モデルは無変更で、カテゴリは v0.9.0 と同じく規則のみ
