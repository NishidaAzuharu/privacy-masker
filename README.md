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
- **意味を保つ偽名への置換（Surrogate）**: 人名は性別傾向と姓リンク（同じ姓は同じ偽姓）を保ち、組織・施設・地名・住所は同じ粗いカテゴリの実在風の別名に置き換えます（病院 → 別の病院名、製造業 → 別の製造業の社名、東北の市 → 東北の別の市）。偽名は元の固有部分を含まず、文書本文にある語や公知の実体名とも衝突しません。vault 戦略なら偽名から元の値へ完全一致で復元できます。
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
        HandleDetector(),             # @handle・行頭の発言者名・user= の値 → USERNAME（NER の取りこぼしを補う）
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
- **文字単位ヘッド付きの自前モデル**: `CharHeadNerDetector(model=<フォルダ または Hugging Face Hub のリポジトリ ID>)`。日本語のサブワード境界に左右されない文字単位のスパンを出します。Hub のリポジトリ ID（`owner/name`）を渡すと推論に必要なファイルだけを取得してキャッシュします（`revision=` で固定、`local_files_only=True` でオフライン）。当方が `cl-nagoya/ruri-v3-30m` を基に学習したモデルは Hugging Face Hub の [`az225/privacy-masker-ja-pii-ner-30m`](https://huggingface.co/az225/privacy-masker-ja-pii-ner-30m)（Apache-2.0）で配布しています: `CharHeadNerDetector(model="az225/privacy-masker-ja-pii-ner-30m")`。構成・学習データ・評価はモデルカードに、評価のスクリプトは `tools/` にあります。

どのモデルでも、導入前に自分の文書で評価することを勧めます（後述「評価」）。

## 仕組み

```
テキスト → Detector（NER・正規表現・ハンドル名）→ Refiner（境界補正）→ Resolver（重複解決）
        → Annotator（識別性の判定、任意）→ Policy（型ごとの Operator）→ 変換後テキスト
```

- **Refiner**: 既定の `SpanRefiner` は文字種の規則で端の空白・敬称・助詞を削り、隣接する種別語尾・法人格・住所末尾を取り込みます（PII を落とす方向の変更はしません）。`MorphologicalRefiner` は品詞で判定し、スパンを形態素境界に揃えます。
- **Resolver**: 重なった検出結果は型の優先順位（MY_NUMBER > CREDIT_CARD > EMAIL > PHONE_NUMBER > PERSON > ADDRESS > …）で決め、同じ型ならスコア、同点なら長い方を採ります。URL・EMAIL の内側に完全に収まる検出（`?card=4111…` のカード番号、メールのローカル部の 12 桁など）は優先順位に関わらず外側に吸収し、URL やメールアドレスを分断しません（`EntityResolver(container_types=())` で無効化）。
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

**カテゴリの決め方**: 種別語の規則で決めます。組織は 医療法人・学校法人・教育機関（大学・大学院・学校など）・官公庁（省庁・局・議会・公的な委員会など）・メディア（新聞社・放送・球団名）・政党・団体（協会・組合・連合・同盟・協議会・労組・労連・〜の会）・部署（〜部・〜課）・運輸・製造・IT・金融・小売・建設、それ以外は業種不明の組織。施設は 医療・教育・交通・商業・公共・建物・文化。地名は 国（国名一覧と略称）・都道府県・地区（〜地区・選挙区）・外国（カタカナの地名、漢字の外国都市）・市区町村・町名・自然地名。規則で決まらないものは型ごとの既定の生成器に落ちます。

教育機関・官公庁・団体では、末尾の種別語が閉じた語彙に一致すると、その語（大学・省・市役所・協会など）を保存し、固有部分を架空の造語に置き換えます。
種別語を特定できない場合は教育なら大学、官公庁なら省・庁・市役所、団体なら協会を使い、安全な候補が得られなければトークンへ退避します。

金融・施設・地区・住所の建物部分も、次の閉じた語彙と定型形式を保存し、固有部分だけを架空の名前に置き換えます。

| 対象 | 保存する種別語・形式 |
|---|---|
| 金融 | 銀行・信用金庫・信金・信用組合・証券／證券・生命保険・損害保険・火災保険・保険・リース・ファイナンス・カード・キャピタル。方角プレフィックスと法人格の扱いを維持 |
| 施設 | 小学校・高校・高等学校・幼稚園・駅・空港・港・総合病院・図書館・ビルなど、全カテゴリの閉じた種別語。学校の設置者（県立・市立・私立・都立・府立・道立・国立・町立・村立・区立）は元にあるときだけ保存 |
| 施設の末尾 | 前、北口・南口・東口・西口・中央口・正面口、本館・別館・新館・分館、`N階`・`NF`・`地下N階`・`N号館`・`N号棟` のいずれか一つ |
| 地区 | 駅前・商店街・地区・繁華街・界隈。駅前の核には町・丁目を含めない |
| 住所の建物 | ビル・マンション・ハイツ・コーポ・メゾン・レジデンス・タワー・ハウス等の前置／後置と、`N階`・`NF`・`N号室`・末尾なし。番地と建物、建物と末尾の空白の有無も保存 |

数字は元と異なる1〜12に置き換えます。住所の都道府県・市区町村は、以下の地名の対応を共有します。
語彙・形式に当てはまらない場合は従来のカテゴリ内の生成へ退避します。保存可能な語がある場合は同じ形式で再抽選し、安全な候補が得られなければトークンへ退避します。

**地名の連動（`semantic=True`）**: 日本郵便由来の市区町村1,892件を遅延読み込み・キャッシュします。欠落・破損時は `ConfigurationError` になります。
元の県を同じ地方ブロックの別県へ写し（北海道は東北）、市区町村をその県内から選びます。hmac の県対応は鍵で並べたブロック内の県を巡回させるため、不動点がなく、異なる県の対応先が重なりません（本文がない場合は北海道と東北の1県の重複を許容）。LOCATION・ADDRESS・`LOCATION_PUBLIC` 等で同じ対応を使い、`宮城県仙台市` のような複合地名では県と市を連結します。
本文に複数の県名がある場合は、本文の県と割当済みの県を避けて鍵付き巡回順に割り当て、同じ地方の県が多く地方内に空きが足りないときは別の地方の県を使います（北海道と東北も重複を避けます）。本文がない場合・県名が1つだけの場合のhmac出力は従来どおりで、複数県の割当でも安全な未使用候補が尽きた場合はトークンへ退避します。

| 元の種別 | 置換先の種別（不足・除外時の代替） |
|---|---|
| 市／町 | 市／町 |
| 村 | 村（町） |
| 政令市 | 政令市（市） |
| 政令市の区／特別区 | 政令市の区（市） |

hmac の市区町村は元の県の同種別の一覧と写し先候補をそれぞれ鍵順に並べ、順位に対応する候補を割り当てます。安全条件で使えない候補と先行順位の割当先は飛ばし、代替種別は候補列の後ろへ足します。政令市が3つある県から政令市が1つだけの県へ写す場合、残りは通常市にします。政令市の割当を通常市より先に予約し、種別間の重複も避けます。候補不足では他種別との重複、同種別内の再利用の順に許すため、完全な単射を保証できません。本文に出る順序には依存しません。

政令市の区の正式名は市＋区、略記の「青葉区」は区だけにします。特別区は置換先の正式な市＋区（区が無ければ市）にします。
郡付きの町村は郡付き、郡を省いた表記は郡なしにします。県なし住所に県を補う既存挙動と `keep="city"` を維持します。
同名候補は、まず本文に市区町村名そのもの（郡を省いた表記も含む）がちょうど一候補についてだけ現れる場合に解決し、次に県名・親の政令市名を使います。駅の幹に限り、それでも決まらず、市・政令市・特別区がちょうど1つで残りが町村だけなら、その市・区を採用します。例えば「横浜市の横浜駅」は横浜市に連動し、県などの手がかりがない「横浜駅」も横浜市を選びます。
決まらない場合や一覧外の名前は通常の生成へ退避します。安全フィルタによる候補除外もあるため、hmac の地名の決定性は同じ鍵・scope・値・本文で保証します。県の固定対応が本文中の県名や禁止語に当たる場合、対応を変更せずトークンへ退避します。例えば関東7都県のうち4都県が本文にあると、本文との衝突を避けながら4つの異なる県名へ写すことはできません。

random / vault は同じブロックの県・同じ元県の市区町村について、ストアの内部対応と通常のLOCATION対応を参照し、未使用の候補から選びます。県の対応が未保存でも、市の保存済み対応から先の県が一意に分かれば共有します。県付きの保存値も再利用します。保存済み対応を優先するため、既存の重複は修正せず、候補が尽きればトークンへ退避します。新規割当は呼出順・ストアに依存し、独立したストア間や、北海道と東北が同じ県へ写る場合の市同士の単射性は保証しません。

FACILITY の市区町村名＋役所／役場（市役所・区役所・町役場・村役場を含む）は、置換後の市区町村の種別で作ります。区役所は区名だけを使います。
FACILITY・LOCATION のどちらで検出されても、名前＋駅（既存の定型付随表現を含む）は駅として処理します。市区町村名または末尾の市・区・町・村を除いた幹なら地名に連動し、それ以外は架空の駅名にします。「駅」と北口・西口・前などを保存し、数値付きの定型表現は施設と同じ規則で数字を変えます。同じ値の駅は型をまたいで生成・保存を共有します。ただしhmacで候補が尽きた退避トークンは元の検出型を使います。
置換先が政令市の区なら駅には親の市の幹を使います。例えば市の対応が仙台市→川崎市なら、仙台市役所→川崎市役所、仙台駅→川崎駅、仙台駅前→川崎駅前です。
幹が一文字になる駅名や、同名で候補を決められない駅名は対象外です。新大久保駅など市区町村名に一致しない駅、大学・病院・郵便局・学校などは従来の架空名生成を続けます。
この連動経路の役所・駅・駅前だけは、地名との整合を保つため実在名を許容し、公知データ・同梱禁止リストによる除外を行いません。元の値・固有部分・本文との衝突検査と利用者の `denylist=` は引き続き適用します。
random / vault では内部対応をストアで共有します。内部対応に復元値を付けず、実際に置換した表記だけを復元対象にするため、変換・復元件数に内部対応は加算されません。
異なる元表記が同じ偽名になり復元先が衝突する場合は、後の表記をトークンに退避します。

業種が決まらない組織（既定カテゴリ `service`）は、自然・天候・時間・色の前半24種×後半12種を組み合わせた架空のコアを使います。
Wikipedia見出し語との衝突と一般語等を除外し、漢字258通り・カタカナ253通りを用意しています。教育・官公庁・団体にもこのコアを使います。
service で法人格がない場合はグループ／ホールディングス等8種の企業風語尾を必ず付けます（漢字2,064通り・カタカナ2,024通り）。
法人格がある場合は語尾なしも含む9種から選びます（法人格・位置を固定して漢字2,322通り・カタカナ2,277通り）。法人格の種類と位置、英字3文字の略称は従来どおりです。
同じ鍵・入力・scope でも、業種不明・官公庁・団体や新規則で分類が変わる組織の hmac 出力は0.13.0と変わります。vault に保存済みの対応はそのまま再利用します。
0.15.0 から、同じ鍵・入力・scope でも金融機関・施設・地区・建物付き住所に加え、地名・住所と市区町村名を冠した役所・駅の偽名は0.14.0と変わります。
それ以外のカテゴリと `semantic=False` の出力は維持します（`證券`・`信用組合` は金融と判定）。vault に保存済みの対応は変わりません。

**安全性**: 偽名が元の値に依存するのは、カテゴリと表層特徴（文字種・法人格・閉じた種別語・住所の粒度・地方ブロックなど）、市区町村一覧による所属県と表記の解決、および鍵付き HMAC のシードを通してです。候補は次の条件をすべて満たすものだけ採用します。

- 元の値と同じでない。元の姓・名、組織・施設のコア語、住所の市区町村・町名・解析できた建物の固有部分、地名の語幹を含まない。元の値に含まれる国名や、略称に対する同じ国の標準名（米国 → アメリカ）も使わない。
- 変換中の文書本文に既に出ている語でない（検出漏れの実名との同一化を防ぐ）。
- 架空名は同梱の禁止リスト（生成候補と Wikipedia 見出し語の衝突分）と実行時の公知データ `knowledge=` に無い。実在地名と上記の連動する役所・駅・駅前は公知照合の対象外です。利用者の `denylist=` はすべてに適用します。

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

knowledge = PublicKnowledge.load()                          # 同梱の公知データ（Wikipedia 日本語版の見出し語）
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

公知データはパッケージに同梱しています（約 22MB）。`PublicKnowledge.load()` は明示の `path` → 環境変数 `PRIVACY_MASKER_KNOWLEDGE` → 同梱データの順に読み込みます。
環境変数によらず同梱データを使う場合は `PublicKnowledge.packaged()` を指定します。同梱データはプロセス内で一度読み込み、同じインスタンスを返します。
自分の語彙で作る場合は `tools/build_knowledge.py --src PATH --out PATH` や `PublicKnowledge.from_iterable()` が使えます。
設定ファイルでは `knowledge: bundled` で同梱データを選べます。注釈器の `knowledge=None` や設定での `knowledge` 省略時は、従来どおり公知データを使いません。

- 公人と判定するのは、公職の肩書き（総理・大臣・知事・議員など）が隣接するか、フルネームが公知データに一致するときだけです。企業の肩書きや姓だけでは公人にしません。
- 公知の地名・施設・組織は、Wikipedia 見出し語に一致し、かつ私人に帰属していないときだけ残します（「田中さんの最寄りは◯◯駅」の駅は匿名化）。
- **保持は正の証拠があるときだけ**です。Policy に `*_PUBLIC` の規則が無ければ従来どおりマスクされます。
- 既知の限界: 著名人と同姓同名の私人は `standard` で保持され得ます。公人との共起自体が手掛かりになる文は、私人名は匿名化されても文脈からの絞り込みは残ります。許容できない用途では `strict` を使ってください。

`standard` では、同姓同名の私人を残さないための任意の**文脈による判定**を追加できます。
公知データに一致した人名でも、役割ヘッドの public 確率が閾値未満なら公知判定を取り消します。
ヘッドだけで公知にすることはなく、公職の肩書きによる判定には適用しません。ベクトルが無い・判定不能の場合も取り消し、
その人名の近くの施設・地名などの帰属判定には私人として扱います。`strict` / `minimal` の動作は変わりません。
この拒否判定は**既定では無効**で、アノテータに `person_role` を明示したときだけ有効です。

配布モデルのヘッド（既定の閾値 0.8）を、検出から判定まで通して測った結果です。数字は公人として残した人名の件数です。

| 文書 | 人名 | 規則だけ | ヘッドあり |
|---|---|---|---|
| 私人の文書の人名を、見出し語と一致する一般的な姓名に差し替え | 152 | 152 | 1 |
| 私人の文書の人名を、著名人の名前に差し替え | 152 | 152 | 3 |
| 報道の文（公知データに一致する公人 40 件を含む） | 62 | 40 | 37 |

学習に使っていないひな形と名前の合成文では、閾値 0.8 のとき、私人としての言及を公人と判定する割合が 0.75%、公人としての言及を残す割合が 85.7% でした。
上の文書は開発中の確認にも使ったもので、件数も小さいため、目安として見てください。短い文や文脈が乏しい言及では公人を隠すことがあります。

```python
ner = CharHeadNerDetector(model="model-dir", person_role_head=True)
annotator = IdentifiabilityAnnotator(
    level="standard", knowledge=knowledge,
    person_role=ner.person_role_head,
    # person_role_threshold=0.8,  # 省略時はヘッドの threshold を使用
)
```

閾値は `person_role_threshold` → `person_role.threshold` の順で選び、[0, 1] の数値を指定します。
確率が閾値ちょうどなら保持候補のままです。ヘッドを指定しても閾値が得られない場合は `ConfigurationError` になります。
`span_vectors=True`（既定）が必要です。分類ヘッドと併用する場合も、検出器が必要な文脈ベクトルを選びます。
モデルフォルダまたは Hub のモデルに `person_role_head.json` と `person_role_head.safetensors` が必要で、
ヘッドが無い古い版のモデルでは利用できません。検出器の `person_role_head="auto"`（既定）は JSON があれば読み込み、
`True` は必須、`False` は読み込みません。`True` でファイルが無ければ `FileNotFoundError` です。
自動ロードされても、アノテータに渡さなければ拒否判定は有効になりません。
ヘッドを別フォルダに置く場合は `PersonRoleHead.load(model_dir, head_dir=None)` で直接読み込めます。

設定ファイルでは、`char_head` 検出器に `person_role_head: true`、`identifiability` アノテータに
`person_role: detector` を指定します。`person_role_threshold` で閾値を上書きできます。
参照できるヘッドが無い、または複数あって一意に選べない場合は `ConfigurationError` です。
評価ツールは `tools/eval_identifiability.py standard model-dir --person-role-head` で規則だけとの保持件数を比較できます。
`--person-role-threshold` で閾値を上書きでき、省略時や `--no-person-role-head` では規則だけを使います。本文・表層は出力しません。

## 検出できる型

- **正規表現** `RegexDetector.default_ja()`: EMAIL、PHONE_NUMBER（全角・国際番号・空白区切り）、POSTAL_CODE、CREDIT_CARD（主要ブランドの先頭桁 + Luhn）、IP_ADDRESS（IPv4 / IPv6）、MAC_ADDRESS、URL、MY_NUMBER（検査用数字）、CORPORATE_NUMBER（法人番号）、PASSPORT、RESIDENCE_CARD、文脈語が必須の DRIVERS_LICENSE・PENSION_NUMBER・BANK_ACCOUNT。`default_ja(dates=True)` で DATE と AGE も検出します。全角の入力は長さを変えない写像で半角化して照合するので、`０４８－２２２－７８１５` のような表記もそのまま拾えます。
- **番号類のスコア**: CREDIT_CARD・MY_NUMBER・CORPORATE_NUMBER は、検査数字が合うだけなら 0.5、前後 30 字に文脈語（カード・クレジット / マイナンバー・個人番号 / 法人番号・登録番号・`T` …）があれば 1.0 です。既定（閾値 0）ではどちらもマスクされ、`score_threshold={"MY_NUMBER": 0.6, "CREDIT_CARD": 0.6}` のように閾値を上げると、伝票番号などが偶然検査数字に合っただけの数字列を外せます。英字・`_`・`/`・小数点が直結した数字列（`ID_4111…`、URL のパス、`0.4111…`）は番号として取りません。
- **NER**: PERSON / ADDRESS / ORG / FACILITY / LOCATION。
- **ハンドル名** `HandleDetector`: USERNAME。`@yuki_t`、行頭の発言者名（`yuki_t: …`、`[09:12] yuki_t: …`）、`user=yuki_t` のようなキーの値など、ハンドル名と分かる書き方だけを規則で拾います。文中にそのまま書かれたハンドル名（「担当 yuki_t」）は拾わないので、NER と併用してください。NER も同じ箇所を拾った場合は PERSON として扱われます。
- 自前の型は `RegexPattern("PROJECT", re.compile(r"PJ-\d{4}"), score=0.6, context=("案件",))` のように追加できます。文脈語は NFKC・小文字化して前後 30 字（`context_window`）と比べ、英数字だけの語（`T`・`tel`）は前後に英数字が続かない単語としてだけ一致します（日本語の語は部分一致）。

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
`ContextEnhancer()` を加えると、スパンの前後に型ごとの手掛かり語（氏名・住所・電話 …）があるとき score を上げられ、`score_threshold` と組み合わせて「文脈のない低スコア検出だけを落とす」運用ができます。手掛かり語の照合規則は `RegexPattern` の文脈語と同じです。

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
- `CharHeadNerDetector(device="cuda", dtype="float16")` で forward を半精度にできます（重みとヘッドの softmax は fp32 のまま）。`dtype` の既定は `None`（fp32）です。MPS / CPU では半精度にしても速くなりません。

Colab の T4 で測るノートブック: [examples/colab_t4_benchmark.ipynb](examples/colab_t4_benchmark.ipynb)

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

同梱の公知データ `privacy_masker/data/public_knowledge_ja.pmk` は、Wikipedia 日本語版の見出し語（ns0、2026-09-02 取得のダンプ。[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）を 64bit ハッシュにしたもので、見出し語の文字列は含みません。由来・件数・ハッシュは同じフォルダの `public_knowledge_ja.manifest.json` に記録しています。

同梱の `privacy_masker/data/surrogate_denylist.pmk`（偽名が公知の実体名と衝突しないための禁止リスト）は、Wikipedia 日本語版の見出し語（ns0、2026-09-02 取得のダンプ。[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)）と生成候補の衝突分を 64 bit ハッシュにしたもので、見出し語の本文は含みません。

同梱の `privacy_masker/data/ja_municipalities.tsv` は、[日本郵便の郵便番号データ（UTF-8）](https://www.post.japanpost.jp/service/search/zipcode/download/utf/zip/utf_ken_all.zip)（2026-09-22 取得）から都道府県・市区町村の組を重複除去して抽出した1,892件です。
日本郵便は郵便番号データについて著作権を主張していません。取得日、出典URL、元のzipとTSVのsha256は `ja_municipalities.manifest.json` に記録しています。
再生成には `tools/build_municipalities.py --src input.csv --acquired-date YYYY-MM-DD --source-sha256 SHA256` を使います。既定は公式CSVの7・8列目、ヘッダーなしです。抽出済みCSVには `--pref-column 1 --city-column 2 --header` を指定できます。
