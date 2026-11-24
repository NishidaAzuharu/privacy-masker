---
language:
  - ja
license: $license
library_name: privacy-masker
pipeline_tag: token-classification
inference: false
base_model: cl-nagoya/ruri-v3-30m
base_model_relation: finetune
tags:
  - ner
  - pii
  - anonymization
  - de-identification
  - japanese
  - modernbert
  - character-level
  - privacy-masker
datasets:
  - stockmark/ner-wikipedia-dataset
  - nvidia/Nemotron-Personas-Japan
---

# privacy-masker 用 日本語 PII 検出モデル（文字単位 NER）

[privacy-masker](https://github.com/NishidaAzuharu/privacy-masker)（日本語テキストの個人情報を検出・仮名化・匿名化する Python ライブラリ、MIT）が
NER 検出器として使う学習済みモデルです。日本語テキストから、個人情報につながる固有表現 — **人名 PERSON・組織 ORG・地名 LOCATION・施設 FACILITY・住所 ADDRESS** — を
**文字単位**のスパンで検出します。ライブラリの使い方、評価の手順、精度の目安は privacy-masker の README を参照してください。

*The NER model used by [privacy-masker](https://github.com/NishidaAzuharu/privacy-masker), a Japanese PII de-identification library.
It returns character-level spans for PERSON / ORG / LOCATION / FACILITY / ADDRESS. Load it with privacy-masker's `CharHeadNerDetector`;
it is not a `transformers` token-classification model, so the Hub inference widget does not apply. Documentation is in Japanese.*

## 使い方

```bash
pip install 'privacy-masker[transformers]'
```

```python
from privacy_masker import PrivacyEngine, Policy
from privacy_masker.detectors import CharHeadNerDetector, HandleDetector, RegexDetector

ner = CharHeadNerDetector(model="$repo_id")   # 初回に Hub から取得してキャッシュ。revision="<commit>" で固定できます
engine = PrivacyEngine(
    detectors=[ner, RegexDetector.default_ja(), HandleDetector()],   # NER + 電話・メール等の正規表現 + ハンドル名
    policy=Policy.preset("llm_input"),                                # すべて <TYPE_001> 形式のトークンに
)
print(engine.transform("山田太郎さんは株式会社ABCに勤務しており、東京都新宿区に住んでいる。").text)
# <PERSON_001>さんは<ORG_001>に勤務しており、<LOCATION_001>に住んでいる。
```

`transformers` の `pipeline("token-classification")` や `AutoModelForTokenClassification` では使えません（`AutoModel` で読めるのは encoder だけです）。
検出結果だけが要るときは `engine.detect(text)`、偽名への置換や公知の実体を残すレベルの使い方はライブラリの README にあります。

## 仕様

| 項目 | 内容 |
|---|---|
| 用途 | privacy-masker の NER 検出器（`CharHeadNerDetector`）。Refiner（敬称・助詞の削除、種別語尾の取り込み）と Resolver（重複解決）はライブラリ側 |
| 対象 | 日本語。ラベルは PERSON・ORG・LOCATION・FACILITY・ADDRESS（文字単位の BIO、11 ラベル） |
| encoder | [`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m)（ModernBERT、10 層・hidden 256、Apache-2.0。commit `$base_revision`）を全層 fine-tune |
| 文字ヘッド | サブワードの隠れ状態を文字に展開し、文字埋め込み・文字種・トークン内位置の特徴を足して、文字ごとに BIO ラベルを出す小さなヘッド（追加パラメータ 87,947） |
| 重み | 1 回の学習の複数エポックの単純平均 |
| 入力 | 400 字のチャンク（重なり 50 字、最大 512 トークン）。被覆されない文字や非有限の出力があれば例外で止まる（空結果を返さない） |
| 規約 | ADDRESS は番地あり、LOCATION は番地なし。敬称・肩書きはスパン外。電話・メール・郵便番号・日付は対象外（privacy-masker の正規表現が担当） |
| ファイル | `model.safetensors`（encoder）、`char_head.safetensors` + `char_head_config.json`（文字ヘッド）、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`privacy_masker_model.json`（由来と SHA-256） |

<!-- person_role_head:start -->
## 人名の役割ヘッド（任意）

`person_role_head.json` / `person_role_head.safetensors` は、文中の人名が「公人としての言及」か
「私人としての言及」かを、前後の文脈から判定する小さなヘッドです。
privacy-masker の `standard` レベルで、公知データに一致した人名を私人としての言及と判定したら
残さないための任意機能です。**拒否だけに使い、既定では使いません**。ヘッドだけで公人として保持することはなく、
公職の肩書きによる判定には適用しません。

```python
from privacy_masker import IdentifiabilityAnnotator, MorphologicalRefiner, PrivacyEngine, Policy, PublicKnowledge
from privacy_masker.detectors import CharHeadNerDetector, RegexDetector

ner = CharHeadNerDetector(model="$repo_id")
annotator = IdentifiabilityAnnotator(
    knowledge=PublicKnowledge.load(), person_role=ner.person_role_head,
    # person_role_threshold=0.8,  # 閾値を上書きする場合（0〜1）
)
engine = PrivacyEngine(
    detectors=[ner, RegexDetector.default_ja()], refiner=MorphologicalRefiner(),
    annotators=[annotator], policy=Policy.preset("standard"),
)
```

公人としての言及の確率が閾値 **$person_role_threshold** 未満なら、公知データによる保持を取り消します。
`IdentifiabilityAnnotator` の `person_role_threshold` で上書きでき、省略時はヘッドの `threshold` を使います。
確率が閾値ちょうどなら保持候補のままです。ベクトルが無い場合や判定不能の場合も保持を取り消します。
検出器は同梱されたヘッドを自動で読み込みますが、上の例のようにアノテータに渡したときだけ拒否判定が有効になります。
`span_vectors=True`（既定）が必要です。

役割ヘッドは、次のデータでエンコーダを凍結し、ヘッドだけを学習しました。

- 役割のラベルを付けた文章のひな形を大規模言語モデル（OpenAI の gpt-6.1-terra）で生成し、役割と無関係に選んだ名前を差し込んだ合成文。
- NER の学習データの人名（報道・百科事典に由来する文の人名を公人、合成した私人文書の人名を私人とする弱いラベル）。

生成したひな形・学習データは配布しません。以下の「学習データ」は NER 本体についての説明です。
役割ヘッドの重みと設定にも「ライセンスと帰属」の重み・設定ファイルのライセンスが適用されます。

閾値 0.8 で、検出から判定まで通して測った結果です。数字は公人として残した人名の件数です。

| 文書 | 人名 | 規則だけ | ヘッドあり |
|---|---|---|---|
| 私人の文書の人名を、見出し語と一致する一般的な姓名に差し替え | 152 | 152 | 1 |
| 私人の文書の人名を、著名人の名前に差し替え | 152 | 152 | 3 |
| 報道の文（公知データに一致する公人 40 件を含む） | 62 | 40 | 37 |

学習に使っていないひな形と名前の合成文では、閾値 0.8 のとき、私人としての言及を公人と判定する割合が 0.75%、公人としての言及を残す割合が 85.7% でした。
上の文書は開発中の確認にも使ったもので、件数も小さいため、目安として見てください。

文脈からの推定であり、判定は誤ることがあります。短い文・文脈が乏しい言及は私人側に倒れやすく、公人を隠すことがあります。

<!-- person_role_head:end -->
## 学習データ

Wikipedia 由来の固有表現データ（[`stockmark/ner-wikipedia-dataset`](https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset)、CC BY-SA 3.0）、
[UD Japanese GSD](https://github.com/megagonlabs/UD_Japanese-GSD) の学習・開発分割（CC BY-SA 4.0）、
LLM（[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)、Apache-2.0）に書かせたテンプレートへ
[`nvidia/Nemotron-Personas-Japan`](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan)（CC BY 4.0）の人名と日本郵便の郵便番号データから合成した値を差し込んだ文書、
および PII を含まない負例で学習しました。評価に使う文書の本文・表層は学習データから除外しています。学習データそのものは配布しません。

## 制限

- 検出漏れ・過剰検出は起こり得ます。人手レビューの併用を前提にし、導入前に自分の文書で評価してください（評価の手順と精度の目安は privacy-masker の README）。
- 個人情報保護法などの基準への適合を保証するものではありません。個人を特定・追跡する目的での利用は意図していません。

## ライセンスと帰属

- このリポジトリの重みと設定ファイル: **$license**
- encoder の元モデル `cl-nagoya/ruri-v3-30m`: Apache-2.0（その元の `sbintuitions/modernbert-ja-30m`: MIT）
- 学習データ: 上記の各ライセンス。文書・名前そのものは再配布していません。

## ファイルの SHA-256

| ファイル | バイト数 | SHA-256 |
|---|---:|---|
$sha_table
