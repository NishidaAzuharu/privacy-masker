# 意味を保つ Surrogate（カテゴリ保持の仮名化と往復）— 設計（2026-09-24）

## 0. 結論

`Surrogate` operator を拡張し、**元の実体と同じ粗いカテゴリ（病院／小学校／製造業／東北地方の市 …）の架空の名前**に置き換える。
カテゴリは **NER の encoder が既に計算している隠れ状態から作るスパンベクトル + 線形の分類ヘッド**で決める（追加の forward なし、追加の配布物は数十 KB）。
規則（種別語）で決まるものは規則で決める。置換先が元の値に依存するのは**カテゴリと表層特徴を通してだけ**にし、置換先から元を逆算できない構造にする。
あわせて **`strategy="vault"` で可逆な Surrogate**（外部 LLM への往復）を足す。

方針（2026-09-24 決定）: 意味の保持は A（文の意味が壊れない）を土台に C（文脈との整合）を載せる。B（統計の保存）は対象外。
逆算不能は must。実在の公知の実体を指さない・元の実体を狭めない・一貫性 も must。往復を v1 に含める。
hmac 戦略でも分類ヘッドを使い、文書横断のカテゴリ一貫性は best effort とする（案 (a)）。

## 1. 用途と脅威モデル

| 用途 | 求めるもの | 戦略 |
|---|---|---|
| LLM 学習データの匿名化（元の分布と文意を保つ） | 一方向、文書横断で同じ実体は同じ偽名、カテゴリ保持 | `hmac` + `semantic=True` |
| 外部 LLM への入力の仮名化と応答の復元（RAG・要約・チャット） | 往復、自然な偽名（`<PERSON_001>` は LLM の出力品質を落とす）、復元の確実さ | `vault` + `semantic=True` |
| テスト・開発用データ | 書式と参照整合性 | `hmac`（`FormatPreserving` と併用） |
| 社内共有・外部委託・公開コーパス | 読みやすさ、実在の人と誤認されない、最も強い一方向性 | `random` |

**攻撃者**: 置換後のテキストと、このライブラリのコード・プール・カテゴリ体系・分類ヘッドをすべて持つ。鍵（hmac）と対応表（random / vault）は持たない。
往復用途では外部 LLM 事業者がこの攻撃者に当たる。

**守るもの**

| | 内容 | 位置づけ |
|---|---|---|
| A 逆算不能 | 置換後の名前から元の名前を、候補集合の総当たりでも当てられない | **must** |
| B 公知の実体を指さない | 置換先が Wikipedia 見出し語にある実体（人物・組織・施設・作品）と一致しない | **must**（公知について。無名の小規模事業者との偶然の一致は防げない） |
| C 元の実体を狭めない | 意味の保持はカテゴリ（種別・地方ブロック・粒度）まで。同じ市・同じ規模・同じ業界の細分類まで合わせない。外部知識で元の実体の属性を引かない | **must**（設計原則） |
| D 一貫性 | scope 内で同じ実体は同じ偽名。hmac は文書横断で決定的（偽名について。カテゴリは best effort） | **must**（現状維持） |

A を満たす構造: 偽名 = Generator(カテゴリ, 表層特徴, rng)。rng は鍵付き HMAC（hmac）か乱数（random / vault）。カテゴリと表層特徴はいずれも
母集団が数千以上の粗い属性なので、置換後の名前と整合する元の候補は常に多数ある。「元の値の埋め込みと最も近い候補を選ぶ」方式は、
同じモデルと候補集合を持つ攻撃者が写像を再現できるため**採用しない**。分類ヘッドは候補名との比較を一切行わない。

## 2. 全体の構造

```
Engine.transform(text)
  Detector（CharHeadNerDetector）→ Entity に vector（スパンベクトル）を添える
  → Refiner → Resolver → Annotator → Policy
  → Surrogate.apply(value, entity, ctx)        ctx.text に文書全文
      ① 正規化キーで scope の対応表を引く。あれば再利用                          … D
      ② CategoryResolver: 規則 → 分類ヘッド（entity.vector があれば）→ 型レベル     … C
      ③ 表層特徴の抽出（§5）
      ④ Generator[カテゴリ](features, rng) → 候補                               … A（元の値を受け取らない）
      ⑤ SafetyFilter（§7）。不合格なら再抽選（上限 10）。全部落ちたら型レベルの生成器 → それも落ちたら <TYPE_hmac8> トークン。元の値を返す経路は無い … B
      ⑥ 対応表に保存。vault なら元の値を暗号化して同じ器に保存                     … 往復
Engine.restore(text, scope_id)
  既存の <PERSON_…> パターン + scope に保存された偽名（と姓リンクの偽姓）を長い順の 1 本の正規表現で 1 回だけ置換
```

新しい部品と変更点:

| 部品 | 場所 | 変更 |
|---|---|---|
| `Entity.vector: bytes \| None = None` | `privacy_masker/types.py` | float16・リトルエンディアンのスパンベクトル（30m: 256 次元 = 512 バイト、70m: 384 次元）。既定 None。frozen dataclass に既定値付きで追加（互換） |
| `CharHeadNerDetector(span_vectors=True)` | `detectors/char_head.py`、`char_head.py` | 予測時に `last_hidden_state` を、エンティティのスパンに重なるサブワードで平均し L2 正規化。チャンク境界をまたぐスパンは最初のチャンクの被覆トークンで計算。追加の forward なし |
| `TransformContext.text: str \| None` | `operators/base.py`、`scopes.py`（DocumentContext / StoreContext）、`engine.py` | transform ごとに文書全文を持たせる。§7 の「文書内既存表層」に使う |
| `PseudonymStore.iter_tokens(scope_id)` | `stores/base.py`、`memory.py`、`sqlite.py` | `Iterable[tuple[entity_type, token]]`。往復の復元に使う。**破壊的変更**（自作 Store は要実装） |
| `CategoryResolver`（`RuleCategorizer` + `HeadCategorizer`） | `operators/semantic.py`（新規） | §4・§8 |
| `Features` と `extract_features` | `operators/semantic.py` | §5 |
| カテゴリ別 Generator | `operators/surrogate_generators.py`（新規）、`data/ja_pools.py`（拡張） | §6 |
| `SafetyFilter` と同梱の禁止リスト | `operators/surrogate_filter.py`（新規）、`data/surrogate_denylist.pmk`（新規、`tools/build_surrogate_denylist.py` で生成） | §7 |
| `category_head.safetensors` + `category_head.json` | NER モデルのフォルダ | §8。`load_char_model` が見つければ読む。無ければ規則のみ（警告 1 回） |
| `Surrogate(kind="facility"\|"location"\|"auto", semantic=, knowledge=, denylist=, min_margin=, strategy="vault")` | `operators/surrogate.py` | §10 |

## 3. 実行時間とメモリ

- 規則: µs。分類ヘッド: 1 実体あたり 256（384）次元の行列積 1 回 = µs。フィルタ: ハッシュ照合と `str in text`。NER の 12 ms/文書に対して測定誤差の範囲
- スパンベクトル: encoder の隠れ状態はすでに計算しており、チャンク単位でプーリングするだけ。Entity 1 件あたり 512 バイト
- 別の埋め込みモデルは使わない（+5〜10 ms/文書・150MB の配布物を避ける）。`[semantic]` extra は作らない

## 4. カテゴリ体系

粗さの下限は「日本に数千件以上ある」。分類ヘッドの信頼度が閾値未満なら各型の「その他」に落とす。

| 型 | カテゴリ（ヘッドの出力）| 規則で決まる典型 |
|---|---|---|
| ORG（14） | 企業/製造、企業/IT・通信、企業/金融・保険、企業/小売・飲食、企業/建設・不動産、企業/運輸・物流、企業/サービス・その他（既定）、医療法人・福祉法人、学校法人、官公庁・自治体・公的機関、団体（協会・組合・財団・NPO）、メディア・出版・芸能・スポーツ球団、政党・政治団体、部署 | `〜電機/製作所/工業/化学`→製造（`電気` は外した）、`〜システム(ズ)/テクノロジー/ソフト/電気通信/テレコム`→IT、`〜銀行/信用金庫/証券/保険`→金融、`〜商店/ストア/百貨店/フーズ`→小売、`〜建設/工務店/不動産`→建設、`〜運輸/物流/交通/商船/航空/鉄道/電鉄/電気鉄道`→運輸、`医療法人〜/社会福祉法人〜`、`学校法人〜`、`〜省/庁/局/議会/機構/自衛隊/陸軍/警察/裁判所/県庁/市役所` と公的な修飾の `〜委員会`（`教育委員会`・`〜県…委員会` など）→官公庁（業種語より先に見る。`放送局/郵便局/支局/薬局/事務局/編集局` は除く）、`〜協会/連盟/組合/財団/〜の会`→団体、`〜新聞/放送/出版/テレビ/通信社/通信` と球団名（`〜ホークス/ジャイアンツ/球団/巨人軍` など）→メディア、`〜党/政治連盟`→政党、`〜部/課/室/チーム/係` と公的でない `〜委員会`（法人格なし）→部署。`商事/ホーム/グループ` は種別語でなくなった（既定の企業/サービス・その他）。v0.10.0 で変更（設計は開発記録、非公開） |
| FACILITY（8） | 医療、教育、交通、商業、公共、建物・住宅、文化・観光、その他 | `〜病院/クリニック/医院/歯科/薬局`、`〜大学/高校/中学校/小学校/幼稚園/保育園/塾`、`〜駅/空港/港/IC/SA`、`〜ホテル/旅館/ストア/モール/食堂`、`〜市役所/図書館/公民館/文化会館/体育館/公園/神社/寺`、`〜ビル/マンション/タワー/ハイツ/団地`、`〜美術館/博物館/温泉/遊園地` |
| LOCATION（8） | 国、都道府県、市区町村、町域・丁目、地区・通称、外国の都市・地域、自然地形、その他 | 国名リスト（`COUNTRIES` + `COUNTRIES_EXTRA` + 漢字略称 `COUNTRY_ALIASES`、`〜共和国/帝国/合衆国/連邦`）→国、`都/道/府/県`、`〜地区/駅前/商店街/繁華街` と選挙区 `〜1区`→地区・通称（市区町村より先）、漢字の外国都市（`上海/北京/香港` など）とカタカナ+`市/町/州/省`→外国の都市・地域（市区町村より先）、`市/区/町/村`（PREF_CITIES との照合）、`〜丁目/〜番`、カタカナ外国地名、`〜山/川/湖/島/岬/湾/海`（本体 2 字以上で市区町村・町名に無いときだけ）。v0.10.0 で変更 |
| ADDRESS（1） | 粒度で決まる（§5） | — |
| PERSON（0） | カテゴリを持たず表層特徴で決める（§5） | — |

地方ブロック（都道府県・市区町村・住所の都道府県に使う。8 区分）: 北海道 / 東北 / 関東 / 中部（新潟・富山・石川・福井・山梨・長野・岐阜・静岡・愛知）/ 近畿 / 中国 / 四国 / 九州・沖縄。
北海道は同ブロックに他の都道府県が無いので、都道府県レベルの置換だけ東北で代替する（市区町村は北海道内の別の市に置き換える）。

カテゴリの決定は決定的（同じ入力 → 同じカテゴリ）。random / vault では scope 内の初出で確定し対応表で再利用する。hmac では対応表を持たないので
出現ごとに決めるが、規則で決まるものは文書横断でも同じ、分類ヘッドで決まるものは文脈込みのため別文書で別カテゴリになりうる（best effort。
偽名の一貫性ではなく、同一人物が 2 つの偽名を持つのと同種の劣化で、安全性には影響しない）。

## 5. 表層特徴（`Features`）

Generator が受け取るのは次の特徴だけ。元の値そのものは渡さない（元の文字が偽名に漏れる経路を型で塞ぐ）。

| 型 | 特徴 |
|---|---|
| 共通 | `category`、`script`（漢字 / カタカナ / ひらがな / ローマ字 / 半角カナ / 英字略称）、`length_class`（短 / 中 / 長） |
| PERSON | `structure`（フル / 姓のみ / 名のみ / あだ名）、`gender_hint`（男 / 女 / 不明。名を `GIVEN_KANJI_M` / `GIVEN_KANJI_F` などのプールに照合。姓には使わない）、`kata_foreign`（カタカナ外国人名か）、区切り（空白・中黒）の有無 |
| ORG | `legal_form`（株式会社 / 合同会社 / 有限会社 / 医療法人 / 学校法人 / 一般社団法人 … / なし）と位置（前 / 後） |
| FACILITY | `place_prefix`（地名で始まるか。始まるならその地方ブロック）、`org_prefix`（`市立/県立/私立/国立`） |
| LOCATION | `granularity`（国 / 都道府県 / 市区町村 / 町域 / 地区 / 外国 / 自然）、`region_block`（都道府県名か `PREF_CITIES` の市から決める。決まらなければ乱択したブロック） |
| ADDRESS | `granularity`（都道府県〜番地〜建物のどこまであるか）、`keep`（`"city"` なら市区町村まで元のまま。既存）、`has_building`、`fullwidth_digits`、`region_block` |

姓リンク（既存）は維持する: フルネームの偽姓は姓部分だけの鍵付き HMAC で決めるので、姓だけの言及と一致する。

## 6. カテゴリ別の生成器

`ja_pools.py` を拡張し、カテゴリごとのテンプレートで組み立てる。地名は LOCATION と ADDRESS の都道府県・市区町村・町域にだけ実在のもの
（同じ地方ブロックの別の地名。町域は KEN_ALL 由来の `TOWNS` を拡充）を使い、ORG・FACILITY・PERSON・番地・建物はすべて架空にする。

| カテゴリ | テンプレート（`{}` はプールからの抽選） |
|---|---|
| 企業/製造・IT・金融・小売・建設・運輸・サービス | `{法人格}{コア}{業種語}`。業種語はカテゴリ別（製造: 電機・製作所・工業・化学・精工、IT: システムズ・テクノロジー・ソフト・ネットワークス、金融: 銀行・信用金庫・証券・保険、小売: 商店・ストア・マート・フーズ、建設: 建設・工務店・不動産・ハウジング、運輸: 運輸・物流・交通・商船、サービス: 既存 `ORG_INDUSTRIES`）。法人格の有無と位置、文字種（カタカナ名は `ORG_KATA_PARTS`、英字略称は 2〜4 字の大文字）は元と同じ |
| 医療法人・福祉法人 / 学校法人 | `医療法人{コア}会` `社会福祉法人{コア}会` / `学校法人{コア}学園` |
| 官公庁・自治体・公的機関 | `{地名（同ブロック）}{市役所\|県庁\|町役場}` `{コア}{機構\|センター\|事業団}` |
| 団体 | `{地名\|コア}{協会\|連盟\|組合\|財団\|研究会}` |
| メディア・芸能・球団 | `{コア}{新聞\|放送\|出版\|テレビ}` `{地名}{カタカナ}（球団風）` |
| 政党 | `{コア}{党\|の会}` |
| 部署 | `{営業\|総務\|人事\|開発\|企画\|経理\|品質保証\|情報システム\|広報}{部\|課\|室\|グループ}`。元と同じ機能語は避ける |
| FACILITY 医療 | `{地名\|姓\|コア}{病院\|総合病院\|クリニック\|医院\|内科\|歯科\|薬局}` |
| FACILITY 教育 | `{市立\|県立\|私立}{地名\|コア}{小学校\|中学校\|高等学校}` `{コア}大学` `{コア}{幼稚園\|保育園}` |
| FACILITY 交通 | `{地名}{駅\|空港\|港}` `{地名}IC` |
| FACILITY 商業 | `{カタカナ}{ホテル\|ストア\|モール\|マート}` `{コア}{食堂\|亭}` |
| FACILITY 公共 | `{地名}{市役所\|図書館\|文化会館\|体育館\|公民館\|公園}` `{コア}{神社\|寺}` |
| FACILITY 建物・住宅 | 既存 `BUILDING_A/B` + `{カタカナ}{ビル\|タワー}` |
| FACILITY 文化・観光 | `{地名\|コア}{美術館\|博物館\|温泉\|水族館}` |
| LOCATION | 粒度を保存。国 → 別の国名、都道府県 → 同ブロックの別の都道府県、市区町村 → 同ブロックの別の市区町村（`PREF_CITIES` を全都道府県・主要市に拡充）、町域 → 別の町域、地区・通称 → `{地名}駅前` 等、外国の都市 → カタカナ外国地名プール、自然地形 → `{コア}{山\|川\|湖\|岬}` |
| ADDRESS | 既存の生成器 + 都道府県を同ブロックに |
| PERSON | 既存の生成器 + 名のプールを性別傾向で選ぶ（不明なら混合）。カタカナ外国人名・ローマ字・ひらがな・半角カナはそれぞれのプール（学習データ生成ツールの値プール `tools/synth_v11_values.py` から移植） |

型レベルの生成器（退避先）= 現状の `generate_person / generate_org / generate_address` と、FACILITY・LOCATION 用の「その他」テンプレート。

## 7. 安全フィルタ

候補は次をすべて通ったものだけ採用する。順に判定し、不合格なら次の候補を抽選（上限 `_DRAW_RETRIES = 10`）。全部落ちたら型レベルの生成器で同じ手順を繰り返し、
それも落ちたら `<TYPE_hmac8>` 形式のトークンに退避する（`Tokenize(hmac)` と同じ表現。**元の値を返す経路は作らない**）。

| フィルタ | 内容 |
|---|---|
| 元との一致 | 正規化後に元と同一。元の姓（PERSON）・コア語（ORG/FACILITY、2 字以上）を含む |
| 公知の実体（B）第 1 層 | **同梱の禁止リスト** `data/surrogate_denylist.pmk`（`PublicKnowledge` と同じ 64bit ハッシュ集合）。`tools/build_surrogate_denylist.py` が、プールの全組み合わせ（姓×名、法人格×コア×業種語、地名×施設語 …）を Wikipedia 見出し語 282 万件と照合し、**衝突した組み合わせだけ**を書き出す。知識ファイルの無い環境でも B が成立する。生成後に `tests/test_surrogate_denylist_file.py` で同梱ファイルの存在と、生成器の候補全体から作られていることを確認 |
| 公知の実体（B）第 2 層 | `knowledge=` に `PublicKnowledge`（Annotator と同じ 282 万件）かそのパスが渡されればそれとも照合 |
| 文書内の既存表層 | 候補が `ctx.text`（元の文書本文）に既に出現していたら不可。検出漏れの実名と偽名が同一化する・別の実体と混ざるのを防ぐ |
| scope 内の衝突 | 別の実体に既に割り当てた偽名と同一なら不可（対応表の token → value が 1:1） |
| 利用者の禁止リスト | `denylist=` の語（正規化して照合） |

## 8. 分類ヘッド

**推論**: `HeadCategorizer(entity_type, vector) -> (category, margin)`。`entity.vector` が無い Entity（正規表現・他の NER・古い呼び出し）は規則だけで決める。型ごとの線形層（ORG 14 / FACILITY 8 / LOCATION 8 クラス）+ softmax。
1 位と 2 位の確率差 `margin < min_margin`（既定 0.15）なら「その他」。ファイル: NER モデルのフォルダに `category_head.safetensors`
（重みとバイアス）と `category_head.json`（型ごとのカテゴリ名、`min_margin` の推奨値、encoder の `model.safetensors` の SHA-256、学習データの SHA、保留精度）。
`load_char_model` が読み、encoder の SHA が一致しなければ読まない（別モデルの隠れ状態には使えない）。

**学習**（`tools/train_category_head.py`、CPU 数分）

- 入力: 本番 NER モデル（凍結）で 学習データ（train / valid）と Wikipedia・GSD の ORG / FACILITY / LOCATION スパンのベクトルを計算
- ラベル: 合成層は差し込んだ値のカテゴリ（学習データ生成ツール `tools/synth_v11_fill.py` と前の世代の生成器の form を §4 の体系に写像）、実テキストは `RuleCategorizer` で付く分だけ（付かないものは学習に使わない）
- **種別語を隠す拡張**: 学習スパンの 30% で、本文から種別語（`病院` `株式会社` `市` …）を削った写しを作り、削った後の文で再びベクトルを計算して同じラベルで学習する。
  ヘッドが種別語のトークンだけに頼らず、文脈（attention 経由）から決められるようにする
- 検証は文書単位で分け、valid の (i) 規則ラベル付きスパンの macro accuracy、(ii) **種別語を削った**スパンの accuracy と、`min_margin` ごとの「その他に落とさなかったもの」の precision を出す
- 受け入れ: (i) ≥ 0.90、(ii) で precision ≥ 0.95 になる `min_margin` を採用値にする。届かないカテゴリは隣と統合する（細かさより落とさないことを優先）
- NER モデルを差し替えたら作り直す。manifest に encoder SHA・データ SHA・精度を残す（開発記録、非公開）

**2026-09-28 結果**: 実装済み・全 kind 不採用（v0.9.0 は規則のみ）。上の記述との差分（読み込みは `load_char_model` ではなく `CharHeadNerDetector`、
`min_margin` の既定は `None` = ヘッドの推奨値）と判定の経緯は `docs/superpowers/specs/2026-08-08-privacy-masker-design.md` §28（数値は開発記録、非公開）。

**2026-09-29 v2**: 実文書の Qwen ラベルで作り直したが全 kind 不採用（設計・結果は開発記録、非公開）。第 2 段（多タスク）の発動条件を満たす。

## 9. 往復（`strategy="vault"`）

前提は `Tokenize(strategy="vault")` と同じ: `scope_id`・`CryptoProvider`・鍵。無ければ `ConfigurationError`（既存の検査に Surrogate を含める）。

- **保存**: token = 偽名そのもの。`store.put_token(scope, etype, value_hash, token=偽名, encrypted_value=暗号化(scope, 偽名, 元の値))`。
  姓リンクの別名（`\x00sei:` キー）も token = 偽姓、値 = 暗号化した元の姓で保存する
- **復元** `Engine.restore(text, scope_id)`: 既存の Tokenize パターンに加え、`store.iter_tokens(scope_id)` で scope の全 token を列挙し、
  長い順に `re.escape` して 1 本の alternation にまとめ、既存と同じく**元の text に対して 1 回だけ** `sub()` する（復号した平文が後続に再マッチしない）。
  各一致は `get_value(token)` → 復号 → `(scope, token, 元の値)` の整合検査（既存）
- **完全一致のみ**。NFKC・読み・言い換え・埋め込み類似では戻さない（LLM が偶然作った別の名前を実名で上書きする事故を避ける）。
  戻せなかった偽名は復元後の文に残る。`RestoreResult` に `restored: int`（戻した件数）を足し、`unresolved` の意味は変えない
- 偽姓だけの言及は姓リンクの別名で戻す。偽姓は **2 字以上**に限る。LLM が知識から別の同姓を書いた場合に誤って戻す可能性は残る（復元は文書の持ち主の手元で行うので第三者への漏洩ではないが、文意が変わる）— README に制限として書く
- 偽名の衝突: 同じ scope で別の実体に同じ偽名が出たら §7 の scope 内衝突で再抽選。偽名が元の文書本文に既に含まれていたら §7 の既存表層で再抽選

## 10. API と互換性

```python
Surrogate(
    kind="auto",              # "person" | "org" | "facility" | "location" | "address" | "auto"（Entity の型で選ぶ）
    strategy="hmac",          # "random" | "hmac" | "vault"
    key=None, key_env=None,   # 既存
    semantic=True,            # False で v0.8 までの挙動（型が合った偽名だけ。カテゴリ判定・分類ヘッドを使わない）
    link_surnames=True,       # 既存
    keep=None,                # 既存（ADDRESS の "city"）
    knowledge=None,           # PublicKnowledge | パス | None。同梱の禁止リストは常に有効
    denylist=(),              # 利用者の禁止語
    min_margin=0.15,          # 分類ヘッドの閾値
    generator=None, normalizer=None,   # 既存（generator を渡すとカテゴリ別生成器を使わない）
)
```

- `Policy(rules={"PERSON": Surrogate(kind="person"), ...})` の書き方は不変。`default=Surrogate(kind="auto")` を新しく許す。`auto` は PERSON→person、ORG→org、FACILITY→facility、LOCATION→location、ADDRESS→address。それ以外の型（EMAIL・PHONE など）は `<TYPE_hmac8>` トークンに退避する（fail-closed）
- **v0.9.0 の破壊的変更**: (1) `semantic=True` が既定になるので hmac の偽名は v0.8 と変わる（`semantic=False` で従来値。既存のゴールデン値テストは `semantic=False` に付け替えて維持し、`semantic=True` のゴールデン値を追加）、(2) `PseudonymStore` に `iter_tokens` が必要、(3) `Entity` に `vector` フィールドが増える（既定 None。位置引数で作っている呼び出しは影響なし）
- 同梱の禁止リストによる再抽選は `semantic=False` でも効く（B は既定の挙動にする。これも v0.8 との差分として CHANGELOG に書く）
- Presidio 互換の名前は変えない（`Surrogate` は Presidio に無い operator）

## 11. テスト・検証

| 観点 | テスト |
|---|---|
| A 逆算不能 | 同カテゴリの原文 N=200 件に対し、(i) 偽名に元の 2 字以上の部分文字列が無い（カテゴリ由来の語 — 種別語・法人格・都道府県名・地方名 — を除く）、(ii) 別の鍵で生成した偽名の一致率が 1/プール規模 の 2 倍以内、(iii) 攻撃者が鍵なしで総当たりした正解率がカテゴリ母集団の逆数 × 2 以内（プロパティテスト） |
| B 公知の実体 | 全テンプレートから 1,000 件ずつサンプルし同梱の禁止リストに 0 件。著名人・大企業・大学・球団を含むフィクスチャ `PublicKnowledge` を `knowledge=` に渡して弾かれること |
| C 粒度 | 市 → 同じ市にならない、都道府県 → 同じ都道府県にならない、地方ブロックが保存される、カテゴリが体系外にならない |
| D 一貫性 | 既存のゴールデン値（`semantic=False`）維持、`semantic=True` の新ゴールデン値、hmac で文書横断の同値、random で scope 内の同値 |
| 生成器の隔離 | `Features` に元の値が無いことを型で保証（`Features` は `str` フィールドとして元の値を持たない）。フィルタ全滅時にトークン退避になり元の値が返らない |
| 分類ヘッド | §8 の受け入れ値。encoder SHA 不一致で読まないこと。ヘッド無しで規則のみでも全 API が動くこと |
| 往復 | 敬称の付け替え（さん→氏）、姓だけの言及、LLM が文を追加した応答、偽名が含まれない応答（何も戻さない）、別の scope の偽名（戻さない）、`iter_tokens` 未実装の Store で明確なエラー |
| 性能 | 100 文書のベンチで `semantic=True` と `False` の差が NER 時間の 5% 以内 |

## 12. 実装の順序（実装計画の単位）

1. 土台: `Entity.vector`、`TransformContext.text`、`PseudonymStore.iter_tokens`（Memory / SQLite）、`CharHeadNerDetector` のスパンベクトル出力
2. `RuleCategorizer` + `Features` + カテゴリ別生成器 + `SafetyFilter`（分類ヘッド無しで動く状態。ゴールデン値を作る）
3. 禁止リストの生成ツールと同梱、`knowledge=` 連携
4. `tools/train_category_head.py`、`HeadCategorizer`、`load_char_model` での読み込み、本番モデルへの配置
5. `strategy="vault"` と `restore()` の拡張、往復テスト
6. §11 の検証テスト一式、README「Surrogate」節と仕様（`2026-08-08-privacy-masker-design.md`）の追補、CHANGELOG（v0.9.0）、開発記録

実装計画は 2 本に分ける: **計画 A**（1・2・3・5・6 = operator と往復。分類ヘッド無しで完結し、規則だけでカテゴリ保持が動く）、**計画 B**（4 = 分類ヘッドの学習と組み込み。データと本番モデルに依存し、受け入れ精度の判定が要る）。

## 13. リスクと未決

- 分類ヘッドが種別語のない名前で受け入れ精度に届かない可能性。届かない場合は閾値で「その他」に落とす（安全側）ので機能は成立し、
  カテゴリ保持の効果が種別語のある名前に限られるだけ
- hmac + 分類ヘッドの文書横断カテゴリ不一致（best effort と明記済み）
- 禁止リストの網羅は Wikipedia 見出し語の範囲。無名の実在事業者との一致は残る（現状と同じ。README に明記）
- LOCATION の実在地名の再利用は「同じ地方ブロックの別の地名」に限るが、地名 + 文脈（「〜市の〜病院」）で元が絞れることはない（別の市になる）
- 往復での偽姓の誤復元（§9）

## 14. 関連

- 現行実装: `privacy_masker/operators/surrogate.py`、`privacy_masker/data/ja_pools.py`、仕様 §21・§22（`docs/superpowers/specs/2026-08-08-privacy-masker-design.md`）
- 公知判定: `privacy_masker/annotators/knowledge.py`（`PublicKnowledge`、`jawiki_titles_ns0.pmk`）
- NER: `privacy_masker/char_head.py`（`last_hidden_state` は 805 行付近）、本番モデル（隠れ層 256）
- 値プール（移植元）: `tools/synth_v11_values.py`（カタカナ外国人名・ローマ字・ひらがな・半角カナ・国名・外国地名）
- Presidio の置換: `replace`（`<ENTITY_TYPE>`）、Faker は `custom` lambda の例のみ（一貫性なし）、往復は `InstanceCounterAnonymizer` のプレースホルダ方式
