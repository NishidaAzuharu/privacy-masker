# Changelog

## Unreleased

- **文書**: README の `HandleDetector` の説明を実際の検出範囲に合わせて修正。拾うのは `@handle`・行頭の発言者名・`user=` などのキーの値の形だけで、文中にそのまま書かれたハンドル名は拾わない（NER との併用が前提）こと、NER も同じ箇所を拾うと PERSON として扱われることを明記した。動作の変更はなし。

## 0.11.0 - 2026-10-05

- **examples**: `examples/library_usage.py`・`examples/anonymize_demo.py` の NER を Hub の文字ヘッドモデル（`DEFAULT_MODEL_ID`）に切り替え、
  `PRIVACY_MASKER_MODEL`（フォルダか Hub ID）で差し替え可能に。公知データが無いときは standard の節を飛ばす。
- **追加（Hugging Face Hub）**: `CharHeadNerDetector(model=...)` がローカルのフォルダに加えて Hub のリポジトリ ID（`owner/name`）を受け付ける
  （`privacy_masker/hub.py` の `resolve_model_dir`。`snapshot_download` で推論に必要なファイル（`*.json`・`*.safetensors`・`*.txt`・`*.model`）だけを取得してキャッシュ。
  `revision` で固定、`cache_dir`、`local_files_only`）。取得できないときは `ConfigurationError` で止まる（空のモデルで続けない）。モデルオブジェクトにこれらの引数を付けると `ConfigurationError`。
  公開用のツール `tools/publish_model_hf.py`（ステージング → `char_head_config.json` の `averaged_from` から学習機の絶対パスを消す → 読み込み・検出のスモークテスト →
  モデルカード `docs/hub/model_card_template.md` と由来 `privacy_masker_model.json` の生成 → 学習機のパス・メールアドレスが残っていないかの検査 → `--dry-run` でなければ
  `create_repo(exist_ok)` + `upload_folder`）。トークンは引数で受け取らず `hf auth login` / `HF_TOKEN` に任せる。テスト `tests/test_hub.py`・`tests/test_publish_model_hf.py`。
  配布先は `privacy_masker.hub.DEFAULT_MODEL_ID` = `az225/privacy-masker-ja-pii-ner-30m`（重みは Apache-2.0）。examples の既定はこの Hub モデルで、
  環境変数 `PRIVACY_MASKER_MODEL` か `--model` でローカルのフォルダか別の Hub ID に差し替えられる。README「NER モデルについて」に配布先を記載。
  モデルカードは第三者向けに「privacy-masker が使うモデル」の簡潔な仕様（使い方・構成・学習データの出所とライセンス・制限・SHA-256）だけを書き、内部の版名は出さない。
  配布する `char_head_config.json` からは学習時の内部情報（`averaged_from`・`average_weights`・`data_tag`・`epoch`）を外す。
- **移植性（テスト・ツール）**: ツール・examples・テストの既定パスを開発機固有の値から `~/privacy-masker-data`（環境変数 `PRIVACY_MASKER_DRIVE_DIR`）に、
  外部評価 gold の既定を `PRIVACY_MASKER_EXTERNAL_EVAL`（既定 `<データフォルダ>/external_eval`）に変更。`tests/test_char_head_chunks.py` の実 tokenizer のパスも
  `PRIVACY_MASKER_DRIVE_DIR` から組み立てる。`tools/eval_ood_dashboard.py`・`tools/eval_all.py` は import 時にデータフォルダが無いと `SystemExit` していたため、
  データフォルダの無い環境（CI など）では `tests/test_eval_ood_f1.py` の収集で pytest 全体が INTERNALERROR で止まっていた → `drive_dir(require=False)` で import し、
  存在確認を `main()` の冒頭に移した（CLI として実行したときの挙動は不変）。生成する Colab ノートブックのデータパスも中立な既定にした。
- **公開準備**: 内部の開発記録をリポジトリから外し、残した設計文書・CHANGELOG から内部文書への参照と非公開の評価数値を除いた。
  `README.md` を公開（GitHub / PyPI）向けに書き直した。`pyproject.toml` の Repository / Changelog URL を修正し、`.gitignore` に `.DS_Store` を追加。

## 0.10.0 - 2026-10-02

- **変更（意味を保つ Surrogate のカテゴリ規則）**: `RuleCategorizer`（`privacy_masker/operators/semantic.py`）の系統的な癖を、Qwen ラベル付き・監査済みの実文書
  で検証して直した（設計は開発記録、非公開）。変わるのは意味を保つ偽名の種別（カテゴリ）だけ。
  - ORG: 公的機関の語（`省・庁・局・議会・機構・自衛隊・陸軍・海軍・空軍・国防軍・解放軍・連合軍・警察・裁判所・公社・事業団・独立行政法人` など）を業種語より先に見て
    `government` にする（航空局・交通局が `transport` にならない）。`局` は 放送局・テレビ局・ラジオ局（`media`）と 郵便局・支局・薬局・事務局・編集局 を除き、裸の `軍` は含めない。
    委員会は公的な修飾語が直接隣接するとき（`国家・中央・選挙管理・公正取引・教育・公安・人事・原子力規制・労働・農業・監査・都市計画・欧州・国連・証券取引・運輸安全`、
    `連邦通信・連邦準備・連邦取引・連邦選挙・連邦捜査` など）と、都道府県・市区町村の文字を含む接頭（`〜県…委員会`・`〜市…委員会`）のときだけ `government`、それ以外は `department`。
    球団名（`〜ホークス・ジャイアンツ・タイガース …`・`球団`・`巨人軍`）、`通信社・通信`（電気通信を除く）、`テレビ局・ラジオ局` → `media`。`〜の会` → `association`（`party` から移す）、
    `機構` → `government`（`association` から移す）。`電鉄・電気鉄道・電気軌道` → `transport`（`manufacturing` より先）、`電気通信・テレコム` → `it`。
    `電気`（製造）・`商事`（小売）・`ホーム`（建設）・`グループ`（部署）は種別語から外した（既定の `service`）。これらを末尾に持つ名前のコアの切り出しは、
    `core_split` 専用の語尾 `_ORG_CORE_SUFFIXES`（商事・商会・ホーム・電気・グループ・サービス・興業・産業・事業。カテゴリには使わない）で保つ。
  - LOCATION: 判定順を country → prefecture → district → foreign（一覧・カタカナ+行政区画）→ city → town → foreign（カタカナ全般）→ nature に変えた。
    国名は生成プールの `COUNTRIES` に加え、`ja_pools.COUNTRIES_EXTRA`（171 件: 国連加盟国・主要地域・構成国・歴史的な国のカタカナ表記）、`COUNTRY_ALIASES`
    （28 件: 米国・英国・大韓民国 などの漢字略称・正式名と単漢字略称）、`〜共和国・〜帝国・〜合衆国・〜連邦` で終わる名前 → `country`（`王国・公国` は検証で外国の地名の判定が多く足さない）。
    `〜地区・駅前・商店街・繁華街・界隈` と選挙区（`〜1区`）→ `district`（city より先）。`ja_pools.FOREIGN_CITIES_KANJI`（25 件: 上海・北京・香港・釜山 など）と
    カタカナ+`市・町・州・省` → `foreign`（city より先）。`nature` は本体（接尾語の前）が 2 字以上で、名前全体が市区町村・町名の一覧に無いときだけ。
  - FACILITY・PERSON・ADDRESS の判定は不変。
- **安全性**: 国名・外国の地名に分類される名前が増えたことで開いた漏洩経路と、検証の途中で見つかった既存の経路を、`avoid_parts`（偽名が含んではいけない元の部分）を
  広げて塞いだ（いずれも避ける部分を増やす方向だけの拡張。数値は 400 scope での元の部分の再出現件数）。
  - 国名の別名・略称・前身国家（米国・大韓民国・ソ連 など）が `country` になったことで、偽名に同じ国の標準名（アメリカ・韓国・ロシア）が出る経路が開いた（最大 12/400）
    → `ja_pools.COUNTRY_CANONICAL`（別名 → `COUNTRIES` の標準名）を足し、`avoid_parts("location")` が標準名も避ける（0/400）。
  - カタカナ+州・省の語幹が避けられていなかった（ニューヨーク州（語幹が `FOREIGN_CITIES_KATA` にある）で 9/400）→ 語幹を外す接尾語に 州・省 を追加（0/400）。
  - 〜地区・選挙区・自然地名の語幹（若葉地区 → 若葉）が避けられていなかった（既存の問題、最大 18/400）→ district / electoral / nature の接尾語を外した語幹（2 字以上）も避ける（0/400）。
  - 国名の正式名（〜共和国・〜連邦・〜帝国・〜合衆国）が `country` になったことで、短い国名（フランス共和国 → フランス 18/400 など、本文なしの経路）が偽名に出る経路も開いた
    → 元の値に含まれる `COUNTRIES` の項目も避ける（0/400。既存の 北アイルランド → アイルランド・インドネシア → インド も同時に解消）。
  - 委員会の規則で一致が名前の先頭から始まる名前（中央青葉人事委員会・若葉県教育委員会・汐見町議会 など）は、コア語の内部にある生成器の固有部分の語
    （`ORG_CORES`・`KATA_CORES`・`TOWNS` の 青葉・若葉・汐見）が避けられず偽名に再出現していた（既存の問題、最大 35/400（施設名でも 33/400））→ ORG・FACILITY・LOCATION で、
    元の値に含まれるそれらの語彙の語も避ける（0/400）。
  - 別名に 〜連邦・〜帝国・〜合衆国・〜共和国 が付いた名前（ソ連邦・大韓帝国・米合衆国 など）は `COUNTRY_CANONICAL` にも `COUNTRIES` にも完全一致せず、
    同じ国の標準名が偽名に出ていた（最大 20/400。別名×語尾の総当たり 334 通り中 143 通りで漏れ）→ 元の値に含まれる別名の標準名も避ける（総当たりで 0。`COUNTRY_CANONICAL` に無い稀な表記 — アラブ連合共和国 → エジプト 7/400、ブリテン帝国 → イギリス 6/400 — は許容として残した）。
    `COUNTRY_CANONICAL` に ソヴィエト連邦・ソヴィエト社会主義共和国連邦・大清帝国 を追記した。
  - 先頭から一致する委員会の規則（自治体接頭の委員会、部署の `^…委員会$`）ではコアが切り出せず、名前全体しか避けられなくなっていた
    （営業委員会 の「営業」が偽名に 46/400）→ core_split 専用の語尾 `_ORG_CORE_SUFFIXES` に 委員会 を足した（カテゴリは不変。0/400）。
    あわせて、department の偽名生成器が部署名の元の機能名を再利用していた分（既存の問題 営業部 24/400・人事部 36/400、本版で 営業推進委員会 0 → 42/400）は、
    ORG で `DEPT_FUNCTIONS` の語も避けて 0/400 にした。
  - 検証で一度足して戻したもの: 委員会の修飾語と「委員会」の間に最大 6 字を許す形（労働安全衛生委員会 などの私的な委員会が `government` になり固有部分が漏れる）、
    委員会の自治体接頭に 省・州（省エネ・九州 で誤発火）。最終形は隣接する修飾語だけ（`連邦` は 連邦通信・連邦準備・連邦取引・連邦選挙・連邦捜査 の複合語に限る）。
  - v0.8 の hmac ゴールデン（`tests/test_surrogate.py`）は無変更で通る。
- **検証**（`tools/rule_quirk_check.py`、Qwen ラベル付き実文書。結果は開発記録、非公開）:
  (a) 監査済みの不一致 ✗（Qwen ≠ 規則 かつ 監査 ok、規則ラベル付きスパン）は org 303 → 201、facility 19 → 19、location 425 → 260、合計 747 → 480（0.643）。
  **設計の閾値「半分以下（≤ 300）」は未達。**
  (b) 新しく発火するスパン（旧 OTHER → 新カテゴリ）の Qwen（監査 ok）との不一致率は org 23/290 = 0.079（**閾値 5% 未達**。局 17/54・議会 5/38・解放軍 1/4）、
  location 11/300 = 0.037（達成。country 5/225・foreign 6/75）、facility は新規発火なし。
  (c) 発火しなくなるスパンは org 66・location 29。OTHER 率（規則が決めないスパンの割合）は org 0.734 → 0.690、facility 0.650（不変）、location 0.295 → 0.220。
  **閾値 (a)(b-org) は未達で、残りは分類体系の違い**（Qwen は外国の地名を district / city / nature に分けるが規則は `foreign` 1 つ、〜町 / 〜村 は Qwen では town、
  省庁の「〜局」を Qwen は department と呼ぶ）と設計で据え置いた癖（〜町・システム・軍の部隊名 など）。指標 (a) は正しく発火するようになった件数を数えず、
  局・議会を戻すと ✗ は 24 件減るが正しい発火 70 件を失う。閾値未達のまま採用した（2026-10-03。残りは分類体系の違いと据え置きの癖）。
- **ゴールデン**: `tests/golden/surrogate_semantic_v0_9.json` を `surrogate_semantic_v0_10.json` に作り直した（`tools/freeze_surrogate_golden.py` の既定の出力先も v0_10）。
  今回の癖の例（ORG 9 件（架空名）・LOCATION 5 件（架空名と公知の国名・都市名））を CASES に足した。既存 26 ケースの値は変わっていない（既存ケースのカテゴリは変わらず、`avoid_parts` が増えた
  2 ケースも最初に引いた偽名が新しい部分を含まないため）。
- **互換**: `semantic=True` の hmac の偽名は、カテゴリが変わる名前と、`avoid_parts` が広がって引き直しになる名前で 0.9.0 と変わりうる（`semantic=False` は不変）。
- **不変**: 国名の生成プール `COUNTRIES`（45 件、v0.8 互換）。偽名の国名は従来どおりこの中から選ぶ（`COUNTRIES_EXTRA`・`COUNTRY_ALIASES`・`FOREIGN_CITIES_KANJI` は判定にだけ、
  `COUNTRY_CANONICAL` は `avoid_parts` にだけ使う）。
- **追加（分類ヘッド 第 2 段、学習専用）**: 補助カテゴリ損失で encoder を学習し直すための部品を `privacy_masker/char_head.py` に足した。`span_ctx_batch`（`span_vector(features="span_ctx")` と同じ規則の batched torch 実装で、float16 への丸め後にバイト一致。各区画の L2 正規化は分母にも勾配を流す微分可能な形）、
  `CategoryAuxHead`（kind 別の `Linear(3H, k)`。`CharBioNerModel` の重みには含めない）、`category_loss`（batch 内のカテゴリスパンの kind 別 cross entropy の平均、スパンが無ければ 0）、
  `encode_chunk(category_spans=)`（`category_spans` はチャンク相対の文字位置で渡す。呼び出し側 = ノートブックの `build_samples` がチャンクに完全に含まれるものだけを相対化して渡し、範囲外は `ValueError`、切れたスパンの件数はノートの `n_cat_cut`）、`CharNerCollator` の `cat_spans` / `cat_mask`、
  `sqrt_class_weights`・`category_spans_by_text`・カテゴリ語彙 `CATEGORY_KINDS` / `CATEGORY_NAMES`。**推論経路と保存形式は不変**（`category_spans` 省略時は従来どおり、collator の追加キーは `forward` が無視する。ゴールデン値は不変）。
  ツール: `tools/category_head_data.py --judge-split`（head_judge を文書単位・source で層化・seed 42 で A / B に分け、A を `spans_dev.jsonl`、B を `spans_judge.jsonl`、head_train + head_dev を `spans_train.jsonl` に。manifest に `judge_split` の分割表）、
  `tools/make_multitask_notebook.py`（文字ヘッドの学習ノートブックから多タスク版を生成。`loss = ner + 0.3 × cat_loss`、教師を本文の完全一致で結合し結合漏れ・`surface_sha` の不一致を検査、補助層を optimizer の `aux` group に足してエポックごとに `category_aux.safetensors` と manifest の `category_aux` を保存）、
  `tools/bench_category_head.py`・`tools/bench_category_head_realcheck.py` の `--model`（既定は本番モデル）。
  **分類ヘッドは引き続き同梱しない。多タスク学習の encoder も不採用**: 盲検セットと 4 評価セットで NER の採用条件を満たさず、その上で作り直したヘッドも judge-B で受け入れた kind が無かった（judge に回した linear / span_mean は 3 kind とも dev で min_margin が決まらない。判定は参考。開発記録、非公開）。本番モデルのフォルダは無変更で、カテゴリは規則のみ
- **追加（分類ヘッド v2、第 1 段）**: スパン特徴量 `span_ctx`（スパン平均に直前・直後 16 サブワードの平均を連結、区画ごとに L2 正規化、30m で 768 次元。
  `char_head.span_vector(features=)`・`encode_span_vectors(features=)`・`predict_char_spans(vector_features=)`。既定は従来の `span_mean`）。
  `HeadCategorizer` は `category_head.json`（version 2）の `head_type`（`linear` = 従来の純 Python / `mlp` = Linear → GELU(tanh) → Linear を numpy で推論。numpy は関数内で遅延 import）と
  `features` を読み、未知の値や重みの入力次元と `features` の不一致（encoder の `config.json` の `hidden_size` から決まる次元）は `ConfigurationError`。
  `HeadCategorizer.load(model_dir, head_dir=None)`（ヘッドを別フォルダから読み、encoder の SHA は `model_dir` と照合する）。
  `CharHeadNerDetector` は読み込んだヘッドの `features` で `Entity.vector` を計算する（`vector_features`。ヘッドが無ければ `span_mean` で、ゴールデン値は不変）。
  ツール: `tools/train_category_head.py` に `--features span_mean|span_ctx`・`--head linear|mlp`（`--hidden` / `--dropout` / `--patience`）・`--class-weight sqrt|inv`（既定 sqrt、上限 5）・
  `--judge`（dev で選んだヘッドと min_margin を judge で kind ごとに 1 回判定。`--install` は `--judge` 必須）、`tools/category_head_data.py --labels`（train / dev / judge のスパン。
  LOCATION の種別語を削る写しを市区町村に加えて都府県・丁目にも拡大、削った後 1 字以下になる写しは作らない）、実文書スパンの Qwen ラベル付け・監査・選別 `tools/category_labels.py`（`plan` / `audit-plan` / `filter`）と
  ノートブック生成 `tools/make_category_labels_notebook.py`（`qwen3_8_27b_category_labels.ipynb`、`--dryrun`）、1,870 件の確認 `tools/bench_category_head.py`・
  `tools/bench_category_head_realcheck.py`（`--head-dir` で候補を読み、処理を `main()` に包んだ）。
  **分類ヘッドは引き続き同梱しない**: 実文書のラベル（Qwen3.8-27B-FP8 + 同じ Qwen の監査 + 機械検査）で学習・選択・判定しても全 kind が受け入れ条件に届かなかった
  （ORG は dev で min_margin が決まらず judge の macro accuracy 0.869、FACILITY は judge の precision 0.878、LOCATION は 0.820 / 0.833。開発記録、非公開）。
  本番モデルのフォルダは無変更で、カテゴリは規則のみ

## 0.9.0 - 2026-09-28

- **追加（v0.9.0）**: 分類ヘッドの仕組み（計画 B、設計 §8）。規則（種別語）で既定カテゴリに落ちた ORG / FACILITY / LOCATION を、NER と同じ forward の
  encoder スパンベクトル（`Entity.vector`）+ 型ごとの線形層 + softmax で分類する（1 位と 2 位の確率差が `min_margin` 未満なら既定のまま。追加の forward 無し、
  推論は純 Python で新しい実行時依存なし）。`HeadCategorizer`（`category_head.safetensors` / `category_head.json` を読み、encoder の SHA-256 が
  一致しない・JSON や重みの形が違う・kind の `min_margin` が無い（null・欠落・[0, 1] 外）なら `ConfigurationError`、safetensors が無ければ `ImportError`）、
  `CharHeadNerDetector(category_head="auto")`（モデルフォルダに `category_head.json` があれば自動で読む。`True` / `False`、bool 以外は `"auto"` のみ）、
  `Surrogate(category_head=..., min_margin=None)`（`None` ならヘッドの推奨値）、
  `char_head.encode_span_vectors()`、ツール `tools/category_head_data.py`（学習データから規則ラベル・合成の form ラベル・種別語を削った写しのスパンデータセット）/
  `tools/train_category_head.py`（既定の書き出し先は候補 `<--spans>/candidate`。kind ごとの受け入れ判定 `accepted` を記録し、モデルフォルダへは
  `--install` で `accepted` の kind だけを書く。`--merge` / `--kinds` / `--cache-dir`。min_margin の見つからない kind は保存しない）。
  **ヘッドは採用せず、v0.9.0 のカテゴリは規則のみ**（本番モデルのフォルダにヘッドのファイルは無い）: ORG は valid で受け入れ条件
  （macro accuracy 0.905 ≥ 0.90、min_margin 0.55 で種別語を削ったスパンの precision 0.967 ≥ 0.95）を満たしたが、実文書 1,870 件で種別語を削って測ると
  同じ margin で 0.741（n=27）、FACILITY・LOCATION は統合 1 回を試しても valid で届かなかった（開発記録、非公開）。ゴールデン値はヘッド無しで不変
- **変更（v0.9.0）**: `examples/try_privacy_masker.py` は FACILITY・LOCATION も意味を保つ Surrogate で置換する（従来はトークン化）。起動時に「分類ヘッド: あり/なし」を表示する
- **修正（v0.9.0）**: 区切り（空白・中黒）付きのカタカナ・平仮名・ローマ字の氏名で、元の姓・名を偽名から除外していなかった（`avoid_parts` が名前全体しか
  避けず、元の名や姓を再利用した偽名が出えた）。姓・名のプール語との前方・後方一致で要素に分けて避ける（ローマ字は大文字小文字を区別しない）。
  姓リンク経路（`Surrogate._ok_linked` / `_given_pool`）でも「山田・太郎」のような中黒付き漢字名の名を除外する（従来は「・太郎」のまま照合して素通りしていた）。
  「四日市市」型の市区町村名の延長（`MUNICIPALITIES_WITH_INNER_SHI`）に境界条件（語彙の直前が先頭か 都/道/府/県/郡）を付けた
- **修正（保留 Minor の解消、v0.9.0）**: 意味を保つ Surrogate（v0.9.0）の最終レビューで保留にした Minor 群をまとめて解消した。
  `semantic.py` の内部ヘルパーを公開名に改名（`address_city_and_town`・`address_granularity`・`HAS_BUILDING_RE`、
  `RuleCategorizer.type_word_span()` と、偽名に混入させないコア語の切り出し位置 `RuleCategorizer.core_split()` を追加）、
  性別判定を偶奇規約 + 明示の上書き（`GENDER_OVERRIDES`。読み取り専用）に変更、
  区切り無しの長いカタカナ外国人名（「マイケルジョンソン」型）を姓+名として生成・列挙できるようにした
  （`enumerate_candidates` の禁止リスト閉包にも反映）。カタカナ外国人名は元の名・姓を要素ごとに避ける
  （「・」区切りの要素、区切り無しは `GIVEN_KATA` の最長前方一致と残り。従来は名前全体しか避けず、元の名を
  再利用した偽名が出えた）。「四日市市」のように市区町村名の途中に市/区/町/村を含む市で
  住所の分解が途中の「市」で切れ、`keep="city"` の出力から市区町村名末尾の「市」が 1 字落ちていた不具合を修正し
  （延長は該当 8 市の語彙 `MUNICIPALITIES_WITH_INNER_SHI` に当たるときだけ。「市谷加賀町」「市場町」のように市/区で
  始まる町名は取り込まない）、丁目の漢数字表記（`一〜五丁目`）を表層特徴として保存、北海道の追加 5 市
  （函館・小樽・帯広・釧路・北見）を semantic 経路の判定・地方ブロック引きに追加（既存の `PREF_CITIES` タプル
  自体は無変更）。`SurrogateFilter` の却下理由の判定順を設計 §7 のとおりに明確化し、文書全文のメモ
  （`_text_cache`）を `transform`/`transform_batch` の例外経路も含めて確実に破棄するようにし、`restore_batch` は
  scope の対応表を1回だけ読むよう変更（最初に一度だけ検査するため、空の入力 `restore_batch([])` でも CryptoProvider が
  無い・vault の Surrogate があるのに store が `iter_tokens` を持たない場合は `ConfigurationError` を送出する＝fail-fast）、
  `Policy.has_strategy(None)` が strategy 属性の無い operator に誤って一致しないよう修正、一部の例外メッセージを英語に統一した。ツールでは `tools/build_surrogate_denylist.py` の
  manifest パス生成を `str.replace` から `os.path.splitext` ベースに変更し（二重拡張子でも安全）、manifest
  ファイル末尾に改行を追加。ライブラリの公開 API・既定挙動は変えない（偽名の一部は候補集合の変化で変わりうる。
  ゴールデン値テストは別途再固定）
- **修正（v0.9.0）**: `Surrogate` は正規化後に空になる値（空文字・空白のみ）を渡されると、フェイルセーフを経ずに
  通常の生成器（PERSON なら「姓のみ」扱い）から実在しうる普通の偽名を返していた。`apply()` の先頭で検出し
  `<TYPE_xxxxxxxx>` トークンへ退避する（random / vault は保存して同一 scope で一貫させ、vault は原文（空白等）も
  保存して `restore()` で復元できる）
- **修正（v0.9.0）**: `Surrogate(kind="auto")` が対象外の型（EMAIL・PHONE 等）を `<TYPE_xxxxxxxx>` に退避する経路が store に保存していなかったため、
  `strategy="vault"` で復元されず（`unresolved` にも出ない）、`random` では同じ値が別トークンになり得た。random / vault は保存して同一 scope で一貫させ、vault は原文も保存して往復させる
- **修正（v0.9.0）**: `semantic=False`（v0.8 互換経路）のカタカナ単独の人名が、公知照合（同梱禁止リスト）で 10 回連続却下されて
  `ConfigurationError` になっていた（本番 NER + 実文書 1,870 件で 36 文書。v0.8 のカタカナ単独プール 60 語のうち 58 語が Wikipedia 見出し語と衝突）。
  姓だけ・名だけと同じく辞書語として公知照合の対象外にし、フィルタ却下で候補が尽きた場合は戦略に関わらず `<TYPE_xxxxxxxx>` に退避する（設計 §7）。
  生成器が原文と区別できる値を作れない設定ミスだけは従来どおり送出する
- **追加（v0.9.0）**: 意味を保つ Surrogate。`Surrogate(semantic=True)`（既定）は元と同じ粗いカテゴリ（病院・小学校・製造業・同じ地方ブロックの市 …）の架空の名前に置換
  （カテゴリは種別語の規則。`operators/semantic.py`・`surrogate_generators.py`・`surrogate_filter.py`）。`kind="facility" / "location" / "auto"`、`knowledge=` / `denylist=`。
  安全フィルタ: 元との一致・元の姓/コアの混入・**同梱の禁止リスト**（`data/surrogate_denylist.pmk`。生成器の候補全体 654,258 件 × Wikipedia 見出し語の衝突分 15,393 件（pmk 収録 15,376 件、
  うち3文字の英字略称が8,998件を占め latin_abbr の ORG 偽名の多様性を制約する）、`tools/build_surrogate_denylist.py`）・
  文書内の既存表層・利用者の禁止語。全部落ちたら `<TYPE_xxxxxxxx>` に退避（元の値を返す経路は無い）。**往復**: `Surrogate(strategy="vault")` + `restore()`（完全一致、偽姓は 2 字以上）。
  土台: `Entity.vector`（文字ヘッド NER の encoder スパンベクトル。`CharHeadNerDetector(span_vectors=True)`）、`TransformContext.text`、`PseudonymStore.iter_tokens`、`RestoreResult.restored`
- **破壊的変更（v0.9.0）**: (1) `semantic=True` が既定になり hmac の偽名が v0.8 と変わる（`semantic=False` で従来値。ゴールデン値テストは両方を固定）、(2) 自作 `PseudonymStore` は `iter_tokens(scope_id)` の実装が必要、
  (3) 同梱禁止リストは `semantic=False` でも掛かる（衝突した偽名は再抽選される）、(4) `Policy.has_strategy` が Tokenize 以外の operator の `strategy` も見る、
  (5) hmac の文書横断の一貫性に例外ができた: 候補が変換中の文書本文に既に出ていると引き直すため、同じ値・同じ鍵でも文書が変われば別の偽名になりうる
  （`semantic=True` の全経路と、`semantic=False` でも PERSON の姓リンク経路が対象。`semantic=False` の ORG・ADDRESS の従来経路は文書本文を見ないので文書によらず決まる）
- **注記（v0.9.0）**: `RestoreResult.restored` は復元した**出現数**であってトークンの種類数ではない（同じ偽名が 3 か所にあれば 3 と数える）
- **変更（推奨モデル）**: `examples/try_privacy_masker.py` の既定 NER を、学習データを更新して再学習した文字ヘッド 30m モデル（epoch 1〜4 の重み平均）に
  変更（2026-09-23 採用）。学習データは、Qwen にスロット付きテンプレートを書かせた合成 14,720 文書と UD Japanese GSD の学習・開発分割 7,556 文を、
  従来の合成データに足したもの。開発に未使用の盲検セットと 4 評価セットのすべてで完全見逃しが減少。
  既知の後退は過剰マスクの増加と住所の端の削れ、ハンドル名 1 件。
  新しい合成データだけで学習したモデルは帳票の裸の姓・ハンドル名・字間空白・部署名を落とし不採用。ライブラリ本体の変更なし
- **修正（ツール）**: `tools/gsd_train_to_ner.py` が GSD の NE タグ（BILOU）の U/L を読まず、単独トークンの実体を無ラベル・末尾を欠落させていた。
  この不具合データで学習した 1 回目のモデルは無効とした（開発記録、非公開）
- **追加（ツール）**: 学習データ生成の一式 `tools/synth_v11_values.py`（値プールと表記生成。Nemotron ペルソナ・KEN_ALL の
  スナップショット、評価セット表層の機械除外）/ `synth_v11_plan.py`（10 ジャンル・49 サブジャンル × 書式 × 文体 × 長さ × スロット処方の
  網羅行列を seed 固定で 8,000 件の計画に）/ `synth_v11_templates.py`（Qwen 出力の受け入れ検査・近重複・枠）/ `synth_v11_fill.py`
  （値の差し込みと位置の確定、役割語の対の負例）/ `make_synth_notebook_v11.py`（Qwen 生成用の Colab ノートブック）/
  `gsd_train_to_ner.py`（UD Japanese GSD の学習・開発分割 7,556 文を `neutral` 付きで変換）/ `build_data_v11.py`
  （新しい合成データ単体と、従来の合成データを足した学習データを書き出す）/ `make_train_notebook_v11.py`（neutral を損失と過剰マスクから
  除き、ベースモデルを commit に固定）。網羅行列から Qwen にスロット付きテンプレートを書かせ、値は機械差し込み。
  従来のテンプレート・フレーム生成は新しい合成データには含めない。
  ライブラリ本体の変更なし
- **修正（文書）**: 事実誤りを 5 件訂正。README の重複解決の順序（「同じ型なら長いスパン、次にスコア」→ 実装どおり
  「スコア → 長さ」）、公知データの件数（見出し語 246 万行 → ハッシュ 282 万エントリ）、評価記録の数値 3 件
- **変更（推奨モデル）**: `examples/try_privacy_masker.py` の既定 NER を token 分類モデルから自前学習の **文字ヘッドモデル（30m、
  epoch 4〜6 の重み平均）**に変更（2026-09-17 採用）。
  `char_head.safetensors` の有無で `CharHeadNerDetector` / `TransformersNerDetector` を自動選択。学習・開発に未使用の
  盲検セットで前の本番モデルを全指標で上回る。
  既知の後退は姓の平仮名の読み、文語の短い名の部分被覆。ライブラリ本体の変更なし
- **追加（ツール）**: 第 2 の盲検評価セットの作成 `tools/blind_v2_from_kwdlc.py`（公開コーパスの構文注釈から括弧削除を戻して固有表現を
  復元、IREX → 本ライブラリの型に写像）、`tools/eval_ood_f1.py --only-extra`、
  学習データの構築 `tools/build_data_v10.py`（前版の学習データに 帳票の名比率 85%・1,600 文書・珍しい姓の重み 3・字間空白の姓 5% を加えた）、
  `tools/make_train_notebook_v10.py`。`build_data_v9.make_form_docs_v9` に `name_fn`、
  `build` に `form_kwargs` / `surname_bank_fn` を追加（既定動作は不変）
- **追加（ツール）**: 学習データの構築 `tools/build_data_v9.py`（テンプレート上限 5、LLM 負例 2,000 でハード負例優先、
  帳票のメール・URL を非エンティティ値に、カタカナ人名 600・裸の町名 500 の文書、評価セットの地名表層も機械除外）、
  `tools/make_train_notebook_v9.py`、`tools/average_checkpoints.py`（文字ヘッドモデルのチェックポイント平均、GPU 不要）
- **追加（ツール）**: 学習データの構築 `tools/build_data_v8.py`（テンプレート上限 6、新規文書半減、帳票・ログの
  裸の姓フレーム 1,100 文書と対の負例、LLM 負例 4,000 の結合と負例なしの変種、評価セットの PERSON 表層を
  姓候補から機械除外）と `tools/make_train_notebook_v8.py`（`MASK_NER_DATA_VARIANT=noneg` で負例なしの変種を学習）
- **追加（ツール）**: 学習データ用の負例生成一式。`tools/negatives_v8.py`（生成計画 8,000 件と、Colab 出力の
  決定的選別: 地名・人名辞書、NER の兆候は review、近重複）、`tools/make_negatives_notebook.py`
  （Qwen 生成用の Colab ノートブックの生成と RUN_LLM=False 乾走）。モデルは Qwen3.8-27B-FP8、vLLM 0.27.1 のまま。
  ライブラリ本体の変更なし
- **修正**: `AffixExpander` の住所末尾拡張が電話番号を吸い込んでいた（`栄浜2-14-9　047-000-0000` の
  `　047` を部屋番号、続く `-000-0000` を番地の連鎖として 2 段で取り込む）。単位（号室・号・番地）で
  終わらない数字列の直後が ハイフン＋数字 なら取らない。保険として取り込む数字は 8 桁まで
  （`max_tail_digits`）。建物名の後の部屋番号（`スカイコート栄浜 908`、`別館　302号室`）は従来どおり取る
- **変更**: `AffixExpander` の数字付き語尾（`302号室` `3階` `５Ｆ` `1号館` `B棟`）を既定では付けない。
  文字ヘッドモデルは階・号室を自分で取り込むため、拡張は gold 外の追加（`青柳会中央病院1階`）になる
  だけだった。旧挙動は `numeric_suffix_types=("FACILITY", "ORG", "LOCATION")`
- **変更**: LOCATION に付ける語尾を地形・交通・公共機関（駅・空港・役所・庁・署・裁判所・温泉・公園…、
  `DEFAULT_LOCATION_SUFFIXES`）に限定。企業の種別語尾（支社・本社・事業所・センター）を地名に付けると
  `鳥取市支社` のように別の実体まで飲み込んでいた。ORG / FACILITY の語尾は変えていない。
  `location_suffixes` で調整可
  - 4 評価セット + 住所書式セットの実測: 文字ヘッドモデル併用は見逃・危険・1 字漏れ
    すべて不変で過剰マスク 固定OOD 436→414 / 盲検v1 818→765 / 住所書式 1,113→711 字。
    当時の本番 token モデル（30m）は 337→310 / 594→546 / 823→463 字、危険が盲検v1 62→63・住所書式 191→192、
    完全見逃しは不変。文字ヘッドモデル併用では加えて `MorphologicalRefiner(expand_affixes=False)` を推奨
    （開発記録、非公開）
- **追加**: `IdentifiabilityAnnotator` の住所成分ガード（`address_guard=True` が既定）。
  NER は「番地があれば ADDRESS、なければ LOCATION」で学習しているため（学習データ実測: ADDRESS
  4,698 件の 100% が番地付き / LOCATION 6,075 件の 99.98% が番地なし）、住所の町名部分は LOCATION
  として出力される。それが jawiki 見出し語に一致すると `LOCATION_PUBLIC` = Keep になり、
  番地だけマスクされて町名が平文で残っていた。住所文脈の正の証拠（表層の丁目・番地、隣接する
  ADDRESS エンティティ、直前直後の番地、〒・郵便番号、`住所:` 等の帳票ラベル）があり、かつ
  市区町村以上でないものを住所成分として公知判定から外す。`minimal` でも適用する。
  市区町村以上（`都道府県市区郡`）と法人格を含む組織名は例外として公知のまま残す
  （`Surrogate(kind="address", keep="city")` と同粒度）。
  4 評価セット実測（standard）: 本物の住所の漏れ 122→29 文字（token30m）/ 94→14（char30m）、
  代償は正しい Keep の 32 文字（0.55%）/ 18 文字（0.29%）。`minimal` では 511→139 / 467→118、
  代償 1,233 文字（6.8%）/ 1,036（5.5%）。`strict` の挙動は不変

- **修正**: `MorphologicalRefiner` の `drop_ascii_len` が実在の 2 文字略称（`AI` `LG` `JR` `PS`）を
  棄却していた。検出済みエンティティを消すと平文に戻るため、2 文字以上で全て大文字のものは残す。
  小文字を含む断片（`di` `Hb`）と 1 文字は従来どおり棄却。4 評価セット実測で完全見逃し -7、
  過剰マスク +25 文字（+1.2%）
- **修正**: 助詞・接尾語トリムが空白区切りの複合人名を分断していた（`かな 麻里 松本` → `麻里 松本`、
  `えりか なな` → `えりか`）。スパン内部に空白があるときは先頭/末尾トリムを行わない。
  4 評価セット実測で 1 字以上漏れ -8、過剰マスク +1 文字

## 0.8.0 - 2026-09-02

識別性ベースの匿名化（`docs/design_identifiability.md`）。既定の挙動（strict）は変わらない。

- **追加**: Annotator ステージ。`PrivacyEngine(annotators=[...])` が resolve 後・Policy 前に
  エンティティ型を書き換える。`Annotator` Protocol、不正な出力は `AnnotatorError`（原文を載せない）
- **追加**: `IdentifiabilityAnnotator(level="strict"|"standard"|"minimal", knowledge=...)`。
  公職肩書きの隣接または公知データに一致する 4 文字以上のフルネームを `PERSON_PUBLIC`、
  公知データに一致し私人に帰属しない LOCATION / FACILITY / ORG を `*_PUBLIC` にする。
  保持は正の証拠があるときだけ。裸姓・企業肩書き・4 文字以下の英字略称は公知にしない
- **追加**: `PublicKnowledge`（Wikipedia 見出し語などの 64bit ハッシュ集合、.pmk 形式、
  `PRIVACY_MASKER_KNOWLEDGE`）。`tools/build_knowledge.py` で生成（282 万エントリ、22.6MB）
- **追加**: `Policy.preset("strict" | "standard" | "minimal")`。standard / minimal は
  `*_PUBLIC → Keep`、それ以外は default でトークン化（fail-closed）。`public_types` で調整可
- **追加**: `Surrogate(kind="address", keep="city")`。都道府県・市区町村を残し、町名・番地・
  建物・部屋を偽住所に置換。分解できない住所は従来どおり全体を置換する
- **変更**: 偽住所の建物名は「原文に建物・部屋の表記（号室・階・ビル・ハイツ等）があるとき」だけ
  付与する（従来は住所の長さ > 18 文字で付与）。長い住所の出力が変わる
- **追加**: 種別語尾・法人格プレフィックスへのスパン拡張（`refiner_affix.AffixExpander`。`SpanRefiner` /
  `MorphologicalRefiner` の `expand_affixes=True` が既定）。NER が「水戸データセンター」「仙台地方裁判所」
  「株式会社琉伸メディカルサポート」「社会福祉法人ひだまり会」の固有名部分だけを検出したとき、隣接する
  センター・病院・支店・駅・株式会社・地方裁判所 等の語尾と 社会福祉法人・医療法人・特別養護老人ホーム 等の
  プレフィックスをスパンに取り込む（最大 3 段・24 文字）。ORG / FACILITY / LOCATION のみ、他エンティティの
  境界で停止、PERSON は対象外。数字を含む語尾（302号室・3階・５Ｆ・1号館・B棟）も取り込む。
  拡張のみで PII を落とさない（総当たり 46,200 件で欠け 0）。
  出力が変わる: 「くらしさん株式会社」「トヨタ自動車株式会社」のように法人格まで 1 スパンになる
- **変更**: `IdentifiabilityAnnotator` の公知判定は法人格（株式会社・社会福祉法人 等）を外した表層でも
  見出し語を引く（語尾拡張で「トヨタ自動車株式会社」になっても公知と判定できる）
- **修正（PII 漏れ）**: 分布内評価（学習データの検証分割）で見つかった Refiner の 3 種の欠けを修正。
  (1) ひらがな名の末尾 1 文字（石川まり▮の・あさひ▮な・佐藤まり▮も・矢野▮ねねか・金子▮かの）を Janome が
  助詞・助動詞と解析して削っていた → スパン末尾の 1 文字付属語は、直後（スパン外）が改行・句読点・敬称・肩書き・
  別の付属語・文末なら名前の一部として残す。ひらがなで終わる名前の後ろの 1 文字付属語は文脈に関わらず残す。
  スパン内で連鎖する助詞（日本などでは）と複数文字の機能語（など・まで・です）は従来どおり削る。
  (2) ひらがなだけの名前の先頭「は」（はすみ→すみ）を係助詞として削っていた → ひらがなだけのスパンと、直前が助詞の
  ときは先頭の付属語を削らない。(3) 単字姓（平が担当）を Janome が「平が」の 1 形態素にするため 1 文字断片として
  棄却していた → 形態素の残りがひらがなだけなら原スパンを残す。スナップは付属語・記号の形態素へ拡張しない。
  `SpanRefiner` にも同じ文脈ガードを追加。総当たり検査に該当の名前・文脈を追加（46,200 → 74,520 件。残る欠け 4 件は「金子かの産業に発注」のように漢字姓 + 助詞と解析されるひらがな名の直後に内容語が続く人工的な文脈で、意味なしには区別できないため許容）
- **変更**: `IdentifiabilityAnnotator` は、非公知の場所エンティティに隣接（間 1 文字以内）する公知の場所エンティティの公知を
  取り消す。NER が「株式会社アオバ情報システム」を 株式会社アオバ | 情報システム に分割したとき、一般名詞の断片が
  見出し語に一致して残るのを防ぐ（fail-closed）
- **追加**: `tools/eval_all.py`（総当たり・固定 OOD・分布内 3 ソース・外部 2 セットを 1 コマンドで、`--extra name=path` で盲検セット追加）、
  `tools/eval_in_distribution.py`、`tools/eval_external.py`、盲検評価セット作成ツール `tools/blind_set_prepare.py` /
  `tools/blind_set_finalize.py`（タグ付きテキストをレビューして JSONL 化・SHA 凍結）、手順 `docs/blind_eval_protocol.md`
- **修正**: 全角の入力が型ごとにバラバラだった問題。検出前に「長さを変えない全角→半角の写像」（`narrow()`）をかける方式に変更し、PHONE / POSTAL / CREDIT_CARD / MY_NUMBER / CORPORATE_NUMBER / PASSPORT / RESIDENCE_CARD / BANK_ACCOUNT / PENSION_NUMBER / IP / EMAIL / URL / DATE / AGE の全型が全角に対応（`平成２３年３月１１日`・`０４８－２２２－７８１５`・`ＴＫ１２３４５６７` 等）。offset は原文をそのまま指す
- **修正**: `SpanRefiner` / `MorphologicalRefiner` の「直後が肩書きなら末尾の助詞を残す」判定が型に関係なく適用され、`医療法人伊織会の理事長` の「の」が ORG スパンに残っていた。この曖昧性は PERSON 固有（ひらがな終わりの名前）で、品詞では判別できない（`佐藤まりも氏` の「も」と `田中は先生` の「は」は同じ 助詞,係助詞）ため、判定を `honorific_types`（既定 PERSON）に限定した。過剰マスクが 1〜5 字減り、漏れ・危険・coverage は全評価セットで不変
- **追加**: `MorphologicalRefiner(backend="auto"|"janome"|"fugashi")`。fugashi + IPADIC（extra `[mecab]`）があれば自動で使い、Janome と同じ品詞体系で 10 倍以上速い。`TransformersNerDetector(dtype=, model_kwargs=)` で bf16 や `attn_implementation`（sdpa / flash_attention_2）を指定可能。README に GPU・attention・精度型の選び方を追加
- **追加**: 正規表現カタログを拡張。法人番号（検査用数字）・旅券番号・在留カード番号、文脈必須の運転免許証番号・基礎年金番号・口座番号、URL・IPv6・MAC アドレス。`default_ja(dates=True)` で DATE（和暦・西暦、生年月日文脈で 1.0）と AGE。`RegexPattern` に `score` / `context` / `context_score` / `require_context`
- **追加**: `ContextEnhancer`（文脈語でスコアを上げる。`PrivacyEngine(enhancer=)`）、`engine.explain()`、`engine.restore_batch()`、`Mask` の別名引数 `chars_to_mask` / `masking_char` / `from_end`
- **追加**: オペレータ `DateShift`（日付を一定日数ずらす。和暦・西暦・スラッシュ形式を保持）、`Bucketize`（数値の区間化）、`FormatPreserving`（桁数・文字種を保った仮名。非可逆）
- **追加**: `privacy_masker.structured.transform_records / transform_dataframe`（列型の自動推定つき）、`privacy_masker.evaluation.evaluate`（gold 付き文書の coverage / 危険 / 過剰マスク）、`privacy_masker.config.load_engine`（YAML / JSON / dict から組み立て）
- **追加**: 他ライブラリに倣った API。呼び出し時オプション `entities=`・`score_threshold=`（型別 override）・`allow_list=`（exact / regex）・`deny_list=`（型指定）・`explain=`（判定の経緯を `TransformedEntity.explanation` に。原文は含まない）。`PrivacyEngine(...)` の同名引数で既定値を設定可
- **追加**: オペレータ `Hash(algorithm, salt, length)`、`Encrypt(key)` と `decrypt_text(text, key)`（本文中の Fernet 暗号文を復号）、`Custom(fn)`。README に重複解決の規則を追加
- **追加**: `HandleDetector`（チャット・ログ・SNS のハンドル名を `USERNAME` 型で検出: `@handle`、行頭の話者ラベル、`user=` 等のキー値。ヘッダ語・1 文字ラベルは除外）。盲検評価 v1 で PERSON の危険見逃し 107 件のうち約 20 件がこの表層だった
- **変更**: `RegexDetector.default_ja()` の取りこぼしを修正: 全角ハイフン、空白・「の」区切りの電話（桁数 10〜11 のみ）、国際番号（+81 / +44 / +886）、文末ピリオド直前の IPv4、4 桁区切りの個人番号（16 桁カード番号の一部は取らない）
- **変更**: `AffixExpander` が ADDRESS / LOCATION の末尾に直結する番地・号室（東原|5685、栄浜|3-4）と、ADDRESS の建物名の後の空白＋部屋番号（スカイコート栄浜| 908）を取り込む。年月日・金額・数量の単位が続くものは取らない
- **追加**: 盲検評価セット v1（246 文書、Web 実文書 96 + 執筆 150。構成は開発記録、非公開）と採取ツール `tools/blind_set_collect_web.py`。`tools/blind_set_finalize.py` に `--tagged`・lint・CRLF 正規化を追加
- **変更**: ツールのデータフォルダのパスを環境変数 `PRIVACY_MASKER_DRIVE_DIR`（`tools/_paths.py`）で差し替え可能に。`requirements-dev.txt`
- **推奨モデル**: `examples/try_privacy_masker.py` の既定を学習データを更新して再学習した **30m モデル**、`--quality` を同じデータの **70m モデル** に変更。学習データは Wikipedia 由来の文書と合成データの数世代分。盲検評価セット v1（246 文書・55 ジャンル）で PERSON 危険 27→19 / 803、ADDRESS 危険 5→2 / 71、OpenPII-JA で PERSON 33→9・ADDRESS 138→88。直接識別子は全評価セットで悪化なし。固定 OOD の危険 11→14 は公知エンティティ（空母打撃群・リュウグウ・中東 等）のみで、PERSON/ADDRESS は不変
- 旧: 前の本番モデル（30m / `--quality` 70m）は
  11 ジャンル OOD 327 文書（語尾拡張 Refiner 込み）で cov 98.0%・危険 11・過剰マスク 268 字。その前の既定（30m）は
  97.9%・17・558 字で、PERSON/ADDRESS の危険数は同じ（全指標で同等以上、推論速度は同じ）
- **追加**: `examples/try_privacy_masker.py --level standard|minimal [--knowledge PATH]`、
  `tools/eval_identifiability.py`（安全コスト測定）
- 実測（30m モデル）: 実在ニュース 178 文書の過剰マスク 77 → 14 文字。私人文書 431 文書・私人名
  743 件のうち standard で保持されるのは 18 件（「議員」肩書き・著名人との同姓同名・名前の分裂）
- 既知の限界: 著名人と同姓同名の私人は standard で保持され得る。次版で Wikidata の知名度
  （sitelinks）による人物リストと、肩書きを巻き込んだ PERSON スパン（「山本部長」）の Refiner 対応を予定
- `PrivacyEngine.transform_batch()` が、batch detectorの返却件数と入力件数の不一致を
  `DetectorError` として扱うようにした。`on_detector_error="skip"` では該当チャンク全件を
  未変換のままロールバックし、件数不足による文書と検出結果の取り違えを防ぐ
- `TransformersNerDetector` がHugging Face token-classification pipelineの実トークン長を
  推論前に確認するようにした。上限超過時はfast tokenizer・aggregation・整合した
  tokenizer/model上限が揃う場合だけHF標準strideで全文を処理し、安全に分割できない場合は
  `ConfigurationError` で停止する（任意のカスタムcallableの従来契約は維持）

## 0.7.0 - 2026-08-25

`MorphologicalRefiner` のPII漏れ修正が主眼のリリース。いずれも v0.3.2 以降存在していた
既存バグで、多視点レビューと総当たり検査（名前3,080種×文脈15種=46,200通り）で発見した。

- **修正1**: 付属語トリムを「表層がひらがな＋記号のみ」に限定。品詞が助詞でも表層が
  漢字・カタカナなら削らない。従来は「有之」→「有」、「勝之進」→「勝」（之=助詞,格助詞）、
  「ブルーノ」→「ブルー」、「ミサヲ」→「ミサ」（ノ・ヲ=助詞）とPIIが欠けていた。
  敬称（様・氏・殿）は別経路なので従来どおり削れる
- **修正2**: スパン全体が付属語と解析された場合に棄却せず原スパンを残すようにした
  （1文字以下の検出ノイズのみ棄却）。従来は「チャン」「かな」「りか」「よね」等の実在の
  かな・カナ人名が**検出済みでもマスクされず平文で残っていた**（実モデルで実証。
  既定の `SpanRefiner` では正しくマスクされるため、形態素refinerを使うと危険側に悪化していた）
- **修正3**: 単一形態素内部の断片棄却を1文字のみに限定。従来は「見越さ」内の「見越」の
  ように実名がスパンごと棄却されていた（「拡張」の拡 は引き続き棄却）
- **撤回**: 自立語（動詞句・接続詞）のトリムは実装を試みたが撤回した。「松本駅から歩いて」
  →「松本駅」は実現できるが、IPADICが 之/ノ/が を格助詞と解析するため
  「勝之進」→「勝」「元湯ノ出」→「元湯」「浅が野」→「野」とPIIが分断される漏れが実測された。
  過剰マスク数文字の改善に対し代償が大きいため実装しない
- 検証: 46,200通りの総当たりでPII欠け0件。11ジャンル878エンティティでcoverage無劣化
  （過剰マスクは550→555字と微増。安全側への振り直しの結果）

## 0.6.0 - 2026-08-21

- `MorphologicalRefiner` に語境界スナップを追加（**既定有効・破壊的変更**: 既存の
  MorphologicalRefinerユーザーは出力が変わる。旧挙動に戻すには
  `MorphologicalRefiner(snap_boundaries=False, drop_ascii_len=0)`）:
  スパン端を形態素境界へ拡張、スナップで重なった同typeスパンを統合、
  単一形態素の内部断片（非名詞系のみ。名詞,一般・固有名詞は拡張して保護）と
  英字1〜2文字の断片を誤検出として棄却。MeCab語単位予測と同等の境界品質を後処理で実現
- **トレードオフ（漏れ方向）**: 英字2文字以下の単独スパン（JR・au等の実在組織名や
  裸のイニシャル）は既定で棄却され平文のまま残る。該当データを扱う場合は
  `drop_ascii_len=0` にすること。敬称付きイニシャル（Kさん・TK君）は棄却されない
- `TransformersNerDetector` に `min_score`（信頼度閾値、既定0.0=無効）を追加
- お試しスクリプトがjanome導入時に `MorphologicalRefiner` を自動使用

## 0.5.1 - 2026-08-20

- Surrogate人名: 漢字+カタカナ混在名（例: 漢字姓+カタカナ名）を漢字名と同じ扱いに変更
  （従来はカタカナ優勢だとカタカナプールに落ちていた）。姓リンクも有効になる。
  カタカナプールは漢字を1文字も含まない名前のみに適用
- 混在名のhmac出力が変わる（漢字のみ・カタカナのみの名前の出力は不変）

## 0.5.0 - 2026-08-15

- バッチ推論: `detect`/`detect_batch` が全チャンクを1回のpipeline呼び出しに集約
  （長さ順ソートでパディング削減）。end-to-end実測でCPU約1.9倍・MPS約3.8倍、
  検出結果は逐次処理と完全一致
- `TransformersNerDetector` に `device` / `pipeline_batching` を追加、
  `batch_size` 既定を8→16に変更（従来は未使用だった）
- str専用のカスタムpipeline callableは自動判定で逐次経路にフォールバック

## 0.4.0 - 2026-08-15

- Surrogateに姓リンクを追加（`link_surnames=True` 既定）: フルネームと姓のみ言及が
  同一人物として連携し、同じ原姓は偽名でも姓を共有する
- **破壊的変更**: 人名のhmac出力が変わる（偽姓の導出が姓部分のみのHMAC決定値に）
- hmac+姓リンクの相関特性（同鍵・同scopeで同姓が同偽姓に収束）をドキュメントに明示

## 0.3.3 - 2026-08-13

- `MorphologicalRefiner` に誤解析耐性（付属語run+自立語アンカー）と
  Janomeユーザー辞書サポート（`user_dict`）を追加

## 0.3.2 - 2026-08-13

- `MorphologicalRefiner` 追加（`[janome]` extra、品詞ベースのスパン端トリム）

## 0.3.1 - 2026-08-13

- `SpanRefiner` に複数文字機能語トリムを追加（において・から・まで等）

## 0.3.0 - 2026-08-13

- `Surrogate` operator追加: 実在風の日本人名・企業名・住所への仮名化
  （kind=person/org/address、strategy=hmac/random、復元非対応）

## 0.2.0 - 2026-08-12

- `SpanRefiner` 追加（スパン端の空白・敬称・助詞トリム、既定有効）

## 0.1.0 - 2026-08-10

- 初版: PrivacyEngine / Detector / EntityResolver / Policy / Operator パイプライン、
  Tokenize（sequential/hmac/vault）、Mask/Redact/Replace/Keep、
  MemoryStore/SQLiteStore、Fernet CryptoProvider、scope一貫性、restore
