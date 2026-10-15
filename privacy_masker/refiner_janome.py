"""形態素解析（Janome）による境界リファインメント（`[janome]` extra）。

ルール式の `SpanRefiner` の上位互換。品詞情報を使ってスパン端の付属語
（助詞・助動詞・記号・人名接尾）を削るため、ルール式では判別できなかった
「ひらがな終わりの固有名詞 + 助詞」（みなと+まで）や「ひらがな連鎖の機能語」
（などでは）も正しく補正できる。

- モデルの再学習・学習/推論の分割器一致は不要（検出後の後処理のみ）
- Janomeは純Pythonでネイティブ依存なし（`pip install 'privacy-masker[janome]'`）
- v0.7の安全強化: 付属語トリムは「表層がひらがな＋記号のみ」に限定した
  （漢字の助詞「有之」の之、カタカナの助詞「ブルーノ」のノ でPIIが欠けていた）。
  また全体が付属語と解析されたスパンは棄却せず残す（実在のかな人名「かな」「チャン」等が
  まるごとマスクされない漏れを防ぐ。1文字以下の検出ノイズのみ棄却）。
  なお自立語（動詞句）のトリムは実装を試みたが、之/ノ/が 等の表記ゆれでPIIが
  分断される漏れが実測されたため撤回した（「松本駅から歩いて」は縮められない）
- 語境界スナップ（v0.6、既定有効）: スパン端が形態素の途中に落ちる場合は境界まで
  拡張し（鮫|島病院→鮫島病院、青葉電力送→青葉電力送配）、スナップで重なった同type
  スパンは統合する。単一形態素の内部断片（拡張の「拡」）は固有名詞でなければ誤検出
  として棄却。英字1〜2文字だけのスパン（S・NS・di等のカルテ記号・略記）も棄却
  （`snap_boundaries=False` / `drop_ascii_len=0` で無効化）。MeCab語単位で予測する
  旧来手法と同等の境界品質を後処理で得るのが狙い
- 削るのは付属語（非PII）のみ。スパン端が形態素の途中に落ちる場合は削らない。
  さらに、削った後に残る端がひらがなの場合は、その形態素が固有名詞と
  解析されたときだけ削る（辞書に無いひらがな名の誤解析による欠落を防ぐ）
- ドメイン固有名詞（社内の人名・製品名等）は `user_dict`（Janomeユーザー辞書、
  MeCab IPADIC形式CSV）に登録すると境界判定の確度が上がる
- 使い方: `PrivacyEngine(refiner=MorphologicalRefiner())`
- バックエンド（v0.8）: `backend="auto"`（既定）は fugashi（MeCab の C 実装、`pip install fugashi ipadic`）が入っていれば
  それを、無ければ Janome（純 Python）を使う。品詞体系は両方 IPADIC で同じ。fugashi は Janome の 10〜30 倍速い。
  `backend="janome"` / `"fugashi"` で固定できる。`user_dict` は Janome 専用（fugashi では ConfigurationError）
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Sequence

from .types import Entity

# スパン端から削ってよい付属語の品詞プレフィックス（全type）
_TRIM_POS = ("助詞", "助動詞", "記号")
_HIRAGANA_ONLY_RE = re.compile(r"^[ぁ-ゟー]+$")
# スパンの内部（両端を除いた部分）に空白があるか。複数語の名前の判定に使う
_INNER_SPACE_RE = re.compile(r"\S[ \u3000]\S")


def _is_hira(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ" or ch == "ー"
_DEFAULT_HONORIFICS_JA = ("さん", "さま", "様", "氏", "くん", "ちゃん", "君", "殿")
# 名前の直後に来る肩書き・呼称・場所語。「田中は先生」のように助詞の直後には来ないので、
# これらが続くなら削ろうとしている末尾の 1 文字は名前の一部と判断する
_TITLES_AFTER_NAME = (
    "先生", "部長", "課長", "係長", "主任", "社長", "会長", "専務", "常務", "取締役", "教授", "医師", "看護師",
    "選手", "監督", "議員", "大臣", "首相", "知事", "市長", "院長", "所長", "店長", "局長", "室長", "班長",
    "主事", "主査", "参事", "理事", "顧問", "弁護士", "税理士", "宅", "邸", "家", "駅", "夫人", "夫妻", "親子",
)
# 削ってよい付属語の「表層」の条件（v0.7）。品詞が助詞・助動詞・記号でも、
# 表層に漢字やカタカナが含まれる場合は名前の一部である可能性が高いので削らない。
# 実測の漏れ: 有之→有（之=助詞,格助詞）、ブルーノ→ブルー（ノ=助詞）、ミサヲ→ミサ（ヲ=助詞）、
# 勝之進→勝。ひらがなと記号・空白のみで構成される表層（の・は・から・。・、・空白）だけを削る。
# 敬称（様・氏・殿）は漢字だが _HONORIFIC_POS の別経路なので従来どおり削れる。
_TRIMMABLE_SURFACE_RE = re.compile(
    r"^[ぁ-ゟ\s　。、．，・「」（）()\[\]:：;；!！?？~〜\-ー]+$")
# PERSONの末尾からのみ削る接尾語の品詞プレフィックス（さん・様・氏 等）
_HONORIFIC_POS = "名詞,接尾,人名"


def _fugashi_available() -> bool:
    try:
        import fugashi  # noqa: F401
        import ipadic  # noqa: F401
        return True
    except ImportError:
        return False


def _require_fugashi():
    """fugashi + IPADIC 辞書の Tagger。品詞体系を Janome と揃えるため unidic ではなく ipadic を使う。"""
    try:
        import fugashi
        import ipadic
    except ImportError as e:  # pragma: no cover
        raise ImportError("backend='fugashi' requires: pip install fugashi ipadic") from e
    return fugashi.GenericTagger(ipadic.MECAB_ARGS)


def _require_janome():
    try:
        from janome.tokenizer import Tokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MorphologicalRefiner requires the 'janome' package. "
            "Install with: pip install 'privacy-masker[janome]'"
        ) from e
    return Tokenizer


class MorphologicalRefiner:
    def __init__(
        self,
        honorific_types: Sequence[str] = ("PERSON",),
        user_dict: str | None = None,
        user_dict_type: str = "ipadic",
        user_dict_enc: str = "utf8",
        snap_boundaries: bool = True,
        drop_ascii_len: int = 2,
        expand_affixes: bool = True,
        affix_expander: "AffixExpander | None" = None,
        backend: str = "auto",
    ) -> None:
        from .refiner_affix import AffixExpander
        from .exceptions import ConfigurationError
        # 種別語尾・法人格プレフィックスへの拡張（v0.8）。スナップ・トリム・統合の後に適用
        self.expand_affixes = expand_affixes
        self._affix_expander = affix_expander if affix_expander is not None else AffixExpander()
        if backend not in ("auto", "janome", "fugashi"):
            raise ConfigurationError("backend must be 'auto', 'janome' or 'fugashi'")
        if backend == "auto":
            backend = "fugashi" if (user_dict is None and _fugashi_available()) else "janome"
        self.backend = backend
        if backend == "fugashi":
            if user_dict is not None:
                raise ConfigurationError("user_dict is supported only with backend='janome'")
            self._tagger = _require_fugashi()
            self._tokenizer = None
        else:
            Tokenizer = _require_janome()
            self._tagger = None
            if user_dict is not None:
                self._tokenizer = Tokenizer(udic=user_dict, udic_type=user_dict_type,
                                            udic_enc=user_dict_enc)
            else:
                self._tokenizer = Tokenizer()
        self.honorific_types = frozenset(honorific_types)
        self.snap_boundaries = snap_boundaries
        self.drop_ascii_len = drop_ascii_len

    def _tokens(self, text: str):
        """(surface, part_of_speech) の列。品詞は IPADIC 形式「名詞,固有名詞,人名,姓」に揃える。"""
        if self._tagger is not None:
            for w in self._tagger(text):
                yield w.surface, ",".join(w.feature[:4])
        else:
            for tok in self._tokenizer.tokenize(text):
                yield tok.surface, tok.part_of_speech

    def _morphemes(self, text: str) -> list[tuple[int, int, str]] | None:
        """(start, end, part_of_speech) のリスト。復元不一致時はNone（no-op化）。"""
        out = []
        pos_cursor = 0
        for surface, pos in self._tokens(text):
            start = text.find(surface, pos_cursor)
            if start < 0:
                return None  # 表層が復元できない（防御的フォールバック）
            out.append((start, start + len(surface), pos))
            pos_cursor = start + len(surface)
        return out

    def _snap_one(self, text, morphemes, char_to_m, ent: Entity) -> Entity | None:
        """スパン端を形態素境界へスナップする（v0.6）。

        - 端が形態素の途中に落ちる場合は境界まで拡張（語単位予測の旧来手法と同等の効果）
        - スパン全体が単一形態素の内部断片の場合: その形態素が固有名詞なら形態素全体へ
          拡張、それ以外（拡張の「拡」等）は誤検出として棄却
        """
        ms = char_to_m.get(ent.start)
        me = char_to_m.get(ent.end - 1)
        if ms is None or me is None:
            return ent  # 形態素で覆われない位置（防御的にそのまま）
        s_new = morphemes[ms][0]
        e_new = morphemes[me][1]
        if ms != me:
            # 端が付属語・記号の形態素の途中なら、その側は拡張しない（付属語を名前に取り込まない）
            if e_new > ent.end and morphemes[me][2].startswith(_TRIM_POS):
                e_new = ent.end
            if s_new < ent.start and morphemes[ms][2].startswith(_TRIM_POS):
                s_new = ent.start
        if (s_new, e_new) == (ent.start, ent.end):
            return ent
        if ms == me and (ent.start > s_new or ent.end < e_new):
            # 単一形態素の内部断片: 名詞,固有名詞・名詞,一般（IPADIC非収載の実名が
            # ここに落ちる: 東雲・琥珀等）は境界まで拡張して守る。それ以外
            # （サ変接続の「拡張」、動詞・助詞等）は誤検出として棄却
            pos = morphemes[ms][2]
            if not (pos.startswith("名詞,固有名詞") or pos.startswith("名詞,一般")):
                if ent.end - ent.start <= 1:
                    # 1文字の断片は誤検出として棄却（「拡張」の拡 等）。ただし形態素の残りが
                    # ひらがなだけ（「平が」「森へ」= 単字姓 + 助詞の誤結合）なら姓を守るため原スパンを残す
                    rest = text[s_new:ent.start] + text[ent.end:e_new]
                    if rest and _HIRAGANA_ONLY_RE.fullmatch(rest):
                        return ent
                    return None
                # 2文字以上は実在名の可能性が高い（「見越さ」内部の見越 等）ので
                # 棄却せず形態素境界まで広げて残す（安全側）
        return dataclasses.replace(ent, start=s_new, end=e_new)

    _ASCII_RE = re.compile(r"[A-Za-z]+")

    @staticmethod
    def _looks_like_acronym(surf: str) -> bool:
        """2 文字以下の英字でも実在の固有名詞なら捨てない。

        drop_ascii_len は「S・NS・di 等の検出ノイズ」を落とす目的だが、実測で
        ``AI``(PERSON) と ``LG``(ORG) を丸ごと捨てていた（被覆 1.00 → 0.00）。
        モデルが正しく検出したものを Refiner が消すので、そのまま平文で漏れる。

        全部大文字（AI・LG・NHK）と数字混じり（3M・au）は略称・社名として実在するので残す。
        小文字を含む断片（di・ns・th）は従来どおり落とす。全部大文字でも 1 文字（S・N）は
        イニシャルの誤検出が多いので落とす側に残す。
        """
        if len(surf) < 2:
            return False
        return surf.isupper()

    def refine(self, text: str, entities: Iterable[Entity]) -> list[Entity]:
        entities = list(entities)
        if not entities:
            return entities
        morphemes = self._morphemes(text)
        if morphemes is None:
            return entities
        by_end = {m[1]: m for m in morphemes}
        by_start = {m[0]: m for m in morphemes}

        def _is_trimmable(m, entity_type):
            if m[2].startswith(_TRIM_POS):
                # 品詞だけでなく表層も見る（漢字・カタカナの助詞は名前の一部の可能性）
                return bool(_TRIMMABLE_SURFACE_RE.match(text[m[0]:m[1]]))
            return (entity_type in self.honorific_types
                    and m[2].startswith(_HONORIFIC_POS))

        def _tail_context_blocks_trim(pos_after, entity_type):
            # 付属語の直後に来るものを見る。文末・空白・句読点・記号・接尾・別の助詞/助動詞なら、
            # 文法上そこに助詞は来ないので、削ろうとしている付属語は名前の一部と判断する。
            # 「直後が敬称・肩書きだから残す」は PERSON 固有の曖昧性（佐藤まりも氏 の も）なので
            # honorific_types に限る。品詞では判別できない（田中は先生 の は も 助詞,係助詞）ため、
            # 型が唯一の一般的な手掛かりになる。組織・施設・地名の名称は裸の助詞で終わらない
            if pos_after >= len(text):
                return True
            nxt = by_start.get(pos_after)
            if nxt is None:
                return text[pos_after].isspace()
            if nxt[2].startswith(("記号", "助詞", "助動詞")) or nxt[2].startswith("名詞,接尾"):
                return True
            if entity_type not in self.honorific_types:
                return False
            return (text[nxt[0]:nxt[1]] in _DEFAULT_HONORIFICS_JA
                    or text.startswith(_TITLES_AFTER_NAME, pos_after))

        def _anchor_ok(anchor_char, anchor_morpheme):
            # 残る端が非ひらがなならOK。ひらがなの場合は、その形態素が
            # 固有名詞と確信できるときだけOK（誤解析による名前の欠落防止）
            if not ("ぁ" <= anchor_char <= "ゟ"):
                return True
            return anchor_morpheme is not None and anchor_morpheme[2].startswith("名詞,固有名詞")

        char_to_m = {}
        for mi, (ms, me, _) in enumerate(morphemes):
            for pos in range(ms, me):
                char_to_m[pos] = mi

        out: list[Entity] = []
        for ent in entities:
            surf = text[ent.start:ent.end].strip()
            if (0 < self.drop_ascii_len and surf and self._ASCII_RE.fullmatch(surf)
                    and len(surf) <= self.drop_ascii_len
                    and not self._looks_like_acronym(surf)):
                continue  # S・NS・di等: 英字1〜2文字の断片は拡張前に棄却（空白は無視）
            orig = ent
            expanded = False
            if self.snap_boundaries:
                snapped = self._snap_one(text, morphemes, char_to_m, ent)
                if snapped is None:
                    continue
                expanded = snapped is not ent
                ent = snapped
            s, e = ent.start, ent.end
            while s < e and text[s].isspace():
                s += 1
            while s < e and text[e - 1].isspace():
                e -= 1
            # スパンの内部に空白がある = 複数語の名前（「かな 麻里 松本」「帯広市 南の森西」）。
            # 付属語トリムは「田中は」のような 1 語 + 助詞を想定した規則で、複数語名では
            # ひらがなの名前要素を助詞と誤認して削る。実測で OpenPII の PERSON 7 件・
            # ADDRESS 5 件が壊れていた（かな 麻里 松本 → 麻里 松本 / えりか なな → えりか /
            # もとのぶ ひろみち 横山 → のぶ ひろみち）。既存のガードは「スパン全体がひらがな」の
            # 場合しか効かず、漢字・欧文混在では素通りする。
            # 語の区切りが明示されている以上、端の 1 語を削るのは常に誤りなのでトリムを止める。
            if _INNER_SPACE_RE.search(text[s:e]):
                out.append(ent if (s, e) == (ent.start, ent.end)
                           else dataclasses.replace(ent, start=s, end=e))
                continue
            # 末尾: 付属語の連続runをまとめて評価（「などでは」の連鎖に対応）
            core_start = e
            m = by_end.get(core_start)
            run_all_single_particles = True
            while m and m[0] >= s and _is_trimmable(m, ent.entity_type):
                is_particle = m[2].startswith(("助詞", "助動詞"))
                # スパン末尾の 1 文字助詞・助動詞（な・の・も・か…）は、直後（スパン外）が改行・句読点・敬称・
                # 別の付属語・文末なら名前の一部（石川まりの⏎・あさひなが・佐藤まりも氏・金子かの、）とみなす。
                # スパン内で連鎖する助詞（日本などでは）や複数文字の機能語（など・まで・です）は従来どおり削る
                if (is_particle and m[1] == e and m[0] > s and m[1] - m[0] == 1
                        and _tail_context_blocks_trim(m[1], ent.entity_type)):
                    break
                if not (is_particle and m[1] - m[0] == 1):
                    run_all_single_particles = False
                core_start = m[0]
                m = by_end.get(core_start)
            if core_start < e:
                if core_start == s:
                    e = core_start         # 全体が付属語 → 下で棄却/復元を判定
                elif run_all_single_particles and _is_hira(text[core_start - 1]):
                    pass  # ひらがなで終わる名前 + 1文字助詞（あさひな・まりの・なつか）は区別できないので削らない（安全側）
                elif _anchor_ok(text[core_start - 1], by_end.get(core_start)):
                    e = core_start
            # 先頭: 同様にrun評価
            if s < e:
                core_end = s
                m = by_start.get(core_end)
                prev = by_end.get(s)
                head_particle_ok = not (
                    _HIRAGANA_ONLY_RE.fullmatch(text[s:e])                    # ひらがなだけの名前（はすみ・はな・のぞみ）
                    or (prev is not None and prev[2].startswith(("助詞", "助動詞")))  # 「担当ははすみ」= 助詞が連続
                )
                while m and m[1] <= e and m[2].startswith(_TRIM_POS) \
                        and _TRIMMABLE_SURFACE_RE.match(text[m[0]:m[1]]):
                    if m[2].startswith(("助詞", "助動詞")) and not head_particle_ok:
                        break
                    core_end = m[1]
                    m = by_start.get(core_end)
                if core_end > s:
                    if core_end == e:
                        s = core_end
                    elif _anchor_ok(text[core_end], by_start.get(core_end)):
                        s = core_end
            while s < e and text[s].isspace():
                s += 1
            while s < e and text[e - 1].isspace():
                e -= 1
            if s >= e:
                # 全体が付属語と解析されて空になった場合。実在のかな・カナ人名が
                # まるごと消える漏れ（チャン・かな・りか・よね等。実モデルで実証）を
                # 防ぐため、1文字以下の検出ノイズだけ棄却し、それ以外は原スパンを残す
                if len(text[orig.start:orig.end].strip()) <= 1:
                    continue
                out.append(orig)
                continue
            if s < e:
                # 注意: 英字断片の棄却は拡張前の原スパンにのみ適用する。トリム後の
                # 再判定は行わない（「Kさん」の敬称トリムで残るイニシャルKを
                # 捨てるとPIIが平文で漏れるため。v0.6レビューで実証されたCritical）
                out.append(ent if (s, e) == (ent.start, ent.end)
                           else dataclasses.replace(ent, start=s, end=e))
        if self.snap_boundaries:
            out = self._merge_same_type_overlaps(out)
        if self.expand_affixes:
            out = self._affix_expander.expand(text, out)
        return out

    @staticmethod
    def _merge_same_type_overlaps(entities: list[Entity]) -> list[Entity]:
        """スナップで重なった同typeスパンを統合する（鮫島∪鮫島病院→鮫島病院）。
        型が異なる重複はEntityResolverの責務なのでここでは触らない。"""
        merged: list[Entity] = []
        for ent in sorted(entities, key=lambda x: (x.entity_type, x.start, x.end)):
            last = merged[-1] if merged else None
            if (last is not None and last.entity_type == ent.entity_type
                    and ent.start < last.end):
                merged[-1] = dataclasses.replace(
                    last, end=max(last.end, ent.end), score=max(last.score, ent.score))
                continue
            merged.append(ent)
        merged.sort(key=lambda x: (x.start, x.end))
        return merged
