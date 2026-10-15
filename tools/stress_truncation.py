"""PII欠け（truncation）の総当たり検査。
Refinerは「スパンを外側に広げる」のは安全、「PIIの文字を落とす」のは事故。
名前×区切り×文脈の総当たりで、goldが1文字でも欠けるケースを列挙する。"""
import itertools, sys
sys.path.insert(0, ".")
from privacy_masker.data.ja_pools import SURNAMES_KANJI, GIVEN_KANJI, ORG_CORES, TOWNS
from privacy_masker.refiner_janome import MorphologicalRefiner
from privacy_masker.types import Entity

# IPADICで接続詞・副詞・動詞等に解析されやすい漢字を含む人名用字（レビュー指摘由来）
TRICKY_SEI = ["尚", "又", "将", "唯", "只", "猶", "彌", "宜", "抑", "或", "豈", "屡",
              "早", "極", "然", "略", "粗", "転", "露", "愈", "否", "即", "但", "亦",
              "見", "与", "行", "有", "生", "出", "上", "及", "相", "走", "泳", "立"]
TRICKY_MEI = ["也", "平", "吾", "大", "樹", "之", "太", "美", "子", "人", "哉", "志",
              "登", "江", "香", "実", "誠", "泳", "走", "越", "見", "木", "川", "山"]
SEI = list(SURNAMES_KANJI[:30]) + ["鮫島", "久遠寺", "小鳥遊", "みずき", "あべ", "おおた",
                                    "あおき", "たかなし", "イザベラ", "ミゲル"]
MEI = list(GIVEN_KANJI[:20]) + ["フェレイラ", "サントス", "ゆうと"]
SEPS = ["", " ", "　", "・", "の"]
CTX = [("", "さんに連絡"), ("", "様よりご連絡"), ("", "を訪問しました。"), ("", "に変更します。"),
       ("", "が担当します。"), ("", "は本日休みです。"), ("", "と契約しました"), ("", "へ連絡"),
       ("担当は", "です。"), ("", "から歩いて10分"), ("", "など"), ("", "、以上"),
       ("追って", "へ連絡"), ("", "駅で待ち合わせ"), ("", "産業に発注"),
       # 分布内評価で欠けが出た文脈: 改行・帳票・敬称・読点・文末・助詞連続
       ("請求者 ", "\n理由 医療"), ("記入 ", ""), ("後任は", "氏です。"), ("研究分担者 ", "、殿崎麻椰"),
       ("掲示を", "が設置"), ("", "さん、今日の"), ("担当は", "です。"), ("名前は", "さん。"), ("", "")]

ref = MorphologicalRefiner()
names = set()
# 1) トリッキー漢字の組み合わせ
for a, b in itertools.product(TRICKY_SEI, TRICKY_MEI):
    names.add(a + b)
# 2) プール姓 × 区切り × プール名
for a, sep, b in itertools.product(SEI, SEPS, MEI[:8]):
    names.add(a + sep + b)
# 2.5) レビューで実証された漏れの表層（漢字/カタカナ助詞・かな人名・表記ゆれ）
KANA_NAMES = ["チャン", "グエン", "ブルーノ", "マリノ", "デボラ", "ミサヲ", "シャラポワ",
              "アドリアナ", "デクラーク", "かな", "りか", "よね", "さえ", "やす", "たか",
              "はな", "かよ", "さよ", "たき", "ともよ", "ますよ", "やすよ", "もん"]
VARIANTS = []
for mid in ["之", "ノ", "の", "が", "ヶ", "ケ"]:
    for a in ["勝", "見", "元湯", "浅", "上", "田", "森", "山", "宮", "少", "牛"]:
        for b in ["進", "越", "出", "野", "見", "広", "増", "田"]:
            VARIANTS.append(a + mid + b)
names.update(KANA_NAMES)
names.update(VARIANTS)
# 2.6) 分布内評価（2026-09-03）で実際に欠けた型: ひらがな名の端が助詞・助動詞と解析されるもの、単字姓
EDGE_NAMES = ["あさひな", "はすみ", "石川まりの", "佐藤まりも", "矢野ねねか", "大野なつか", "篠原秋も", "金子かの",
              "村瀬里な", "のぞみ", "はな", "もえ", "とも", "はるか", "にな", "かのん", "ここな", "ゆかりん",
              "平", "林", "森", "東", "原", "南", "辻", "関"]
names.update(EDGE_NAMES)
# 3) 組織名・地名
for c in ORG_CORES:
    names.update([c + "産業", c + "製作所", c + "信用金庫"])
for t_ in TOWNS[:20]:
    names.add(t_)

leaks = []
n = 0
for name in sorted(names):
    for pre, post in CTX:
        text = pre + name + post
        s0 = len(pre)
        n += 1
        out = ref.refine(text, [Entity("PERSON", s0, s0 + len(name), 0.99, "t")])
        if not out:
            leaks.append((name, text, "スパン消失"))
            continue
        got = text[out[0].start:out[0].end]
        if name not in got:
            leaks.append((name, text, got))

print(f"検査 {n}件（名前 {len(names)}種 × 文脈 {len(CTX)}種）")
print(f"PII欠け: {len(leaks)}件")
seen = set()
for name, text, got in leaks:
    if name in seen:
        continue
    seen.add(name)
    print(f"  {name!r:<16} in {text!r:<28} -> {got!r}")
    if len(seen) >= 40:
        print("  …（以降は同種のため省略）")
        break
