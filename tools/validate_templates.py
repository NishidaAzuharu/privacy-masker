"""LLM生成テンプレートの機械検証。
LLM任せにできない部分（スロット構文の正しさ・実名混入・重複）を決定的に検査する。
使い方: python validate_templates.py <workflow_output.json> [...]
"""
import json, os, re, sys
from collections import Counter

sys.path.insert(0, ".")
from privacy_masker.data.ja_pools import (SURNAMES_KANJI, GIVEN_KANJI, ORG_CORES,
                                          PREF_CITIES, TOWNS)

ALLOWED = ("PERSON", "KPERSON", "HPERSON", "ORG", "FACILITY", "ADDRESS",
           "LOCATION", "PHONE", "EMAIL")
SLOT_RE = re.compile(r"\{([A-Z]+)(\d+)(_SEI)?\}")
BRACE_RE = re.compile(r"[{｛][^}｝]*[}｝]")
PREFS = sorted({p for p, _ in PREF_CITIES}, key=len, reverse=True)
SEI_SET = set(SURNAMES_KANJI)
MEI_SET = set(GIVEN_KANJI)


def load(paths):
    """ワークフロー出力からテンプレートを読む。HTMLエンティティ（&gt;等）を正規化する
    （LLM出力に稀に混入し、そのまま学習データに入ると不自然な文字列になる）。"""
    import html
    tpls = []
    for p in paths:
        raw = open(p).read()
        obj = json.loads(raw[raw.index("{"):])
        data = obj.get("result", obj)
        if isinstance(data, str):
            data = json.loads(data)
        for t in data.get("templates", []):
            t = dict(t)
            t["template"] = html.unescape(t["template"])
            tpls.append(t)
    return tpls


# 生成器の値域と噛み合わないスロット用法（検証エージェントの指摘を機械化）。
# 例: gen_facilityは「○○病院/駅/窓口/百貨店△△店」等を返すため「{FACILITY1}長」は
# 「○○タワービル長」のように破綻する。gen_orgは76%が企業名なので「{ORG1}議会」も破綻。
BAD_USAGE = [
    (re.compile(r"\{FACILITY\d+\}(長|支店|営業所|改札|区域|学区|校区)"), "FACILITY値に付かない語"),
    (re.compile(r"\{ORG\d+\}(長|議会|条例|教育委員会|選挙管理委員会|税務課|会員|区域)"), "ORG値に付かない語"),
    (re.compile(r"貴社\{ORG\d+\}"), "貴社+ORG（官公庁値で破綻）"),
    (re.compile(r"\{LOCATION\d+\}(方面の入口|改札)"), "LOCATION値に付かない語"),
    (re.compile(r"\{KPERSON\d+\}（\{PERSON\d+\}の"), "KPERSONは同番号PERSONのフリガナ規約と矛盾"),
]


def check(t):
    """(致命的な問題のリスト, 警告のリスト) を返す"""
    text, errs, warns = t["template"], [], []
    # スロット構文
    for m in BRACE_RE.finditer(text):
        tok = m.group(0)
        sm = SLOT_RE.fullmatch(tok)
        if not sm:
            errs.append(f"不正なスロット記法: {tok!r}")
            continue
        if sm.group(1) not in ALLOWED:
            errs.append(f"未定義スロット: {tok!r}")
        if sm.group(3) and sm.group(1) != "PERSON":
            errs.append(f"_SEIはPERSONのみ: {tok!r}")
    # 実名混入（スロット外に姓+名の連続、都道府県+市、株式会社+語がある）
    bare = SLOT_RE.sub("", text)
    for i in range(len(bare) - 1):
        for ln_s in (3, 2, 1):
            sei = bare[i:i+ln_s]
            if sei not in SEI_SET:
                continue
            for ln_m in (3, 2):
                if bare[i+ln_s:i+ln_s+ln_m] in MEI_SET:
                    warns.append(f"実名らしき連続: {sei + bare[i+ln_s:i+ln_s+ln_m]!r}")
                    break
            break
    for pref in PREFS:
        idx = bare.find(pref)
        if idx >= 0 and re.match(r"[一-龥ぁ-んァ-ヶ]{1,4}[市区町村]", bare[idx+len(pref):]):
            warns.append(f"住所らしき記述: {bare[idx:idx+10]!r}")
            break
    for form in ("株式会社", "合同会社", "有限会社"):
        for m in re.finditer(re.escape(form) + r"([一-龥ァ-ヶ]{2,6})", bare):
            warns.append(f"社名らしき記述: {m.group(0)!r}")
    # 生成器の値域と噛み合わないスロット用法
    for rx, why in BAD_USAGE:
        m = rx.search(text)
        if m:
            errs.append(f"値域不整合({why}): {m.group(0)!r}")
    # 実在しうる電話番号・郵便番号の直書き（スロットを使うべき箇所）
    if re.search(r"0\d{1,3}[-−－]\d{2,4}[-−－]\d{3,4}", bare) or \
            re.search(r"〒\s*[０-９\d]{3}\s*[-−－]\s*[０-９\d]{4}", bare):
        errs.append("電話/郵便番号の直書き（スロットを使うべき）")
    return errs, warns


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--reject=")]
    reject_texts = set()
    for a in sys.argv[1:]:
        if a.startswith("--reject="):
            import html as _html
            reject_texts.update(_html.unescape(x) for x in json.load(open(a.split("=", 1)[1])))
    tpls = load(args)
    print(f"入力テンプレート: {len(tpls)}件 / 検証エージェントの除外指定: {len(reject_texts)}件")
    if reject_texts:
        before = len(tpls)
        tpls = [t for t in tpls if t["template"] not in reject_texts]
        print(f"  → 除外適用後: {len(tpls)}件（{before - len(tpls)}件除外）")
    ok, dropped, warned = [], [], []
    seen_skel = set()
    for t in tpls:
        errs, warns = check(t)
        if errs:
            dropped.append((t, errs)); continue
        skel = SLOT_RE.sub("<>", t["template"])[:150]
        if skel in seen_skel:
            dropped.append((t, ["骨組み重複"])); continue
        seen_skel.add(skel)
        if warns:
            warned.append((t, warns))
        ok.append(t)
    print(f"合格: {len(ok)}件 / 除外: {len(dropped)}件 / 実名混入の疑い: {len(warned)}件")
    for t, errs in dropped[:8]:
        print(f"  [除外] {errs} :: {t['template'][:60]!r}")
    for t, warns in warned[:10]:
        print(f"  [警告] {warns[:2]} :: {t['template'][:60]!r}")
    # ジャンル別・スロット分布
    print("  ジャンル:", dict(Counter(t["genre"] for t in ok)))
    slots = Counter()
    n_neg = 0
    for t in ok:
        found = SLOT_RE.findall(t["template"])
        if not found:
            n_neg += 1
        for base, _, sei in found:
            slots[base + ("_SEI" if sei else "")] += 1
    print("  スロット出現:", dict(slots.most_common()))
    print(f"  負例（スロットなし）: {n_neg}件")
    # 直後文脈の多様性（テンプレート段階）
    post = Counter()
    for t in ok:
        for m in SLOT_RE.finditer(t["template"]):
            post[t["template"][m.end():m.end()+3]] += 1
    print(f"  スロット直後3gramの異なり数: {len(post)} / 出現{sum(post.values())} = {len(post)/max(1,sum(post.values())):.3f}")
    print(f"  上位: {[f'{k!r}×{v}' for k, v in post.most_common(6)]}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_templates_v5.json")
    json.dump(ok, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
