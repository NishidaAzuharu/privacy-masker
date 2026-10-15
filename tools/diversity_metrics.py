import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
"""学習データの「文脈多様性」を定量化する。
テンプレート過学習の指標: エンティティ前後の文脈パターンがどれだけ違うか。
wiki実テキストを「目標分布」として、合成データがどれだけ近いかを見る。"""
import json, os, sys
from collections import Counter

GD = drive_dir()

def metrics(path, limit=None, label=""):
    docs = []
    for i, line in enumerate(open(path)):
        if limit and i >= limit:
            break
        docs.append(json.loads(line))
    post, pre, skel = Counter(), Counter(), Counter()
    n_ent = 0
    verb_after = 0
    lens = []
    for d in docs:
        text = d["text"]
        lens.append(len(text))
        # 文書の骨組み（エンティティを型名に置換した文字列）= テンプレート再利用の指標
        parts, last = [], 0
        for s, e, lab in sorted(d["entities"]):
            parts.append(text[last:s]); parts.append(f"<{lab}>"); last = e
        parts.append(text[last:])
        skel["".join(parts)[:120]] += 1
        for s, e, lab in d["entities"]:
            n_ent += 1
            post[text[e:e+3]] += 1
            pre[text[max(0, s-3):s]] += 1
    uniq_post = len(post)
    top10 = sum(c for _, c in post.most_common(10))
    print(f"=== {label or path} ({len(docs)}文書 / {n_ent}エンティティ) ===")
    print(f"  平均文書長           : {sum(lens)//len(lens)}字")
    print(f"  直後3gramの異なり数   : {uniq_post}  (エンティティあたり {uniq_post/max(1,n_ent):.3f})")
    print(f"  直後3gram上位10の占有 : {top10/max(1,n_ent)*100:.1f}%  ← 低いほど多様")
    print(f"  直前3gramの異なり数   : {len(pre)}")
    print(f"  骨組みの異なり数      : {len(skel)} / {len(docs)}文書 = {len(skel)/len(docs)*100:.1f}%  ← 高いほど多様")
    print(f"  最頻の直後3gram      : {[f'{k!r}×{v}' for k, v in post.most_common(6)]}")
    return {"uniq_post_ratio": uniq_post/max(1,n_ent), "top10": top10/max(1,n_ent),
            "skel_ratio": len(skel)/len(docs)}

if __name__ == "__main__":
    targets = [(f"{GD}/out_ner.jsonl", 3000, "wiki実テキスト（目標分布）"),
               (f"{GD}/synthetic_pii_v4.jsonl", 3000, "合成v4（手書きテンプレート53種）")]
    extra = sys.argv[1:]
    for p in extra:
        targets.append((p, 3000, f"新データ: {os.path.basename(p)}"))
    for p, lim, lab in targets:
        if os.path.exists(p):
            metrics(p, lim, lab)
            print()
