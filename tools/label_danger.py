"""モデル別・ラベル別の危険gold数（coverage<50%）と PERSON の危険例を出す。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/label_danger.py <model_dir または Hub ID> [...]
モデルはフォルダパスか Hub ID（tools/diff_danger.py の model_path で解決）。データフォルダ内の登録名は
diff_danger.PATHS に加えてここでも追加する。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_danger as dd  # noqa: E402
from privacy_masker.refiner_janome import MorphologicalRefiner  # noqa: E402

for tag in ("v45data", "v45qwen", "v45b", "v45bqwen", "v45c", "v45cq", "v5data", "v5qwen"):
    for size in ("30m", "70m"):
        short = tag.replace("data", "")
        dd.PATHS[f"{size}({short})"] = f"{dd.GD}/outputs_ruri-v3-{size}-{tag}/best"

LABELS = ["PERSON", "ADDRESS", "ORG", "FACILITY", "LOCATION"]


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__); return
    docs = dd.d.load_docs()
    ref = MorphologicalRefiner()
    print(f"{'model':<16}" + "".join(f"{l:>10}" for l in LABELS) + f"{'total':>8}")
    for tag in sys.argv[1:]:
        ds, _ = dd.danger_set(tag, docs, ref)
        c = Counter(v[1] for v in ds.values())
        print(f"{tag:<16}" + "".join(f"{c.get(l, 0):>10}" for l in LABELS) + f"{len(ds):>8}")
        persons = [v for v in ds.values() if v[1] == "PERSON"]
        print("    PERSON危険:", [f"{v[2]}｟{v[3][:30]}｠" for v in persons])


if __name__ == "__main__":
    main()
