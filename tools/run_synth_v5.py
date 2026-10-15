import json, os, re, sys, traceback
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import drive_dir  # noqa: E402
"""synth_pii_data_v5.ipynb をローカルCPUで実行するランナー。
!pip・google.colab セルはスキップし、OUT_DIR をデータフォルダ（PRIVACY_MASKER_DRIVE_DIR）に差し替える。
HF datasetsストリーミングの非デーモンスレッドが残るため最後は os._exit。"""

from _paths import notebooks_dir  # noqa: E402
# usage: PYTHONPATH=. .venv/bin/python tools/run_synth_v5.py [synth_pii_data_v5d.ipynb]
NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(notebooks_dir(), "synth_pii_data_v5.ipynb")
if not os.path.isabs(NB):
    NB = os.path.join(notebooks_dir(), NB)
GD = drive_dir()
os.environ.setdefault("SYNTH_OUT_DIR", GD)   # v5c 以降は OUT_DIR を環境変数で差し替えられる

nb = json.load(open(NB))
g = {"__name__": "__main__"}
try:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!") or "google.colab" in src:
            print(f"[skip] cell {i}", flush=True)
            continue
        src = re.sub(r'OUT_DIR\s*=\s*"/content/drive/MyDrive/[^"]*"', f'OUT_DIR      = {GD!r}', src)   # Colab の出力先をローカルへ
        # LLM増殖セル（RUN_LLM_EXPANSION=False）はそのまま実行して自動no-op
        print(f"[run ] cell {i}", flush=True)
        exec(compile(src, f"<cell {i}>", "exec"), g)
    print("[done] 全セル完了", flush=True)
    os._exit(0)
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
