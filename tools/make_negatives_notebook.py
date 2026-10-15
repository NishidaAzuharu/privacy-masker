"""負例生成ノートブック `qwen3_8_27b_negatives_v1.ipynb` を生成し、RUN_LLM=False で乾走する。

ノートブックはプロンプトを持たない。`tools/negatives_v8.py plan` がデータフォルダに書いた
`negatives_v8/plan_v1.jsonl` と manifest（system prompt・出力 schema 込み）を読み、vLLM で生成して
raw jsonl を書くだけ。選別はローカルの `tools/negatives_v8.py filter`。

usage（リポジトリ直下で）:
  PYTHONPATH=. .venv/bin/python tools/make_negatives_notebook.py            # データフォルダ直下と notebooks_dir に書く
  PYTHONPATH=. .venv/bin/python tools/make_negatives_notebook.py --dryrun   # 一時ディレクトリで全セルを RUN_LLM=False 実行
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import drive_dir, notebooks_dir  # noqa: E402

NOTEBOOK_NAME = "qwen3_8_27b_negatives_v1.ipynb"
MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"   # 2026-08-14 の main

CELL_HEADER = """# Qwen3.8-27B-FP8 による負例（PII を含まない日本語文書）の生成 v1

学習データ v8 の負例を作る。**このノートブックはプロンプトを持たない。** ローカルの
`tools/negatives_v8.py plan` がデータフォルダの `negatives_v8/plan.jsonl` に書いた計画（ジャンル・題材・形式・
文体・長さ・プロンプト本文）と manifest（system prompt・出力 JSON schema）を読み、vLLM で生成して
raw 出力をデータフォルダに書くだけ。出力 JSON は `{"lines": [...]}`（1 行 1 要素。モデルは文字列内の `\n` を書かないため）で、
本文への結合はローカルで行う。選別（地名・人名辞書、NER の兆候、近重複）はローカル CPU の
`tools/negatives_v8.py filter` が行う。LLM 検証器は使わない。

## 実行前の注意

- ランタイムは **Colab G4（NVIDIA RTX PRO 6000 Blackwell 96GB）**。他の GPU では起動時に止まる。
- 依存関係セルは初回に vLLM 0.27.1 を uv で入れて **ランタイムを自動再起動**する。再接続後に
  「すべてのセルを実行」をもう一度。
- モデルは `Qwen/Qwen3.8-27B-FP8`（密結合 27B、FP8、重み約 31 GB）。Qwen3.6-35B-A3B と同じ
  `Qwen3_5ForConditionalGeneration` 系なので vLLM 0.27.1 で動く。MoE ではないので `moe_backend` の
  指定は要らない。SM120 で必要な回避策は `VLLM_USE_FLASHINFER_SAMPLER=0` だけ（設定済み）。
- `RUN_PROFILE`: まず **pilot（先頭 200 件）** を回し、末尾の集計で速度と簡易品質を見てから **full**（計画全件、
  既定 8,000 件）。full は pilot の出力を再利用する（同じ計画・同じ seed なので生成し直さない）。
- 出力はデータフォルダの `qwen3_8_27b_negatives_v8_<plan_tag>_<profile>_seed42/qwen_negatives_raw.jsonl`。バッチごとに
  追記保存するので途中で切れても `RESUME=True` で続きから再開できる。
- **正式な学習データは書き換えない。** 採用は選別ツールと人の確認を経てから。
"""

CELL_DEPS = r'''# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL
# 参照Notebook（qwen3_6_synth_pii_data_v8_11）と同じ uv auto 解決で vLLM/PyTorch を揃え、変更後は1回再起動する。
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


IS_COLAB = (
    "google.colab" in sys.modules
    or module_available("google.colab")
    or "COLAB_RELEASE_TAG" in os.environ
    or "COLAB_BACKEND_VERSION" in os.environ
    or pathlib.Path("/content").is_dir()
)
VLLM_PIN = "0.27.1"
TORCH_BACKEND = "auto"
TEXT_ONLY_UNINSTALL = ("torchaudio", "torchcodec")
READY = pathlib.Path("/content/.qwen38_negatives_vllm_0_27_1_uv_auto.ready")


def distribution_version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_state() -> dict[str, str | None]:
    return {
        package: distribution_version_or_none(package)
        for package in ("vllm", "torch", "torchvision", *TEXT_ONLY_UNINSTALL, "uv")
    }


def metadata_looks_compatible(state: dict[str, str | None]) -> bool:
    return bool(
        state["vllm"] == VLLM_PIN
        and state["torch"]
        and "+cu13" in state["torch"]
        and state["torchvision"]
        and "+cu13" in state["torchvision"]
        and state["torchaudio"] is None
        and state["torchcodec"] is None
    )


PROBE_CODE = r"""
import ctypes
import importlib.metadata
import json
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torchvision
cuda = str(torch.version.cuda or "")
if not cuda.startswith("13."):
    raise RuntimeError(f"PyTorch CUDA must be 13.x for this vLLM wheel: {cuda}")
ctypes.CDLL("libcudart.so.13")
from vllm import LLM, SamplingParams
print(json.dumps({
    "vllm": importlib.metadata.version("vllm"),
    "torch": importlib.metadata.version("torch"),
    "torchvision": importlib.metadata.version("torchvision"),
    "torch_cuda": cuda,
}))
"""


def probe_gpu_stack() -> tuple[dict | None, str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", PROBE_CODE],
            text=True,
            stderr=subprocess.STDOUT,
        )
        required_keys = {"vllm", "torch", "torchvision", "torch_cuda"}
        for line in reversed(output.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and required_keys <= set(candidate):
                return candidate, ""
        return None, "probe出力にruntime JSONがありません: " + output[-3000:]
    except Exception as exc:
        detail = getattr(exc, "output", "") or str(exc)
        return None, str(detail)[-3000:]


def run_cmd(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.check_call(args)


def uninstall_text_only_optionals() -> None:
    subprocess.run(
        ["uv", "pip", "uninstall", "--system", *TEXT_ONLY_UNINSTALL],
        check=False,
    )


if not IS_COLAB:
    print("Colab外: dependency installをskip")
else:
    try:
        import jsonschema  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jsonschema"])

    before = dependency_state()
    probe, probe_error = (probe_gpu_stack() if metadata_looks_compatible(before) else (None, ""))
    if probe is not None:
        READY.write_text(json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8")
        print("dependency ready:", probe)
    else:
        print("uv autoでGPUスタックを修復します:", before)
        if probe_error:
            print("修復前probe:", probe_error)
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"])
        run_cmd([
            "uv", "pip", "install", "--system", "--upgrade",
            f"vllm=={VLLM_PIN}", f"--torch-backend={TORCH_BACKEND}",
        ])
        uninstall_text_only_optionals()
        probe, probe_error = probe_gpu_stack()

        if probe is None:
            print("通常のuv auto導入後もprobe失敗。強制再解決します:", probe_error)
            run_cmd([
                "uv", "pip", "install", "--system", "--upgrade",
                f"vllm=={VLLM_PIN}", "torch", "torchvision",
                "--reinstall-package", "vllm",
                "--reinstall-package", "torch",
                "--reinstall-package", "torchvision",
                f"--torch-backend={TORCH_BACKEND}",
            ])
            uninstall_text_only_optionals()
            probe, probe_error = probe_gpu_stack()

        after = dependency_state()
        if probe is None or not metadata_looks_compatible(after):
            raise RuntimeError(
                "uv autoでGPUスタックを揃えられません。"
                f" state={after} / probe={probe_error}"
            )
        READY.write_text(json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8")
        print("導入完了:", probe)
        print("ランタイムを再起動します。再接続後に「すべてのセルを実行」してください。")
        sys.stdout.flush()
        time.sleep(2)
        os.kill(os.getpid(), signal.SIGKILL)
'''

CELL_IMPORTS = r'''from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from jsonschema import Draft202012Validator

# Colab の ipykernel 内で vLLM を初期化する前に設定する（SM120 では FlashInfer sampler の JIT が失敗する）。
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """一時ファイルへ書いてから置換する（中断時に壊れたファイルを残さない）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
'''

CELL_DRIVE = r'''# Google Drive をマウント。Colab では失敗時にローカルへフォールバックしない（成果物が消えるため）。
import os
from pathlib import Path

COLAB_DRIVE_BASE = Path(
    "/content/drive/MyDrive/privacy-masker-data"   # 自分の Google Drive 上のデータフォルダに合わせる
)

try:
    from google.colab import drive  # type: ignore
except ImportError:
    local_override = os.environ.get("PRIVACY_MASKER_DRIVE_BASE")
    DRIVE_BASE = Path(local_override) if local_override else Path(".")
    print("Colab外で実行中: 出力先 =", DRIVE_BASE)
else:
    try:
        drive.mount("/content/drive")
    except Exception as exc:
        raise RuntimeError(
            "Google Driveのマウントに失敗しました。永続化できないため生成を停止します。"
        ) from exc
    DRIVE_BASE = COLAB_DRIVE_BASE

DRIVE_BASE.mkdir(parents=True, exist_ok=True)
write_probe = DRIVE_BASE / ".qwen38_negatives_write_probe"
try:
    write_probe.write_text("ok", encoding="utf-8")
    if write_probe.read_text(encoding="utf-8") != "ok":
        raise OSError("write probe mismatch")
except Exception as exc:
    raise RuntimeError(f"出力先へ書き込めません: {DRIVE_BASE}") from exc
finally:
    if write_probe.exists():
        write_probe.unlink()

print("DRIVE_BASE:", DRIVE_BASE)
'''

CELL_CONFIG = r'''# @title 負例生成の設定（Qwen3.8-27B-FP8 / Colab G4）
RUN_PROFILE = "pilot"  # @param ["pilot", "full"]
RUN_LLM = True  # @param {type:"boolean"}
RESUME = True  # @param {type:"boolean"}
BATCH_SIZE = 128  # @param {type:"integer"}

# pilot は計画の先頭 200 件（計画はジャンル均衡順なので先頭を取っても比率が保たれる）。full は全件。
RUN_PROFILES = {"pilot": {"max_docs": 200}, "full": {"max_docs": None}}
if RUN_PROFILE not in RUN_PROFILES:
    raise ValueError(f"未知のRUN_PROFILE: {RUN_PROFILE}")
if BATCH_SIZE <= 0:
    raise ValueError("BATCH_SIZEは正の整数にしてください")

QWEN_MODEL_ID = "__MODEL_ID__"
QWEN_MODEL_REVISION = "__MODEL_REVISION__"
GPU_PRESET = "colab_g4_96gb"
GPU_PRESETS = {
    "colab_g4_96gb": {
        "minimum_total_vram_gib": 84.0, "require_gpu_name": "RTX PRO 6000",
        # 密結合 27B FP8（重み約 31 GB）。KV キャッシュに余裕があるので同時系列数は A3B 時の 8 から 32 に上げる。
        "max_model_len": 4096, "gpu_memory_utilization": 0.92, "max_num_seqs": 32, "enforce_eager": False,
    },
}
PRESET = GPU_PRESETS[GPU_PRESET]
SEED = 42

PLAN_DIR = DRIVE_BASE / "negatives_v8"
PLAN_PATH = PLAN_DIR / "plan.jsonl"
PLAN_MANIFEST_PATH = PLAN_DIR / "plan.manifest.json"


@dataclass(frozen=True)
class GenConfig:
    run_profile: str = RUN_PROFILE
    max_docs: int | None = RUN_PROFILES[RUN_PROFILE]["max_docs"]
    gpu_preset: str = GPU_PRESET
    model_id: str = QWEN_MODEL_ID
    model_revision: str = QWEN_MODEL_REVISION
    minimum_total_vram_gib: float = PRESET["minimum_total_vram_gib"]
    require_gpu_name: str = PRESET["require_gpu_name"]
    max_model_len: int = PRESET["max_model_len"]
    gpu_memory_utilization: float = PRESET["gpu_memory_utilization"]
    max_num_seqs: int = PRESET["max_num_seqs"]
    enforce_eager: bool = PRESET["enforce_eager"]
    flashinfer_sampler: str = "0"
    language_model_only: bool = True      # 視覚塔は読まない（Qwen3.8-27B は VL モデル）
    generation_config: str = "vllm"
    tensor_parallel_size: int | None = None
    batch_size: int = BATCH_SIZE
    run_llm: bool = RUN_LLM
    resume: bool = RESUME
    max_structured_attempts: int = 2
    # Qwen 推奨の非思考モード設定。前回の Qwen3.6 系（temperature 0.2）は同型化したので上げる
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    seed: int = SEED
    pipeline_version: str = "qwen3_8_negatives_v1"
    vllm_version: str = "0.27.1"
    dependency_resolver: str = "uv"
    torch_backend: str = "auto"
    required_torch_cuda_major: str = "13"
    excluded_runtime_packages: tuple[str, ...] = ("torchaudio", "torchcodec")


CFG = GenConfig()
print(json.dumps(asdict(CFG), ensure_ascii=False, indent=2))
print("PLAN_PATH:", PLAN_PATH)
print("出力先は計画の plan_tag を含む名前で、次のセルで決まります。正式な学習JSONLは更新しません。")
'''

CELL_PLAN = r'''# 計画と manifest を読む。プロンプト・system prompt・出力 schema はすべて計画側にある。
if not PLAN_PATH.exists() or not PLAN_MANIFEST_PATH.exists():
    raise RuntimeError(
        f"計画がありません: {PLAN_PATH}\n"
        "ローカルで `PYTHONPATH=. .venv/bin/python tools/negatives_v8.py plan` を実行してデータフォルダに置いてください。"
    )
PLAN_MANIFEST = json.loads(PLAN_MANIFEST_PATH.read_text(encoding="utf-8"))
PLAN_SHA256 = file_sha256(PLAN_PATH)
if PLAN_MANIFEST.get("plan_sha256") != PLAN_SHA256:
    raise RuntimeError(
        f"計画ファイルが manifest と一致しません: {PLAN_SHA256[:16]} != {str(PLAN_MANIFEST.get('plan_sha256'))[:16]}。"
        "計画を作り直したか、コピーが壊れています。"
    )
FULL_PLAN = read_jsonl(PLAN_PATH)
if not FULL_PLAN:
    raise RuntimeError("計画が空です")
for required in ("request_id", "index", "genre_key", "subgenre_key", "format_key", "length_key", "max_tokens", "prompt", "prompt_sha256"):
    missing = [row.get("request_id") for row in FULL_PLAN if required not in row]
    if missing:
        raise RuntimeError(f"計画の列 {required} が欠けています: {missing[:3]}")

SYSTEM_PROMPT = PLAN_MANIFEST["system_prompt"]
OUTPUT_SCHEMA = PLAN_MANIFEST["output_schema"]
PROMPT_VERSION = PLAN_MANIFEST["prompt_version"]
PLAN_VERSION = PLAN_MANIFEST["plan_version"]
PLAN_TAG = PLAN_MANIFEST.get("plan_tag") or PLAN_VERSION

# 出力先は計画の版を含む（計画を作り直したら別フォルダになり、旧 run の署名と衝突しない）
OUT_DIR = DRIVE_BASE / f"qwen3_8_27b_negatives_v8_{PLAN_TAG}_{CFG.run_profile}_seed{CFG.seed}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT_DIR / "qwen_negatives_raw.jsonl"
MANIFEST_PATH = OUT_DIR / "qwen_negatives_manifest.json"
# full は pilot の出力を再利用する（同じ計画・同じ seed・同じ prompt sha なら生成し直さない）
EXTRA_CHECKPOINT_PATHS = [
    DRIVE_BASE / f"qwen3_8_27b_negatives_v8_{PLAN_TAG}_pilot_seed{CFG.seed}" / "qwen_negatives_raw.jsonl"
] if CFG.run_profile == "full" else []
print("OUT_DIR:", OUT_DIR)

PLAN = FULL_PLAN if CFG.max_docs is None else FULL_PLAN[: CFG.max_docs]
if PLAN != sorted(PLAN, key=lambda r: r["index"]):
    raise RuntimeError("計画が index 順ではありません")

# 同じモデル・同じ計画・同じサンプリングなら同じ signature。pilot と full で共通（full が pilot の出力を再利用できる）
RUN_SIGNATURE = sha256_short(json.dumps({
    "model_id": CFG.model_id, "model_revision": CFG.model_revision, "plan_sha256": PLAN_SHA256,
    "prompt_version": PROMPT_VERSION, "pipeline_version": CFG.pipeline_version,
    "sampling": {"temperature": CFG.temperature, "top_p": CFG.top_p, "top_k": CFG.top_k, "min_p": CFG.min_p,
                 "presence_penalty": CFG.presence_penalty, "repetition_penalty": CFG.repetition_penalty, "seed": CFG.seed},
}, sort_keys=True, ensure_ascii=False))

print(f"計画 {len(FULL_PLAN)} 件（{PLAN_VERSION} / {PROMPT_VERSION}）、今回の対象 {len(PLAN)} 件、signature {RUN_SIGNATURE}")
print("ジャンル:", dict(Counter(r["genre_key"] for r in PLAN)))
print("形式:", dict(Counter(r["format_key"] for r in PLAN)))
print("長さ:", dict(Counter(r["length_key"] for r in PLAN)))
print("\n--- system prompt ---\n" + SYSTEM_PROMPT[:600] + ("…" if len(SYSTEM_PROMPT) > 600 else ""))
print("\n--- 計画 1 件目の user prompt ---\n" + PLAN[0]["prompt"])
'''

CELL_VLLM = r'''import ctypes
import importlib.metadata


def runtime_distribution_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


RUNTIME_BINARY_VERSIONS = {
    package: runtime_distribution_version(package)
    for package in ("vllm", "torch", "torchvision", "torchaudio", "torchcodec", "uv")
}
runtime_contract_errors = []
if RUNTIME_BINARY_VERSIONS["vllm"] != CFG.vllm_version:
    runtime_contract_errors.append(
        f"vllm={RUNTIME_BINARY_VERSIONS['vllm']!r} (expected {CFG.vllm_version!r})"
    )
cuda_build_marker = f"+cu{CFG.required_torch_cuda_major}"
for package in ("torch", "torchvision"):
    installed = RUNTIME_BINARY_VERSIONS[package]
    if not installed or cuda_build_marker not in installed:
        runtime_contract_errors.append(
            f"{package}={installed!r} ({cuda_build_marker}系buildが必要)"
        )
for package in CFG.excluded_runtime_packages:
    installed = RUNTIME_BINARY_VERSIONS.get(package)
    if installed is not None:
        runtime_contract_errors.append(f"{package}={installed!r} (削除が必要)")
if CFG.run_llm and runtime_contract_errors:
    raise RuntimeError(
        "GPUバイナリがuv autoのCUDA 13系契約を満たしません: "
        + "; ".join(runtime_contract_errors)
        + "。先頭の依存関係セルを実行し、自動再起動後にすべてのセルを再実行してください。"
    )


def gpu_inventory() -> list[dict]:
    try:
        import torch
    except Exception as exc:
        if CFG.run_llm:
            raise RuntimeError("PyTorchを読み込めません。先頭の依存関係セルから再実行してください。") from exc
        return []
    rows = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        rows.append({
            "index": index,
            "name": props.name,
            "vram_gib": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return rows


GPU_ROWS = gpu_inventory()
display(pd.DataFrame(GPU_ROWS))

llm = None
VLLM_RUNTIME_VERSION = RUNTIME_BINARY_VERSIONS.get("vllm")
TORCH_CUDA_RUNTIME_VERSION = None
CUDA_RUNTIME_LIBRARY = f"libcudart.so.{CFG.required_torch_cuda_major}"
LLM_BATCH_TIMINGS: list[dict] = []
INIT_SECONDS = None

if CFG.run_llm:
    if not GPU_ROWS:
        raise RuntimeError("RUN_LLM=True ですがCUDA GPUが見つかりません")
    total_vram = sum(row["vram_gib"] for row in GPU_ROWS)
    if total_vram < CFG.minimum_total_vram_gib:
        raise RuntimeError(
            f"GPU_PRESET={CFG.gpu_preset} には合計{CFG.minimum_total_vram_gib:.1f} GiB以上の"
            f"VRAMが必要ですが、検出値は{total_vram:.1f} GiBです。"
            "このNotebookはColab G4のRTX PRO 6000 96GB専用です。"
        )
    if CFG.require_gpu_name:
        names = " / ".join(row["name"] for row in GPU_ROWS).upper()
        if CFG.require_gpu_name.upper() not in names:
            raise RuntimeError(
                f"{CFG.gpu_preset} を選択していますが {CFG.require_gpu_name} を検出できませんでした: {names}"
            )

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = CFG.flashinfer_sampler
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TorchVisionを読み込めません。uv autoでGPUスタックを入れ直してください。"
        ) from exc

    TORCH_CUDA_RUNTIME_VERSION = str(torch.version.cuda or "")
    expected_cuda_prefix = f"{CFG.required_torch_cuda_major}."
    if not TORCH_CUDA_RUNTIME_VERSION.startswith(expected_cuda_prefix):
        raise RuntimeError(
            f"PyTorchのCUDAビルドが不一致です: {TORCH_CUDA_RUNTIME_VERSION!r}。"
            f"CUDA {CFG.required_torch_cuda_major}.xが必要です。先頭の依存関係セルから再実行してください。"
        )
    try:
        ctypes.CDLL(CUDA_RUNTIME_LIBRARY)
    except OSError as exc:
        raise RuntimeError(
            f"{CUDA_RUNTIME_LIBRARY} を読み込めません。先頭の依存関係セルでuv auto修復後、"
            "自動再起動してからすべてのセルを再実行してください。"
        ) from exc

    print("GPUバイナリ:", json.dumps({
        **RUNTIME_BINARY_VERSIONS,
        "torch_cuda": TORCH_CUDA_RUNTIME_VERSION,
        "cuda_runtime_library": CUDA_RUNTIME_LIBRARY,
    }, ensure_ascii=False))
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        if "libcudart.so.13" in str(exc):
            raise RuntimeError(
                "vLLMがlibcudart.so.13を見つけられません。先頭の依存関係セルを再実行して"
                "uv autoでPyTorch/vLLMを同時に揃え、自動再起動後に全セルを再実行してください。"
            ) from exc
        raise

    if VLLM_RUNTIME_VERSION != CFG.vllm_version:
        raise RuntimeError(
            f"vLLM版が不一致です: {VLLM_RUNTIME_VERSION} != {CFG.vllm_version}"
        )
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.utils.system_utils import set_env_var

    tensor_parallel_size = CFG.tensor_parallel_size or torch.cuda.device_count()
    llm_kwargs = {
        "model": CFG.model_id,
        "revision": CFG.model_revision,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": CFG.max_model_len,
        "gpu_memory_utilization": CFG.gpu_memory_utilization,
        "trust_remote_code": True,
        "enable_prefix_caching": True,     # 共通の system prompt が効く
        "disable_log_stats": True,
        "max_num_seqs": CFG.max_num_seqs,
        "enforce_eager": CFG.enforce_eager,
        "seed": CFG.seed,
        "language_model_only": CFG.language_model_only,
        "generation_config": CFG.generation_config,
        # 密結合モデルなので moe_backend は指定しない。kv_cache_dtype は既定（bf16）のまま:
        # SM120 では FlashInfer + FP8 KV cache + CUDA graph の組み合わせで出力が壊れる既知の問題がある。
    }
    print("vLLM初期化:", json.dumps(
        {k: v for k, v in llm_kwargs.items() if k != "model"}, ensure_ascii=False))
    started = time.perf_counter()
    with set_env_var("VLLM_LOGGING_LEVEL", "DEBUG"):
        llm = LLM(**llm_kwargs)
    INIT_SECONDS = round(time.perf_counter() - started, 1)
    print(f"初期化 {INIT_SECONDS}秒")
else:
    print("RUN_LLM=False: モデルはロードしません（計画と保存経路の確認のみ）")
'''

CELL_BATCH = r'''def parsed_text(parsed) -> str | None:
    """parsed から本文を取る。計画 v3 の {"lines": [...]}（1 行 1 要素）を "\\n" で結合する。旧 {"text": ...} も受ける。"""
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("lines"), list):
        text = "\n".join(str(x).rstrip() for x in parsed["lines"]).strip()
        return re.sub(r"\n{3,}", "\n\n", text) or None
    text = parsed.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def chunk_ranges(total: int, *, start: int = 0):
    if CFG.batch_size <= 0:
        raise ValueError("CFG.batch_size must be positive")
    for left in range(start, total, CFG.batch_size):
        yield range(left, min(total, left + CFG.batch_size))


def run_structured_batch(conversations: list[list[dict]], schema: dict, sampling,
                         *, stage: str, max_attempts: int | None = None) -> list[dict]:
    """schema 失敗を設定回数まで試行し、残存失敗も位置を保って返す。"""
    max_attempts = max_attempts or CFG.max_structured_attempts
    if not conversations:
        return []
    if len(conversations) > CFG.batch_size:
        raise RuntimeError(
            f"unchunked LLM batch: {len(conversations)} > CFG.batch_size={CFG.batch_size}"
        )
    if not CFG.run_llm:
        return [{"parsed": None, "raw_text": "", "error": "RUN_LLM=False", "attempt": 0}
                for _ in conversations]

    validator = Draft202012Validator(schema)
    results: list[dict | None] = [None] * len(conversations)
    pending = list(range(len(conversations)))

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        batch = []
        for index in pending:
            messages = [dict(m) for m in conversations[index]]
            if attempt > 1:
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] += (
                    "\n\n前回の出力はJSON schema検証に失敗しました。"
                    "schemaを厳守してJSONのみを再出力してください。")
            batch.append(messages)

        started = time.perf_counter()
        if isinstance(sampling, (list, tuple)):
            if len(sampling) != len(conversations):
                raise ValueError("request別sampling数がconversation数と一致しません")
            batch_sampling = [sampling[index] for index in pending]
        else:
            batch_sampling = sampling
        outputs = llm.chat(batch, batch_sampling, use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": False})
        elapsed = time.perf_counter() - started
        generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
        timing = {"stage": stage, "attempt": attempt, "requests": len(outputs),
                  "seconds": round(elapsed, 1),
                  "req_per_min": round(len(outputs) * 60 / elapsed, 1) if elapsed else None,
                  "generated_tokens": generated_tokens,
                  "tok_per_sec": round(generated_tokens / elapsed, 1) if elapsed else None}
        LLM_BATCH_TIMINGS.append(timing)
        print("batch:", json.dumps(timing, ensure_ascii=False))

        still_pending = []
        for index, output in zip(pending, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed, error = None, ""
            try:
                candidate = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                error = f"json_decode_error: {exc}"
            else:
                errors = sorted(validator.iter_errors(candidate), key=lambda e: e.path)
                if errors:
                    error = f"schema_error: {errors[0].message[:200]}"
                else:
                    parsed = candidate
            results[index] = {"parsed": parsed, "raw_text": raw_text,
                              "error": error, "attempt": attempt}
            if parsed is None and attempt < max_attempts:
                still_pending.append(index)
        pending = still_pending

    return [r if r is not None else
            {"parsed": None, "raw_text": "", "error": "no_result", "attempt": max_attempts}
            for r in results]
'''

CELL_GENERATE = r'''if MANIFEST_PATH.exists():
    try:
        _existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("既存manifestが破損しています。既存runを保護するため停止しました。") from exc
    _existing_signature = _existing_manifest.get("run_signature") if isinstance(_existing_manifest, dict) else None
    if _existing_signature != RUN_SIGNATURE:
        raise RuntimeError(
            "出力フォルダに異なる run signature の manifest があります。既存 run を保護するため停止しました。"
            " 別の計画・モデルで回すなら OUT_DIR を変えてください。"
        )


def checkpoint_map(paths: list[Path], key: str) -> dict[str, dict]:
    """同じ signature・同じ prompt sha の行だけを再利用する。RESUME=False なら空。"""
    if not CFG.resume:
        return {}
    found: dict[str, dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            if (row.get("run_signature") == RUN_SIGNATURE and row.get(key)
                    and int(row.get("attempt") or 0) >= 1
                    and row.get("structured_error") != "RUN_LLM=False"):
                found[str(row[key])] = row
    return found


def sampling_for(plan: dict):
    if not CFG.run_llm:
        return None
    return SamplingParams(
        temperature=CFG.temperature, top_p=CFG.top_p, top_k=CFG.top_k, min_p=CFG.min_p,
        presence_penalty=CFG.presence_penalty, repetition_penalty=CFG.repetition_penalty,
        max_tokens=int(plan["max_tokens"]),
        seed=CFG.seed + 300_000 + int(plan["index"]),
        structured_outputs=StructuredOutputsParams(json=OUTPUT_SCHEMA),
    )


existing = checkpoint_map([RAW_PATH, *EXTRA_CHECKPOINT_PATHS], "request_id")
rows_by_id: dict[str, dict] = {}
pending_plans: list[dict] = []
for plan in PLAN:
    prior = existing.get(plan["request_id"])
    if prior and prior.get("prompt_sha256") == plan["prompt_sha256"]:
        rows_by_id[plan["request_id"]] = prior
        continue
    pending_plans.append(plan)
reused = len(rows_by_id)
print(f"再利用 {reused} 件、今回生成 {len(pending_plans)} 件")


def persist() -> None:
    if not CFG.run_llm:
        return    # RUN_LLM=False では raw を書かない（RESUME 時に混ざらないため）
    write_jsonl(RAW_PATH, [rows_by_id[p["request_id"]] for p in PLAN if p["request_id"] in rows_by_id])


run_started = time.perf_counter()
if pending_plans:
    conversations = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p["prompt"]}]
                     for p in pending_plans]
    samplings = [sampling_for(p) for p in pending_plans]
    # 最初の 1 件を canary として完走させてから残りを処理する
    batch_ranges = [range(0, 1)]
    batch_ranges.extend(chunk_ranges(len(pending_plans), start=1))
    done = 0
    for positions in batch_ranges:
        positions = list(positions)
        if not positions:
            continue
        outputs = run_structured_batch(
            [conversations[i] for i in positions], OUTPUT_SCHEMA,
            [samplings[i] for i in positions], stage="negatives",
        )
        for index, output in zip(positions, outputs):
            plan = pending_plans[index]
            parsed = output.get("parsed")
            text = parsed_text(parsed)
            rows_by_id[plan["request_id"]] = {
                "run_signature": RUN_SIGNATURE, "request_id": plan["request_id"], "index": plan["index"],
                "genre_key": plan["genre_key"], "subgenre_key": plan["subgenre_key"],
                "format_key": plan["format_key"], "length_key": plan["length_key"],
                "prompt_sha256": plan["prompt_sha256"],
                "sampling_seed": CFG.seed + 300_000 + int(plan["index"]),
                "model_id": CFG.model_id, "model_revision": CFG.model_revision,
                "parsed": parsed, "chars": len(text) if isinstance(text, str) else None,
                "structured_error": output.get("error", ""), "attempt": output.get("attempt"),
                "raw_text": output.get("raw_text", "") if parsed is None else "",
            }
        persist()
        done += len(positions)
        elapsed = time.perf_counter() - run_started
        rate = done / elapsed * 60 if elapsed else 0.0
        remaining = len(pending_plans) - done
        eta_min = remaining / rate if rate else float("nan")
        print(f"進捗 {done}/{len(pending_plans)}  {rate:.1f} 件/分  残り約 {eta_min:.0f} 分")

GENERATION_SECONDS = round(time.perf_counter() - run_started, 1)
ROWS = [rows_by_id[p["request_id"]] for p in PLAN if p["request_id"] in rows_by_id]
ok_rows = [r for r in ROWS if isinstance(r.get("parsed"), dict)]
print(f"対象 {len(PLAN)} 件: 結果あり {len(ROWS)}、JSON 成功 {len(ok_rows)}、構造化失敗 {len(ROWS) - len(ok_rows)}、{GENERATION_SECONDS} 秒")
if len(ROWS) != len(PLAN):
    print("未処理 request あり。RESUME=True で再実行できます")
'''

CELL_SUMMARY = r'''# 集計と manifest。簡易検査は「明らかな違反」を数えるだけで、正式な選別はローカルの tools/negatives_v8.py filter。
PREFECTURES = ("北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県",
               "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県",
               "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県",
               "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県", "熊本県", "大分県",
               "宮崎県", "鹿児島県", "沖縄県")
QUICK_CHECKS = {
    "prefecture": re.compile("|".join(PREFECTURES)),
    "country_region": re.compile(r"日本(?![語人酒茶式製風画])|アメリカ(?![人ン])|中国(?![語料])|欧州|中東|関東|関西|九州"),
    "corporate": re.compile(r"株式会社|有限会社|合同会社|[一-龥]{2,}(?:総務|人事|経理|営業|企画)(?:部|課)"),
    "phone_email_url": re.compile(r"0\d{1,4}-\d{1,4}-\d{3,4}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|https?://|www\."),
    "kanji_name_with_title": re.compile(r"[一-龥]{2,3}(?:さん|様|氏|部長|課長|先生)"),   # 要確認（お客様・皆様も数える）
    "mention_or_handle": re.compile(r"(?<![A-Za-z0-9._%+-])@[A-Za-z][A-Za-z0-9_.]{1,30}|[a-z]+_[a-z0-9]{2,}\d"),
    "foreign_script": re.compile(r"[Ѐ-ӿ가-힯؀-ۿ฀-๿这个们说时为对还现样关从发电让过种经]"),
}
texts = {r["request_id"]: parsed_text(r["parsed"]) or "" for r in ok_rows}
quick_counts = {name: sum(1 for t in texts.values() if rx.search(unicodedata.normalize("NFKC", t))) for name, rx in QUICK_CHECKS.items()}
LINE_FORMATS = {"bullets", "numbered_steps", "key_value", "label_colon", "table_pipe", "log_lines", "chat_roles", "letter"}
quick_counts["newline_docs"] = sum(1 for t in texts.values() if "\n" in t)
quick_counts["line_format_docs"] = sum(1 for p in PLAN if p["format_key"] in LINE_FORMATS and p["request_id"] in texts)
quick_counts["line_format_docs_with_newline"] = sum(
    1 for p in PLAN if p["format_key"] in LINE_FORMATS and "\n" in texts.get(p["request_id"], ""))
lens = sorted(len(t) for t in texts.values())
quantile = (lambda p: lens[int(p * (len(lens) - 1))]) if lens else (lambda p: 0)
plan_by_id = {p["request_id"]: p for p in PLAN}
length_ok = sum(1 for rid, t in texts.items() if plan_by_id[rid]["length_min"] * 0.7 <= len(t) <= plan_by_id[rid]["length_max"] * 1.3)
by_genre_ok = Counter(plan_by_id[rid]["genre_key"] for rid in texts)
by_genre_all = Counter(p["genre_key"] for p in PLAN)
errors = Counter((r.get("structured_error") or "")[:40] for r in ROWS if not isinstance(r.get("parsed"), dict))
total_seconds = sum(t["seconds"] for t in LLM_BATCH_TIMINGS)
total_requests = sum(t["requests"] for t in LLM_BATCH_TIMINGS)
total_tokens = sum(t.get("generated_tokens", 0) for t in LLM_BATCH_TIMINGS)
req_per_min = round(total_requests * 60 / total_seconds, 1) if total_seconds else None
projection_full_min = round(len(FULL_PLAN) / req_per_min) if req_per_min else None

MANIFEST = {
    "pipeline_version": CFG.pipeline_version, "run_profile": CFG.run_profile, "run_signature": RUN_SIGNATURE,
    "status": "complete" if len(ROWS) == len(PLAN) else "partial",
    "model": {"id": CFG.model_id, "revision": CFG.model_revision, "vllm_expected": CFG.vllm_version,
              "vllm_runtime": VLLM_RUNTIME_VERSION, "torch_cuda_runtime": TORCH_CUDA_RUNTIME_VERSION,
              "gpu": GPU_ROWS, "init_seconds": INIT_SECONDS,
              "sampling": {"temperature": CFG.temperature, "top_p": CFG.top_p, "top_k": CFG.top_k, "min_p": CFG.min_p,
                           "presence_penalty": CFG.presence_penalty, "repetition_penalty": CFG.repetition_penalty},
              "max_model_len": CFG.max_model_len, "max_num_seqs": CFG.max_num_seqs},
    "plan": {"path": str(PLAN_PATH), "sha256": PLAN_SHA256, "plan_version": PLAN_VERSION, "prompt_version": PROMPT_VERSION,
             "full_rows": len(FULL_PLAN), "target_rows": len(PLAN)},
    "counts": {"planned": len(PLAN), "reused_from_checkpoint": reused, "generated_now": len(pending_plans),
               "rows": len(ROWS), "json_ok": len(ok_rows), "structured_failed": len(ROWS) - len(ok_rows),
               "length_within_plan_30pct": length_ok, "by_genre_planned": dict(by_genre_all), "by_genre_ok": dict(by_genre_ok)},
    "structured_errors": dict(errors.most_common(10)),
    "chars": {"p10": quantile(.1), "p50": quantile(.5), "p90": quantile(.9), "mean": (sum(lens) // len(lens)) if lens else 0},
    "quick_checks": quick_counts,
    "timing": {"generation_seconds": GENERATION_SECONDS, "llm_seconds": round(total_seconds, 1), "requests": total_requests,
               "generated_tokens": total_tokens, "req_per_min": req_per_min,
               "tok_per_sec": round(total_tokens / total_seconds, 1) if total_seconds else None,
               "projected_full_plan_minutes": projection_full_min},
    "llm_batch_timings": LLM_BATCH_TIMINGS,
    "files": {"raw": str(RAW_PATH), "raw_sha256": file_sha256(RAW_PATH) if RAW_PATH.exists() else None},
    "promote_final": False,
}
write_json(MANIFEST_PATH, MANIFEST)
print(json.dumps({k: MANIFEST[k] for k in ("status", "counts", "structured_errors", "chars", "quick_checks", "timing")},
                 ensure_ascii=False, indent=2))
print("manifest:", MANIFEST_PATH)
print("\n--- 先頭 5 件（先頭 160 字） ---")
for rid in list(texts)[:5]:
    p = plan_by_id[rid]
    print(f"[{p['genre_key']}/{p['subgenre_key']}/{p['format_key']}] {texts[rid][:160].replace(chr(10), '⏎')}")
'''

CELL_AFTER = """## 実行後の手順（ローカル CPU）

1. pilot の `quick_checks` を見る。`prefecture` `corporate` `phone_email_url` `foreign_script` が
   200 件中 10 件を超えていたらプロンプトの見直し（計画の作り直し）を先にする。`kanji_name_with_title` は
   お客様・皆様も数えるので参考値。`newline_docs`（改行を含む文書数）が箇条書き・手順・チャット・ログ・帳票の
   件数より大きく下回るなら行区切りの指示が効いていない。
2. `timing.projected_full_plan_minutes` が Colab のセッション時間に収まるか確認してから `RUN_PROFILE = "full"`。
3. 選別と学習形式への変換はローカルで:

```bash
PYTHONPATH=. .venv/bin/python tools/negatives_v8.py filter \\
  --raw "<データフォルダ>/qwen3_8_27b_negatives_v8_plan_v3_full_seed42" --target 4000
```

   出力は `<データフォルダ>/negatives_v8/filtered_v1/`（`negatives_v8.jsonl`、`report.md`、`rejected.csv`、
   `review_excluded.csv`、`ner_hits.csv`）。review に回った文書は `review_excluded.csv` を見て、
   問題なければ request_id を一覧にして `--include-ids` で戻す。
4. `negatives_v8.jsonl` は v8 の学習データ構築（別ファイル、結合フラグで ON/OFF）に渡す。正式データへの自動統合はしない。
"""


def cells() -> list[tuple[str, str]]:
    config = CELL_CONFIG.replace("__MODEL_ID__", MODEL_ID).replace("__MODEL_REVISION__", MODEL_REVISION)
    return [
        ("markdown", CELL_HEADER),
        ("code", CELL_DEPS),
        ("code", CELL_IMPORTS),
        ("code", CELL_DRIVE),
        ("markdown", "## 1. 設定"),
        ("code", config),
        ("markdown", "## 2. 計画の読み込み（プロンプトは計画側）"),
        ("code", CELL_PLAN),
        ("markdown", "## 3. vLLM 初期化（G4 対応・密結合 FP8）"),
        ("code", CELL_VLLM),
        ("code", CELL_BATCH),
        ("markdown", "## 4. 生成（canary → バッチ、バッチごとにデータフォルダへ保存）"),
        ("code", CELL_GENERATE),
        ("markdown", "## 5. 集計と manifest"),
        ("code", CELL_SUMMARY),
        ("markdown", CELL_AFTER),
    ]


def build_notebook() -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells()):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"neg-v1-{i:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "RTX_PRO_6000", "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(paths: list[str]) -> None:
    nb = build_notebook()
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", path)


def dryrun(plan_rows: int = 40) -> None:
    """一時ディレクトリをデータフォルダに見立て、依存関係セル以外の全コードセルを RUN_LLM=False で実行する。"""
    import negatives_v8 as nv

    tmp = tempfile.mkdtemp(prefix="privacy-masker-negatives-nb-dryrun-")
    plan_dir = os.path.join(tmp, "negatives_v8")
    rows = nv.build_plan(plan_rows, seed=42)
    plan_path = os.path.join(plan_dir, nv.PLAN_FILE)
    nv.write_jsonl(plan_path, rows)
    manifest = nv.plan_manifest(rows, 42, nv.file_sha256(plan_path))
    with open(os.path.join(plan_dir, nv.PLAN_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.environ["PRIVACY_MASKER_DRIVE_BASE"] = tmp

    g = {"__name__": "__main__", "display": lambda x: print(str(x)[:300])}
    ok = True
    try:
        for i, (kind, src) in enumerate(cells()):
            if kind != "code":
                continue
            if "# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in src:
                print(f"[skip] cell {i} (インストール)")
                continue
            src = src.replace("RUN_LLM = True", "RUN_LLM = False")
            print(f"[run ] cell {i}")
            exec(compile(src, f"<cell {i}>", "exec"), g)
        out_dir = g["OUT_DIR"]
        raw = g["RAW_PATH"]
        manifest_path = g["MANIFEST_PATH"]
        assert not raw.exists(), "RUN_LLM=False で raw が書かれた"
        assert manifest_path.exists(), "manifest が書かれていない"
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert m["counts"]["planned"] == plan_rows and m["counts"]["json_ok"] == 0
        assert m["run_signature"] == g["RUN_SIGNATURE"]
        assert g["PLAN"][0]["prompt"] in g["PLAN"][0]["prompt"]
        # 2 回目: RESUME で同じ signature の manifest があっても止まらない
        for i, (kind, src) in enumerate(cells()):
            if kind != "code" or "# DRYRUN_SKIP" in src:
                continue
            exec(compile(src.replace("RUN_LLM = True", "RUN_LLM = False"), f"<cell {i} again>", "exec"), g)
        print("\nDRYRUN OK:", out_dir)
    except Exception:
        ok = False
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", action="append", default=None, help="書き出し先（複数可）。既定: データフォルダ直下と notebooks_dir")
    ap.add_argument("--dryrun", action="store_true", help="RUN_LLM=False で全コードセルを実行して構文・経路を確認")
    args = ap.parse_args(argv)
    if args.dryrun:
        dryrun()
        return
    paths = args.out or [os.path.join(drive_dir(), NOTEBOOK_NAME), os.path.join(notebooks_dir(), NOTEBOOK_NAME)]
    write_notebook(paths)


if __name__ == "__main__":
    main()
