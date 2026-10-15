"""Qwen生成ノートブックのローカル検証（RUN_LLM=False）。
LLMを使わない部分（pilot/full設定・生成計画・検証ロジック・保存・昇格防止）を確認し、
特に「ラベル欠落を検出できるか」を人工ケースで試験する。"""
from __future__ import annotations

import importlib.metadata
import json, os, random, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import notebooks_dir  # noqa: E402

NB = os.environ.get("QWEN_NOTEBOOK_PATH", os.path.join(notebooks_dir(), "qwen_synth_pii_data_v1.ipynb"))
OUT = Path(tempfile.mkdtemp(prefix="privacy-masker-qwen-dryrun-"))
os.environ["PRIVACY_MASKER_DRIVE_BASE"] = str(OUT)
DRYRUN_RUN_MODE = os.environ.get("QWEN_DRYRUN_RUN_MODE", "pilot")
if DRYRUN_RUN_MODE not in {"pilot", "full"}:
    raise ValueError("QWEN_DRYRUN_RUN_MODEはpilotまたはfullにしてください")
DEFAULT_DRYRUN_TARGET_DOCS = "300" if DRYRUN_RUN_MODE == "pilot" else "12000"
DRYRUN_TARGET_DOCS = int(os.environ.get(
    "QWEN_DRYRUN_TARGET_DOCS", DEFAULT_DRYRUN_TARGET_DOCS
))
OFFICIAL_SENTINEL_PATH = OUT / "synthetic_pii_qwen_v1.jsonl"
OFFICIAL_SENTINEL = b'{"sentinel":"must-not-change"}\n'
OFFICIAL_SENTINEL_PATH.write_bytes(OFFICIAL_SENTINEL)

with open(NB, encoding="utf-8") as handle:
    nb = json.load(handle)
g = {"__name__": "__main__", "display": lambda x: print(str(x)[:400])}
try:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if (
            "# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in src
            or ("VLLM_VERSION =" in src and "pip\", \"install" in src)
        ):
            print(f"[skip] cell {i} (インストール)"); continue
        src = src.replace(
            'RUN_MODE = "pilot"', f'RUN_MODE = "{DRYRUN_RUN_MODE}"'
        )
        src = src.replace("RUN_LLM = True", "RUN_LLM = False")
        src = src.replace("RUN_VERIFIER = True", "RUN_VERIFIER = False")
        src = src.replace(
            "PILOT_DOCS = 300", f"PILOT_DOCS = {DRYRUN_TARGET_DOCS}"
        )
        src = src.replace(
            "FULL_DOCS = 12000", f"FULL_DOCS = {DRYRUN_TARGET_DOCS}"
        )
        print(f"[run ] cell {i}")
        exec(compile(src, f"<cell {i}>", "exec"), g)
        if "n_requests = " in src:   # 生成計画セルの直後にプロンプトを1つ確認
            pass
    print("\n=== 検証ロジックの試験（人工ケース）===", flush=True)
    validate = g["validate_document"]

    def with_valid_axis_plan(doc, *, length_key=None, density_key=None):
        """既存の人工ケースへ、対象ルール以外では通過する生成計画を付ける。"""
        prepared = {
            "text": doc["text"],
            "entities": [list(entity) for entity in doc["entities"]],
        }
        while len(prepared["text"]) < 20:
            prepared["text"] += "確認。"

        n_chars = len(prepared["text"])
        if length_key is None:
            for candidate, (minimum, maximum) in g["LENGTH_CHAR_RANGES"].items():
                if minimum <= n_chars <= maximum:
                    length_key = candidate
                    break
        if length_key is None:
            raise AssertionError(f"人工ケースの長さが計画範囲外です: {n_chars}")

        n_entities = len(prepared["entities"])
        if density_key is None:
            for candidate, (minimum, maximum) in g["DENSITY_ENTITY_RANGES"].items():
                if minimum <= n_entities <= maximum:
                    density_key = candidate
                    break
        if density_key is None:
            raise AssertionError(f"人工ケースのentity数が計画範囲外です: {n_entities}")

        prepared["plan"] = {
            "genre": "test",
            "length_key": length_key,
            "density_key": density_key,
        }
        return prepared

    CASES = [
        ("正常", {"text": "担当は樋渡アキです。", "entities": [[3, 7, "PERSON"]]}, True),
        ("ラベル欠落(組織)", {"text": "叶津工機株式会社より連絡がありました。",
                          "entities": []}, False),
        ("ラベル欠落(施設)", {"text": "みなも内科クリニックへ紹介します。", "entities": []}, False),
        ("ラベル欠落(人名+様)", {"text": "百目鬼様よりお電話がありました。", "entities": []}, False),
        ("ラベル欠落(住所)", {"text": "千葉県船橋市海神町に移転します。", "entities": []}, False),
        ("簡体字混入", {"text": "担当は樋渡アキ、这个案件を确认します。",
                    "entities": [[3, 7, "PERSON"]]}, False),
        ("offset不正", {"text": "短い", "entities": [[0, 99, "PERSON"]]}, False),
        ("スパン重複", {"text": "樋渡アキさん", "entities": [[0, 4, "PERSON"], [2, 6, "PERSON"]]}, False),
        ("マーカー痕跡", {"text": "担当は[[PERSON|樋渡]]です。", "entities": []}, False),
        ("正常(PIIなし負例)", {"text": "本日の定例会議は資料を事前共有してください。", "entities": []}, True),
        ("正常(組織はラベル済み)", {"text": "叶津工機株式会社より連絡がありました。",
                             "entities": [[0, 8, "ORG"]]}, True),
        ("正常(人名はラベル済み)", {"text": "百目鬼様よりお電話がありました。",
                            "entities": [[0, 3, "PERSON"]]}, True),
        ("誤検出しない(基本仕様)", {"text": "基本仕様の見直しを行いました。", "entities": []}, True),
        ("誤検出しない(叔父さん)", {"text": "叔父さんに相談してみます。", "entities": []}, True),
        ("誤検出しない(同様に)", {"text": "前回と同様に進めます。", "entities": []}, True),
        ("誤検出しない(皆様)", {"text": "皆様のご協力に感謝します。", "entities": []}, True),
    ]
    ng = 0
    for name, doc, expect_ok in CASES:
        doc = with_valid_axis_plan(doc)
        errs, warns = validate(doc)
        ok = not errs
        mark = "OK " if ok == expect_ok else "NG "
        if ok != expect_ok:
            ng += 1
        print(f"  {mark}{name:<22} 判定={'採用' if ok else '不採用'} {errs[:1]}")
    n_checks = len(CASES)

    def check(name, condition, detail=""):
        nonlocal_marker = "OK " if condition else "NG "
        print(f"  {nonlocal_marker}{name:<32} {detail}")
        return 0 if condition else 1

    print("\n=== CUDAバイナリ依存契約の試験 ===")
    install_cell = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        and "# DRYRUN_SKIP: COLAB_GPU_BINARY_INSTALL" in "".join(cell["source"])
    )

    def execute_install_cell(installed_versions, *, probe_failures=0):
        state = dict(installed_versions)
        commands = []
        probe_count = 0

        def mocked_version(package):
            version = state.get(package)
            if version is None:
                raise importlib.metadata.PackageNotFoundError(package)
            return version

        def mocked_check_call(args, **kwargs):
            commands.append(list(args))
            if list(args[:4]) == [sys.executable, "-m", "pip", "install"]:
                if "uv" in args:
                    state["uv"] = "0.8.12"
            if list(args[:3]) == ["uv", "pip", "install"]:
                state.update({
                    "vllm": "0.27.1",
                    "torch": "2.13.0+cu132",
                    "torchvision": "0.28.0+cu132",
                })
            return 0

        def mocked_run(args, **kwargs):
            commands.append(list(args))
            if list(args[:3]) == ["uv", "pip", "uninstall"]:
                for package in ("torchaudio", "torchcodec"):
                    state[package] = None
            return subprocess.CompletedProcess(args, 0)

        def mocked_check_output(args, **kwargs):
            nonlocal probe_count
            probe_count += 1
            if probe_count <= probe_failures:
                raise subprocess.CalledProcessError(
                    1, args, output="ImportError: libcudart.so.13 not found"
                )
            return json.dumps({
                "vllm": state.get("vllm"),
                "torch": state.get("torch"),
                "torchvision": state.get("torchvision"),
                "torch_cuda": "13.2",
            }) + "\nwarning: simulated post-import diagnostic\n"

        ready_path = OUT / "install-contract-ready.json"
        executable_cell = install_cell.replace(
            "/content/.qwen_pii_synth_v1_5_vllm_0_27_1_uv_auto.ready",
            str(ready_path),
        )
        namespace = {"__name__": "__main__"}
        with (
            mock.patch.object(importlib.metadata, "version", side_effect=mocked_version),
            mock.patch.object(subprocess, "check_call", side_effect=mocked_check_call),
            mock.patch.object(subprocess, "check_output", side_effect=mocked_check_output),
            mock.patch.object(subprocess, "run", side_effect=mocked_run),
            mock.patch.object(os, "kill") as kill_mock,
            mock.patch.object(time, "sleep"),
            mock.patch.dict(os.environ, {"COLAB_RELEASE_TAG": "dryrun"}),
        ):
            exec(compile(executable_cell, "<install contract>", "exec"), namespace)
        return namespace, state, commands, kill_mock.call_count, probe_count

    expected_binaries = {
        "vllm": "0.27.1",
        "torch": "2.13.0+cu132",
        "torchvision": "0.28.0+cu132",
        "torchaudio": None,
        "torchcodec": None,
        "uv": "0.8.12",
    }
    matched_namespace, matched_state, matched_calls, matched_kills, matched_probes = execute_install_cell(
        expected_binaries
    )
    matched_uv_installs = [
        args for args in matched_calls
        if list(args[:3]) == ["uv", "pip", "install"]
    ]
    ng += check(
        "依存一致: uv再導入・再起動なし",
        matched_namespace["probe"]["torch"] == "2.13.0+cu132"
        and not matched_uv_installs
        and matched_kills == 0
        and matched_probes == 1,
        f"uv_installs={len(matched_uv_installs)}, kills={matched_kills}, probes={matched_probes}",
    )
    n_checks += 1

    broken_binaries = {
        "vllm": "0.27.1",
        "torch": "2.13.0+cu129",
        "torchvision": "0.28.0+cu129",
        "torchaudio": "2.11.0+cu128",
        "torchcodec": "0.9.0",
        "uv": None,
    }
    broken_namespace, repaired_state, repair_calls, repair_kills, repair_probes = execute_install_cell(
        broken_binaries, probe_failures=1
    )
    uv_install_calls = [
        args for args in repair_calls
        if list(args[:3]) == ["uv", "pip", "install"]
    ]
    normal_command = uv_install_calls[0] if uv_install_calls else []
    forced_command = uv_install_calls[1] if len(uv_install_calls) > 1 else []
    ng += check(
        "依存不整合: uv autoで修復後1回再起動",
        len(uv_install_calls) == 2
        and "vllm==0.27.1" in normal_command
        and "--torch-backend=auto" in normal_command
        and forced_command.count("--reinstall-package") == 3
        and all(package in forced_command for package in ("vllm", "torch", "torchvision"))
        and repaired_state["torch"] == "2.13.0+cu132"
        and repaired_state["torchvision"] == "0.28.0+cu132"
        and repaired_state["torchaudio"] is None
        and repaired_state["torchcodec"] is None
        and repair_kills == 1
        and repair_probes == 2,
        f"uv_installs={len(uv_install_calls)}, kills={repair_kills}, probes={repair_probes}",
    )
    n_checks += 1

    print("\n=== 生成軸の実測値検証 ===")
    none_with_entity = with_valid_axis_plan(
        {"text": "担当は樋渡アキです。内容を確認しました。",
         "entities": [[3, 7, "PERSON"]]},
        density_key="none",
    )
    errors, _ = validate(none_with_entity)
    ng += check(
        "軸検証: none+entityを不採用",
        any("計画PII密度不一致: none=1件" in error for error in errors),
        str(errors[:2]),
    )
    n_checks += 1

    sparse_text, sparse_entities = g["segments_to_document"]([
        {"t": "担当者は"},
        {"t": "樋渡アキ", "label": "PERSON"},
        {"t": "、"},
        {"t": "楢崎絹恵", "label": "PERSON"},
        {"t": "、"},
        {"t": "海老江誠", "label": "PERSON"},
        {"t": "です。確認しました。"},
    ])
    sparse_out_of_range = with_valid_axis_plan(
        {"text": sparse_text, "entities": sparse_entities},
        density_key="sparse",
    )
    errors, _ = validate(sparse_out_of_range)
    ng += check(
        "軸検証: sparse範囲外を不採用",
        any("計画PII密度不一致: sparse=3件" in error for error in errors),
        str(errors[:2]),
    )
    n_checks += 1

    length_out_of_range = with_valid_axis_plan(
        {"text": "本日の定例会議は予定どおり開催します。", "entities": []},
        length_key="long",
        density_key="none",
    )
    errors, _ = validate(length_out_of_range)
    ng += check(
        "軸検証: length範囲外を不採用",
        any("計画長さ不一致: long=" in error for error in errors),
        str(errors[:2]),
    )
    n_checks += 1

    print("\n=== resume・verifier・品質ゲートの試験 ===")
    plan = g["generation_plan"][0]
    run_signature = g["RUN_SIGNATURE"]
    good_docs = [
        {
            "doc_id": f"{plan['request_id']}-{index:02d}",
            "text": f"文書{index}",
            "entities": [],
        }
        for index in range(plan["docs"])
    ]
    good_row = {
        "request_id": plan["request_id"],
        "run_signature": run_signature,
        "plan": plan,
        "prompt_sha256": "test",
        "error": "",
        "attempt": 1,
        "docs": good_docs,
    }
    is_complete = g["is_complete_generation_row"]
    resume_cases = [
        ("resume: 完全行だけ完了", good_row, True),
        ("resume: schema失敗を再試行", {**good_row, "error": "schema_error"}, False),
        ("resume: 2/3文書を再試行", {**good_row, "docs": good_docs[:2]}, False),
        ("resume: 異なる署名を拒否", {**good_row, "run_signature": "old"}, False),
    ]
    for name, row, expected in resume_cases:
        actual = is_complete(row)
        ng += check(name, actual == expected, f"判定={actual}")
        n_checks += 1

    terminal_generation = g["terminal_generation_rows_by_request"]
    exhausted_generation_row = {
        **good_row,
        "docs": [],
        "error": "schema_error after retry",
        "attempt": g["CFG"].max_structured_attempts,
        "retry_exhausted": True,
    }
    terminal_generation_rows = terminal_generation([exhausted_generation_row])
    ng += check(
        "resume: 再試行済み生成エラーを終端扱い",
        terminal_generation_rows.get(plan["request_id"])
        == exhausted_generation_row,
    )
    n_checks += 1
    legacy_exhausted_generation_row = {
        key: value for key, value in exhausted_generation_row.items()
        if key != "retry_exhausted"
    }
    terminal_generation_rows = terminal_generation(
        [legacy_exhausted_generation_row]
    )
    ng += check(
        "resume: 旧attempt=2生成エラーを終端扱い",
        terminal_generation_rows.get(plan["request_id"])
        == legacy_exhausted_generation_row,
    )
    n_checks += 1
    terminal_generation_rows = terminal_generation(
        [good_row, exhausted_generation_row]
    )
    ng += check(
        "resume: 成功行を後続errorより優先",
        terminal_generation_rows.get(plan["request_id"]) == good_row,
    )
    n_checks += 1

    duplicate_doc = {
        "text": "百目鬼は受付済み。百目鬼が後ほど伺います。",
        "entities": [[0, 3, "PERSON"]],
    }
    outside_positions = g["find_unlabeled_occurrences"](
        duplicate_doc["text"], "百目鬼", duplicate_doc["entities"]
    )
    expected_second = duplicate_doc["text"].rfind("百目鬼")
    ng += check(
        "verifier: 同一表層の2回目を検出",
        outside_positions == [expected_second],
        f"positions={outside_positions}",
    )
    n_checks += 1

    evaluate = g["evaluate_verifier_result"]
    status, missing, _ = evaluate(duplicate_doc, {
        "parsed": {"missing": [{"text": "百目鬼", "label": "PERSON"}], "ok": False},
        "error": "",
    })
    ng += check(
        "verifier: 未ラベル指摘を不採用",
        status == "missing" and missing[0]["positions"] == [expected_second],
        f"status={status}",
    )
    n_checks += 1

    status, _, _ = evaluate(duplicate_doc, {
        "parsed": None,
        "error": "schema_error",
    })
    ng += check("verifier: 応答失敗はfail-closed", status == "error", f"status={status}")
    n_checks += 1

    status, _, _ = evaluate(duplicate_doc, {
        "parsed": {"missing": [], "ok": False},
        "error": "",
    })
    ng += check("verifier: ok/missing矛盾を拒否", status == "error", f"status={status}")
    n_checks += 1

    schema_docs = g["GENERATION_SCHEMA"]["properties"]["documents"]
    ng += check(
        "schema: 1リクエスト3文書を強制",
        schema_docs["minItems"] == schema_docs["maxItems"] == 3,
        f"min/max={schema_docs['minItems']}/{schema_docs['maxItems']}",
    )
    n_checks += 1
    runtime_cell = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        and "from vllm import LLM, SamplingParams" in "".join(cell["source"])
        and "VLLM_IMPORT_SMOKE" in "".join(cell["source"])
    )
    flashinfer_assignment = (
        'os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = CFG.flashinfer_sampler'
    )
    vllm_import = "from vllm import LLM, SamplingParams"
    runtime_signature = g["RUN_SIGNATURE_PAYLOAD"].get("runtime", {})
    ng += check(
        "G4: FlashInfer samplerをimport前に無効化",
        g["CFG"].flashinfer_sampler == "0"
        and runtime_signature.get("flashinfer_sampler") == "0"
        and flashinfer_assignment in runtime_cell
        and runtime_cell.index(flashinfer_assignment) < runtime_cell.index(vllm_import),
    )
    n_checks += 1
    runtime_version_guard = "if CFG.run_llm and runtime_contract_errors:"
    explicit_torch_import = "    import torch"
    torchvision_import = "        import torchvision"
    torch_cuda_guard = "if not TORCH_CUDA_RUNTIME_VERSION.startswith(expected_cuda_prefix):"
    libcudart_probe = "ctypes.CDLL(CUDA_RUNTIME_LIBRARY)"
    ng += check(
        "runtime: CUDA 13/libcudartをvLLM前に検査",
        runtime_version_guard in runtime_cell
        and explicit_torch_import in runtime_cell
        and torchvision_import in runtime_cell
        and torch_cuda_guard in runtime_cell
        and libcudart_probe in runtime_cell
        and "import torchaudio" not in runtime_cell
        and "import torchcodec" not in runtime_cell
        and runtime_cell.index(runtime_version_guard) < runtime_cell.index("def gpu_inventory")
        and runtime_cell.index(explicit_torch_import) < runtime_cell.index(torch_cuda_guard)
        and runtime_cell.index(torch_cuda_guard) < runtime_cell.index(libcudart_probe)
        and runtime_cell.index(libcudart_probe) < runtime_cell.index(vllm_import),
    )
    n_checks += 1
    ng += check(
        "runtime: 参照Notebookの初期化条件を継承",
        g["CFG"].language_model_only is True
        and g["CFG"].generation_config == "vllm"
        and 'os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"' in runtime_cell
        and '"language_model_only": CFG.language_model_only' in runtime_cell
        and '"generation_config": CFG.generation_config' in runtime_cell
        and 'with set_env_var("VLLM_LOGGING_LEVEL", "DEBUG"):' in runtime_cell
        and "fallback_kwargs" not in runtime_cell,
    )
    n_checks += 1
    ng += check(
        "run署名: uv autoの実測構成を固定",
        g["CFG"].pipeline_version == "qwen_pii_synth_v1.5"
        and runtime_signature.get("dependency_resolver") == "uv"
        and runtime_signature.get("torch_backend") == "auto"
        and runtime_signature.get("required_torch_cuda_major") == "13"
        and runtime_signature.get("excluded_runtime_packages") == ["torchaudio", "torchcodec"]
        and runtime_signature.get("packages") == g["RUN_SIGNATURE_RUNTIME_PACKAGE_VERSIONS"]
        and "uv" not in runtime_signature.get("packages", {})
        and runtime_signature.get("torch_cuda") == g["RESOLVED_TORCH_CUDA_VERSION"]
        and runtime_signature.get("language_model_only") is True
        and runtime_signature.get("generation_config") == "vllm",
        str(runtime_signature),
    )
    n_checks += 1
    expected_output_dir = (
        "qwen_synth_v1_pilot" if DRYRUN_RUN_MODE == "pilot" else "qwen_synth_v1"
    )
    ng += check(
        "実行モード: 設定と件数",
        g["CFG"].run_mode == DRYRUN_RUN_MODE
        and g["CFG"].target_docs == DRYRUN_TARGET_DOCS
        and len(g["generation_plan"]) == -(-DRYRUN_TARGET_DOCS // 3),
        f"mode={g['CFG'].run_mode}, target={g['CFG'].target_docs}",
    )
    n_checks += 1
    ng += check(
        "実行モード: 出力と昇格を分離",
        g["CFG"].output_dir_name == expected_output_dir
        and g["OUT_DIR"] == OUT / expected_output_dir
        and g["CFG"].promote_final == (DRYRUN_RUN_MODE == "full"),
        f"out={g['OUT_DIR']}, promote={g['CFG'].promote_final}",
    )
    n_checks += 1
    ng += check(
        "RUN_LLM=False: 生成checkpointを汚染しない",
        not g["RAW_PATH"].exists()
        and not g["generation_residual_error_ids"]
        and len(g["generation_unprocessed_ids"]) == len(g["generation_plan"]),
        str({
            "raw_exists": g["RAW_PATH"].exists(),
            "residual": len(g["generation_residual_error_ids"]),
            "unprocessed": len(g["generation_unprocessed_ids"]),
        }),
    )
    n_checks += 1
    signature_payload = g["RUN_SIGNATURE_PAYLOAD"]
    ng += check(
        "run署名: mode・件数・出力・昇格を固定",
        signature_payload.get("run_mode") == DRYRUN_RUN_MODE
        and signature_payload.get("target_docs") == DRYRUN_TARGET_DOCS
        and signature_payload.get("output_dir_name") == expected_output_dir
        and signature_payload.get("promote_final") == (DRYRUN_RUN_MODE == "full"),
        str({key: signature_payload.get(key) for key in (
            "run_mode", "target_docs", "output_dir_name", "promote_final"
        )}),
    )
    n_checks += 1
    axis_pairs = {
        (plan["length_key"], plan["density_key"])
        for plan in g["generation_plan"]
    }
    ng += check(
        "生成計画: 長さ×密度を独立配分",
        len(axis_pairs) > 4,
        f"observed_pairs={len(axis_pairs)}",
    )
    n_checks += 1
    if DRYRUN_RUN_MODE == "pilot" and DRYRUN_TARGET_DOCS == 300:
        observed_genres = {plan["genre"] for plan in g["generation_plan"]}
        observed_lengths = {plan["length_key"] for plan in g["generation_plan"]}
        observed_densities = {plan["density_key"] for plan in g["generation_plan"]}
        observed_axes = {
            (plan["genre"], plan["length_key"], plan["density_key"])
            for plan in g["generation_plan"]
        }
        ng += check(
            "pilot計画: 300件の品質確認カバレッジ",
            len(g["generation_plan"]) == 100
            and len(observed_genres) == 24
            and len(observed_lengths) == 4
            and len(observed_densities) == 4
            and len(observed_axes) == 100,
            f"genres={len(observed_genres)}, axes={len(observed_axes)}",
        )
        n_checks += 1
    if len(g["generation_plan"]) >= len(g["AXIS_COMBINATIONS"]):
        observed_axes = {
            (plan["genre"], plan["length_key"], plan["density_key"])
            for plan in g["generation_plan"]
        }
        ng += check(
            "生成計画: 24×4×4の全組合せ",
            len(observed_axes) == len(g["AXIS_COMBINATIONS"]) == 384,
            f"observed={len(observed_axes)}",
        )
        n_checks += 1
    none_plans = [
        plan for plan in g["generation_plan"] if plan["density_key"] == "none"
    ]
    ng += check(
        "生成計画: PIIなし条件に矛盾なし",
        bool(none_plans)
        and all(plan["entity_mix"] is None and plan["post_context"] is None
                for plan in none_plans),
        f"none_plans={len(none_plans)}",
    )
    n_checks += 1
    none_prompt = g["build_generation_prompt"](none_plans[0], random.Random(0))
    ng += check(
        "プロンプト: PIIなし時はPII指示を省略",
        "PIIなし条件" in none_prompt
        and "エンティティの構成:" not in none_prompt
        and "必ず取り入れる書き方" not in none_prompt,
    )
    n_checks += 1

    verifier_doc = {**duplicate_doc, "doc_id": "verifier-001"}
    verifier_good_row = {
        "doc_id": verifier_doc["doc_id"],
        "verifier_signature": g["VERIFIER_SIGNATURE"],
        "doc_sha256": g["verifier_doc_sha256"](verifier_doc),
        "status": "ok",
        "verifier_ok": True,
        "missing": "[]",
        "error": "",
        "attempt": 1,
    }
    verifier_complete = g["is_complete_verifier_row"]
    ng += check(
        "verifier resume: 成功行だけ完了",
        verifier_complete(verifier_good_row, verifier_doc),
    )
    n_checks += 1
    ng += check(
        "verifier resume: 失敗行を再試行",
        not verifier_complete(
            {**verifier_good_row, "status": "error", "error": "schema_error"},
            verifier_doc,
        ),
    )
    n_checks += 1
    verifier_error_row = {
        **verifier_good_row,
        "status": "error",
        "verifier_ok": False,
        "error": "semantic_error",
    }
    terminal_verifier = g["terminal_verifier_rows_by_doc"]
    verifier_docs_by_id = {verifier_doc["doc_id"]: verifier_doc}
    ng += check(
        "verifier retry: 1回目errorは未終端",
        verifier_doc["doc_id"] not in terminal_verifier(
            [verifier_error_row], verifier_docs_by_id
        ),
    )
    n_checks += 1
    exhausted_verifier = terminal_verifier(
        [verifier_error_row, verifier_error_row], verifier_docs_by_id
    )
    ng += check(
        "verifier retry: 2回目errorを記録して終端",
        exhausted_verifier.get(verifier_doc["doc_id"], {}).get(
            "retry_exhausted"
        ) is True,
    )
    n_checks += 1
    legacy_exhausted_verifier_row = {**verifier_error_row, "attempt": 2}
    exhausted_verifier = terminal_verifier(
        [legacy_exhausted_verifier_row], verifier_docs_by_id
    )
    ng += check(
        "verifier retry: 旧attempt=2単一行を終端扱い",
        exhausted_verifier.get(verifier_doc["doc_id"], {}).get(
            "retry_exhausted"
        ) is True,
    )
    n_checks += 1
    verifier_success_preferred = terminal_verifier(
        [verifier_good_row, verifier_error_row, verifier_error_row],
        verifier_docs_by_id,
    )
    ng += check(
        "verifier retry: 成功行を後続errorより優先",
        verifier_success_preferred.get(verifier_doc["doc_id"])
        == verifier_good_row,
    )
    n_checks += 1
    verifier_retry_cell = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        and "run_verifier_docs_with_retries" in "".join(cell["source"])
    )
    ng += check(
        "verifier retry: schema/意味を合計2回に制限",
        g["CFG"].max_verifier_semantic_attempts == 2
        and "max_attempts=1" in verifier_retry_cell,
    )
    n_checks += 1
    ng += check(
        "verifier checkpoint: 一時出力へ隔離",
        g["VERIFIER_PATH"].parent == g["OUT_DIR"],
        str(g["VERIFIER_PATH"]),
    )
    n_checks += 1
    ng += check("出力先の差し替え: 一時出力へ隔離", g["DRIVE_BASE"] == OUT, str(g["DRIVE_BASE"]))
    n_checks += 1
    ng += check("品質ゲート: 0件を拒否", bool(g["quality_errors"]), str(g["quality_errors"][:1]))
    n_checks += 1
    manifest = g["manifest"]
    promotion = manifest["promotion"]
    manifest_outputs = manifest["outputs"]
    manifest_software = manifest["software"]
    ng += check(
        "manifest: uv autoの実測ランタイムを記録",
        {
            "dependency_resolver", "torch_backend", "excluded_runtime_packages",
            "packages", "vllm", "torch", "torchvision", "torchaudio",
            "torchcodec", "uv", "torch_cuda", "cuda_runtime_library",
            "vllm_import_smoke", "language_model_only", "generation_config",
        } <= set(manifest_software)
        and manifest_software["dependency_resolver"] == "uv"
        and manifest_software["torch_backend"] == "auto"
        and manifest_software["packages"] == g["RUNTIME_BINARY_VERSIONS"]
        and manifest_software["excluded_runtime_packages"] == ["torchaudio", "torchcodec"]
        and manifest_software["language_model_only"] is True
        and manifest_software["generation_config"] == "vllm",
        str(manifest_software),
    )
    n_checks += 1
    ng += check(
        "manifest: mode・昇格可否を記録",
        manifest["run_mode"] == DRYRUN_RUN_MODE
        and manifest["quality_gate"]["applicable"] == (DRYRUN_RUN_MODE == "full")
        and promotion["allowed"] == (DRYRUN_RUN_MODE == "full")
        and promotion["eligible"] is False
        and promotion["performed"] is False,
        str({"run_mode": manifest["run_mode"], "promotion": promotion}),
    )
    n_checks += 1
    ng += check(
        "manifest: LLM総試行と失敗試行を分離",
        {
            "generation_llm_attempts",
            "failed_attempts",
            "verifier_llm_attempts",
            "verifier_failed_attempts",
        } <= set(manifest["counts"]),
        str(manifest["counts"]),
    )
    n_checks += 1
    residual_errors = manifest.get("residual_errors", {})
    ng += check(
        "manifest: 残存エラー方針と除外を記録",
        residual_errors.get("policy") == "record_exclude_continue"
        and residual_errors.get("excluded_from_candidate") is True
        and {
            "generation_request_ids",
            "generation_unprocessed_ids",
            "verifier_doc_ids",
            "verifier_unprocessed_doc_ids",
        } <= set(residual_errors),
        str(residual_errors),
    )
    n_checks += 1
    ng += check(
        "manifest: pilotの正式出力を誤認させない",
        manifest_outputs["final_jsonl"]
        == (str(g["FINAL_PATH"]) if DRYRUN_RUN_MODE == "full" else None)
        and manifest_outputs["official_final_path"] == str(g["FINAL_PATH"])
        and g["MANIFEST_PATH"].parent == g["OUT_DIR"],
        str({
            "final_jsonl": manifest_outputs["final_jsonl"],
            "official_final_path": manifest_outputs["official_final_path"],
        }),
    )
    n_checks += 1
    quality_cell = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        and "build_quality_gate_errors" in "".join(cell["source"])
    )
    ng += check(
        "昇格分岐: pilotを品質エラーより優先",
        'elif CFG.run_mode == "pilot":' in quality_cell
        and quality_cell.index('elif CFG.run_mode == "pilot":')
        < quality_cell.index("elif quality_errors:"),
    )
    n_checks += 1
    quality_failure_branch = quality_cell.split("elif quality_errors:", 1)[1].split(
        "else:", 1
    )[0]
    ng += check(
        "品質ゲート: 不通過でもcandidate保存後に正常終了",
        "raise RuntimeError" not in quality_failure_branch
        and "正式データは更新していません" in quality_failure_branch,
    )
    n_checks += 1
    stage_b_cell = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        and "def validate_document" in "".join(cell["source"])
    )
    ng += check(
        "QC: 生成残存エラーもrequest単位で記録",
        "generation_residual_error_ids" in stage_b_cell
        and "生成失敗（自動再試行済み）" in stage_b_cell
        and '"request_id": request_id' in stage_b_cell,
    )
    n_checks += 1
    ng += check(
        "正式JSONL: dry-runで既存内容を不変",
        g["FINAL_PATH"] == OFFICIAL_SENTINEL_PATH
        and OFFICIAL_SENTINEL_PATH.read_bytes() == OFFICIAL_SENTINEL,
        str(g["FINAL_PATH"]),
    )
    n_checks += 1
    ng += check("候補JSONL: QC用に作成", g["CANDIDATE_PATH"].exists(), str(g["CANDIDATE_PATH"]))
    n_checks += 1
    ng += check(
        "目視サンプルCSV: QC用に作成",
        g["REVIEW_SAMPLE_PATH"].exists()
        and g["REVIEW_SAMPLE_PATH"].parent == g["OUT_DIR"],
        str(g["REVIEW_SAMPLE_PATH"]),
    )
    n_checks += 1

    print(f"\n検証ロジック試験: {n_checks - ng}/{n_checks} 期待どおり")

    print("\n=== 近重複除去の試験 ===")
    dedup = g["drop_near_duplicates"]
    docs = [
        {"text": "担当は樋渡アキです。よろしくお願いします。", "entities": [[3, 7, "PERSON"]]},
        {"text": "担当は楢崎絹恵です。よろしくお願いします。", "entities": [[3, 7, "PERSON"]]},
        {"text": "本日の会議は中止となりました。次回は来週です。", "entities": []},
    ]
    kept, dropped = dedup(docs, 0.92)
    dedup_ok = len(kept) == 2 and dropped == 1
    ng += 0 if dedup_ok else 1
    print(f"  {'OK' if dedup_ok else 'NG'} 3件中 残り{len(kept)}件 / 除去{dropped}件")
    sys.stdout.flush()
    exit_code = 0 if ng == 0 else 1
    shutil.rmtree(OUT, ignore_errors=True)
    os._exit(exit_code)
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    shutil.rmtree(OUT, ignore_errors=True)
    os._exit(1)
