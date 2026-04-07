import json
import subprocess
import os
import sys
import threading
from datetime import datetime
import gradio as gr
from ui.common import *
from .setting import cnfg


# ─────────────────────────────
# グローバルプロセス管理（学習・前処理の排他制御）
# ─────────────────────────────
_proc_lock = threading.Lock()
_active_log_path: Path | None = None
_active_proc: subprocess.Popen | None = None

# ─────────────────────────────
# Prepare Manifest タブ ロジック
# ─────────────────────────────


def _read_csv_headers(file_path: str) -> list[str]:
    import csv as _csv
    import json as _json

    p = Path(file_path.strip()) if file_path else None
    if not p:
        return []
    try:
        if p.is_dir():
            for name in ("metadata.csv", "metadata.jsonl", "metadata.json"):
                candidate = p / name
                if candidate.is_file():
                    return _read_csv_headers(str(candidate))
            return []

        if not p.is_file():
            return []

        suffix = p.suffix.lower()
        if suffix == ".csv":
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                headers = next(_csv.reader(f), [])
            return [h.strip() for h in headers if h.strip()]

        elif suffix in {".jsonl", ".json"}:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    obj = _json.loads(line)
                    return list(obj.keys())
    except Exception:
        pass
    return []


def _preview_dataset(dataset: str, split: str, audio_col: str, text_col: str) -> str:
    dataset = str(dataset).strip()
    if not dataset:
        return "データセットを入力してください。"

    p = Path(dataset)
    if p.is_file() and p.suffix in {".jsonl", ".json"}:
        try:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            count = len(lines)
            previews = []
            for line in lines[:3]:
                try:
                    obj = json.loads(line)
                    text = obj.get(text_col, "（テキストなし）")
                    previews.append(f"  • {str(text)[:80]}")
                except Exception:
                    previews.append("  • （パース失敗）")
            preview_str = "\n".join(previews)
            return f"✅ ローカルJSONL: {count} 件\n\n【サンプル（最大3件）】\n{preview_str}"
        except Exception as e:
            return f"❌ ファイル読み込みエラー: {e}"

    if p.is_dir():
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = [f for f in p.rglob("*") if f.suffix.lower() in audio_exts]
        count = len(files)
        previews = [f"  • {f.name}" for f in sorted(files)[:3]]
        preview_str = "\n".join(previews) if previews else "  （ファイルなし）"
        return f"✅ ローカルフォルダ: 音声ファイル {count} 件\n\n【サンプル（最大3件）】\n{preview_str}"

    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(dataset)
        info = builder.info
        splits_info = info.splits or {}
        split_info = splits_info.get(split)
        count = split_info.num_examples if split_info else "不明"
        name = info.dataset_name or dataset
        desc = (info.description or "")[:120]
        return (
            f"✅ HuggingFace Dataset: {name}\n"
            f"スプリット '{split}': {count} 件\n\n"
            f"【説明】\n{desc}..."
        )
    except ImportError:
        return "⚠️ `datasets` ライブラリが未インストールです。\n`pip install datasets` を実行してください。"
    except Exception as e:
        return f"⚠️ データセット情報の取得に失敗しました:\n{e}"


def _build_manifest_command(
    data_source_mode,
    dataset,
    split,
    prepare_mode,
    audio_col,
    text_col,
    speaker_col,
    caption_col,
    output_manifest,
    latent_dir,
    device,
    codec_repo,
) -> list[str]:
    def _s(val, fallback=""):
        if val is None or isinstance(val, (dict, list)):
            return fallback
        s = str(val).strip()
        return s if s else fallback

    cmd = [sys.executable, str(cnfg.repo_dir / "prepare_manifest.py")]

    if data_source_mode == "local_csv":
        csv_path = Path(_s(dataset))
        folder = str(csv_path.parent if csv_path.suffix.lower() == ".csv" else csv_path)
        cmd += ["--dataset", "audiofolder", "--data-files", folder, "--split", "train"]
    elif data_source_mode == "local_jsonl":
        cmd += ["--dataset", "json", "--data-files", _s(dataset), "--split", "train"]
    else:
        cmd += ["--dataset", _s(dataset), "--split", _s(split, "train")]

    mode = str(prepare_mode).strip().lower()
    auto_codec_repo = str(codec_repo).strip()
    if mode in {"model_v2", "voice_design"}:
        auto_codec_repo = "Aratako/Semantic-DACVAE-Japanese-32dim"
    elif mode == "model_v1":
        auto_codec_repo = "facebook/dacvae-watermarked"
    if not auto_codec_repo:
        auto_codec_repo = cnfg.default_prepare_codec_repo

    cmd += [
        "--audio-column",
        _s(audio_col, "audio"),
        "--text-column",
        _s(text_col, "text"),
        "--output-manifest",
        _s(output_manifest),
        "--latent-dir",
        _s(latent_dir),
        "--device",
        _s(device, "cpu"),
        "--codec-repo",
        auto_codec_repo,
    ]
    if mode == "voice_design":
        cap = _s(caption_col)
        if cap:
            cmd += ["--caption-column", cap]
    else:
        spk = _s(speaker_col)
        if spk:
            cmd += ["--speaker-column", spk]
    return cmd


def _manifest_cmd_preview(
    data_source_mode,
    dataset,
    split,
    prepare_mode,
    audio_col,
    text_col,
    speaker_col,
    caption_col,
    output_manifest,
    latent_dir,
    device,
    codec_repo,
) -> str:
    return " ".join(
        _build_manifest_command(
            data_source_mode,
            dataset,
            split,
            prepare_mode,
            audio_col,
            text_col,
            speaker_col,
            caption_col,
            output_manifest,
            latent_dir,
            device,
            codec_repo,
        )
    )


def _run_manifest(
    data_source_mode,
    dataset,
    split,
    prepare_mode,
    audio_col,
    text_col,
    speaker_col,
    caption_col,
    output_manifest,
    latent_dir,
    device,
    codec_repo,
) -> tuple[str, str]:
    global _active_proc, _active_log_path
    cmd_list = _build_manifest_command(
        data_source_mode,
        dataset,
        split,
        prepare_mode,
        audio_col,
        text_col,
        speaker_col,
        caption_col,
        output_manifest,
        latent_dir,
        device,
        codec_repo,
    )
    cmd_str = " ".join(cmd_list)
    with _proc_lock:
        if _active_proc is not None and _active_proc.poll() is None:
            return "別のプロセスが実行中です。停止してから再実行してください。", cmd_str

        cnfg.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = cnfg.logs_dir / f"manifest_{stamp}.log"
        _active_log_path = log_path

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        _active_proc = proc

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:  # type: ignore
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()
    return f"実行開始 (PID {proc.pid})\nログ: {log_path}", cmd_str


def _stop_process() -> str:
    global _active_proc
    with _proc_lock:
        if _active_proc is None or _active_proc.poll() is not None:
            return "実行中のプロセスはありません。"
        _active_proc.terminate()
        return f"プロセス (PID {_active_proc.pid}) に停止シグナルを送信しました。"


def _read_manifest_log() -> str:
    global _active_log_path, _active_proc
    if _active_log_path is None or not _active_log_path.exists():
        return ""
    text = _active_log_path.read_text(encoding="utf-8", errors="replace")
    if _active_proc is not None and _active_proc.poll() is not None:
        rc = _active_proc.returncode
        text += f"\n\n--- プロセス終了 (returncode={rc}) ---"
    return text


# ─────────────────────────────
# UI 生成
# ─────────────────────────────


def build(ctx):
    with gr.Tab("📂 Prepare Manifest"):
        gr.Markdown(
            "## データセット前処理\n"
            "音声データセットをDACVAEラテントに変換し、学習用マニフェスト（JSONL）を生成します。\n\n"
            "> **ローカルCSV/JSONLを使う場合**：Dataset作成タブで生成した `metadata.csv` または `metadata.jsonl` を直接指定できます。"
        )

        pm_data_source = gr.Radio(
            label="データソース",
            choices=["ローカルCSV", "ローカルJSONL", "HuggingFaceデータセット"],
            value="ローカルCSV",
        )
        pm_prepare_mode = gr.Dropdown(
            label="モード",
            choices=["model_v1", "model_v2", "voice_design"],
            value="model_v2",
            info="model_v1=dim128, model_v2=dim32, voice_design=dim32 + caption列",
        )

        with gr.Group() as pm_local_group:
            pm_dataset = gr.Textbox(
                label="ファイルパス（CSV または JSONL）",
                value=str(cnfg.default_dataset_dir / "metadata.csv"),
                placeholder=str(cnfg.default_dataset_dir / "metadata.csv"),
                info="Dataset作成タブで生成したmetadata.csv / metadata.jsonlを指定",
            )

        with gr.Group() as pm_hf_group:
            with gr.Row():
                pm_hf_name = gr.Textbox(
                    label="HuggingFaceデータセット名",
                    placeholder="例: myorg/my_dataset",
                    visible=False,
                )
                pm_split = gr.Textbox(
                    label="スプリット名",
                    value="train",
                    visible=False,
                )

        with gr.Accordion("列名設定", open=True):
            gr.Markdown(
                "**ローカルCSV / JSONL（audiofolder形式）の列名**\n"
                "- 音声列: `audio` 固定（CSV・JSONL の `file_name` 列を audiofolder が読み込み時に `audio` へ自動置換）\n"
                "- テキスト列・話者ID列: ファイルパス入力後にヘッダーを自動取得してドロップダウンに反映します。\n\n"
                "HuggingFace データセットの場合はそれぞれの列名を手動で入力してください。"
            )
            with gr.Row():
                pm_audio_col = gr.Dropdown(
                    label="音声列名", value="audio", choices=["audio"], allow_custom_value=True
                )
                pm_text_col = gr.Dropdown(
                    label="テキスト列名", value="text", choices=["text"], allow_custom_value=True
                )
                pm_speaker_col = gr.Dropdown(
                    label="話者ID列名（省略可）", value="", choices=[""], allow_custom_value=True
                )
                pm_caption_col = gr.Dropdown(
                    label="Caption列名（Voice Design）",
                    value="caption",
                    choices=["caption"],
                    allow_custom_value=True,
                    visible=False,
                )
            pm_col_status = gr.Textbox(label="列名取得状況", interactive=False, lines=1)

        with gr.Row():
            pm_output_manifest = gr.Textbox(
                label="出力マニフェストパス（.jsonl）",
                value=str(cnfg.base_dir / "data" / "train_manifest.jsonl"),
            )
            pm_latent_dir = gr.Textbox(
                label="ラテント保存フォルダ",
                value=str(cnfg.base_dir / "data" / "latents"),
            )
            pm_device = gr.Dropdown(
                label="使用デバイス",
                choices=ctx.device_choices,
                value=ctx.default_model_device,
            )

        pm_codec_repo = gr.Dropdown(
            label="DACVAE codec",
            choices=cnfg.prepare_codec_repo_choices,
            value=cnfg.default_prepare_codec_repo,
            allow_custom_value=True,
            info="v2(dim32) default / switch to v1(dim128) as needed.",
        )

        def _on_pm_source_change(mode):
            is_hf = mode == "HuggingFaceデータセット"
            return (
                gr.update(visible=not is_hf),
                gr.update(visible=is_hf),
                gr.update(visible=is_hf),
                gr.update(value="audio"),
            )

        pm_data_source.change(
            _on_pm_source_change,
            inputs=[pm_data_source],
            outputs=[pm_dataset, pm_hf_name, pm_split, pm_audio_col],
        )

        def _on_pm_prepare_mode_change(mode: str):
            mode_key = str(mode).strip().lower()
            is_voice = mode_key == "voice_design"
            codec_repo = (
                "facebook/dacvae-watermarked"
                if mode_key == "model_v1"
                else "Aratako/Semantic-DACVAE-Japanese-32dim"
            )
            status = (
                "voice_design: caption列を使用 / codec dim32"
                if is_voice
                else (
                    "model_v1: speaker_id列を使用 / codec dim128"
                    if mode_key == "model_v1"
                    else "model_v2: speaker_id列を使用 / codec dim32"
                )
            )
            return (
                gr.update(visible=not is_voice),
                gr.update(visible=is_voice),
                gr.update(value=codec_repo),
                status,
            )

        pm_prepare_mode.change(
            _on_pm_prepare_mode_change,
            inputs=[pm_prepare_mode],
            outputs=[pm_speaker_col, pm_caption_col, pm_codec_repo, pm_col_status],
        )

        def _auto_fill_columns(file_path: str, mode: str):
            if mode == "HuggingFaceデータセット":
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "HFデータセット: 列名を手動で入力してください。",
                )
            headers = _read_csv_headers(file_path)
            if not headers:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "⚠️ 列名を取得できませんでした。ファイルパスを確認してください。",
                )
            audio_choices = ["audio"] + [h for h in headers if h not in {"audio", "file_name"}]
            exclude = {"file_name", "audio", "speaker_id", "speaker"}
            text_choices = [h for h in headers if h not in {"file_name", "audio"}]
            text_guess = next(
                (h for h in headers if h not in exclude),
                text_choices[0] if text_choices else "text",
            )
            spk_choices = [""] + [h for h in headers if h not in {"file_name", "audio", text_guess}]
            cap_choices = [""] + [h for h in headers if h not in {"file_name", "audio", text_guess}]
            spk_default = "speaker_id" if "speaker_id" in headers else ""
            cap_default = "caption" if "caption" in headers else ""
            status = f"✅ 列名を取得しました: {headers}"
            return (
                gr.update(choices=audio_choices, value="audio"),
                gr.update(choices=text_choices, value=text_guess),
                gr.update(choices=spk_choices, value=spk_default),
                gr.update(choices=cap_choices, value=cap_default),
                status,
            )

        pm_dataset.change(
            _auto_fill_columns,
            inputs=[pm_dataset, pm_data_source],
            outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_caption_col, pm_col_status],
        )
        pm_data_source.change(
            _auto_fill_columns,
            inputs=[pm_dataset, pm_data_source],
            outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_caption_col, pm_col_status],
        )

        pm_cmd_preview = gr.Textbox(label="実行コマンドプレビュー", interactive=False, lines=3)

        def _get_pm_inputs_values(
            mode,
            local_path,
            hf_name,
            split,
            prepare_mode,
            audio_col,
            text_col,
            speaker_col,
            caption_col,
            output_manifest,
            latent_dir,
            device,
            codec_repo,
        ):
            src_mode = {
                "ローカルCSV": "local_csv",
                "ローカルJSONL": "local_jsonl",
                "HuggingFaceデータセット": "hf_dataset",
            }.get(mode, "local_csv")
            dataset = hf_name if mode == "HuggingFaceデータセット" else local_path
            return (
                src_mode,
                dataset,
                split,
                prepare_mode,
                audio_col,
                text_col,
                speaker_col,
                caption_col,
                output_manifest,
                latent_dir,
                device,
                codec_repo,
            )

        _pm_all_inputs = [
            pm_data_source,
            pm_dataset,
            pm_hf_name,
            pm_split,
            pm_prepare_mode,
            pm_audio_col,
            pm_text_col,
            pm_speaker_col,
            pm_caption_col,
            pm_output_manifest,
            pm_latent_dir,
            pm_device,
            pm_codec_repo,
        ]

        def _update_pm_cmd(
            mode,
            local_path,
            hf_name,
            split,
            prepare_mode,
            audio_col,
            text_col,
            speaker_col,
            caption_col,
            output_manifest,
            latent_dir,
            device,
            codec_repo,
        ):
            args = _get_pm_inputs_values(
                mode,
                local_path,
                hf_name,
                split,
                prepare_mode,
                audio_col,
                text_col,
                speaker_col,
                caption_col,
                output_manifest,
                latent_dir,
                device,
                codec_repo,
            )
            return _manifest_cmd_preview(*args)

        for comp in _pm_all_inputs:
            comp.change(_update_pm_cmd, inputs=_pm_all_inputs, outputs=[pm_cmd_preview])

        with gr.Row():
            pm_run_btn = gr.Button("実行", variant="primary")
            pm_stop_btn = gr.Button("停止", variant="stop")
            pm_log_btn = gr.Button("ログ更新")

        pm_status = gr.Textbox(label="実行状況", interactive=False, lines=2)
        pm_log = gr.Textbox(label="ログ出力", interactive=False, lines=20, max_lines=20)

        def _run_manifest_ui(
            mode,
            local_path,
            hf_name,
            split,
            prepare_mode,
            audio_col,
            text_col,
            speaker_col,
            caption_col,
            output_manifest,
            latent_dir,
            device,
            codec_repo,
        ):
            args = _get_pm_inputs_values(
                mode,
                local_path,
                hf_name,
                split,
                prepare_mode,
                audio_col,
                text_col,
                speaker_col,
                caption_col,
                output_manifest,
                latent_dir,
                device,
                codec_repo,
            )
            return _run_manifest(*args)

        pm_run_btn.click(
            _run_manifest_ui, inputs=_pm_all_inputs, outputs=[pm_status, pm_cmd_preview]
        )
        pm_stop_btn.click(_stop_process, outputs=[pm_status])
        pm_log_btn.click(_read_manifest_log, outputs=[pm_log])
