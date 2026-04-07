from __future__ import annotations
import os
import sys
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

import gradio_conf as cnf

from ui.common import *
from ui.context import UIContext


# ─────────────────────────────
# Dataset Tools タブ ロジック
# ─────────────────────────────

_DS_LOG_PATH: Path | None = None
_DS_PROC: subprocess.Popen | None = None
_DS_LOG_LOCK = threading.Lock()


def _build_dataset_command(
    mode: str,
    input_path: str,
    slice_output: str,
    min_sec: float,
    max_sec: float,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
    target_sr_enabled: bool,
    target_sr: int,
    recursive_slice: bool,
    caption_input: str,
    manifest_output_dir: str,
    manifest_filename: str,
    output_format: str,
    whisper_model: str,
    language: str,
    speaker_id: str,
    recursive_caption: bool,
    device: str,
    model_cache_dir: str,
) -> list[str]:
    fmt_ext = "csv" if output_format == "CSV" else "jsonl"
    manifest_path = str(Path(manifest_output_dir) / f"{manifest_filename}.{fmt_ext}")

    if mode == "スライスのみ":
        cmd = [sys.executable, str(cnf.DATASET_TOOLS), "slice",
               "--input", input_path,
               "--output", slice_output,
               "--min-sec", str(min_sec),
               "--max-sec", str(max_sec),
               "--threshold", str(threshold),
               "--min-silence-ms", str(int(min_silence_ms)),
               "--speech-pad-ms",  str(int(speech_pad_ms)),
               ]
        if target_sr_enabled and int(target_sr) > 0:
            cmd += ["--target-sr", str(int(target_sr))]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if recursive_slice:
            cmd += ["--recursive"]

    elif mode == "キャプションのみ":
        lang = "" if language in ("自動検出", "auto", "") else language
        cmd = [sys.executable, str(cnf.DATASET_TOOLS), "caption",
               "--input", caption_input,
               "--output-manifest", manifest_path,
               "--format", fmt_ext,
               "--model", whisper_model,
               ]
        if lang:
            cmd += ["--language", lang]
        if str(speaker_id).strip():
            cmd += ["--speaker-id", str(speaker_id).strip()]
        if recursive_caption:
            cmd += ["--recursive"]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if str(model_cache_dir).strip():
            cmd += ["--model-cache-dir", str(model_cache_dir).strip()]

    else:
        lang = "" if language in ("自動検出", "auto", "") else language
        cmd = [sys.executable, str(cnf.DATASET_TOOLS), "pipeline",
               "--input", input_path,
               "--slice-output", slice_output,
               "--output-manifest", manifest_path,
               "--format", fmt_ext,
               "--min-sec", str(min_sec),
               "--max-sec", str(max_sec),
               "--threshold", str(threshold),
               "--min-silence-ms", str(int(min_silence_ms)),
               "--speech-pad-ms",  str(int(speech_pad_ms)),
               "--model", whisper_model,
               ]
        if target_sr_enabled and int(target_sr) > 0:
            cmd += ["--target-sr", str(int(target_sr))]
        if lang:
            cmd += ["--language", lang]
        if str(speaker_id).strip():
            cmd += ["--speaker-id", str(speaker_id).strip()]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if str(model_cache_dir).strip():
            cmd += ["--model-cache-dir", str(model_cache_dir).strip()]

    return cmd


def _start_dataset_job(*args) -> tuple[str, str]:
    global _DS_LOG_PATH, _DS_PROC

    with _DS_LOG_LOCK:
        if _DS_PROC is not None and _DS_PROC.poll() is None:
            return "別のジョブが実行中です。停止してから再実行してください。", ""

    cmd_list = _build_dataset_command(*args)
    cmd_str  = " ".join(cmd_list)

    cnf.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cnf.LOGS_DIR / f"dataset_{stamp}.log"

    with _DS_LOG_LOCK:
        _DS_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _DS_PROC = proc

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout: # type: ignore
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()
    return f"実行開始 (PID {proc.pid})\nログ: {log_path}", cmd_str


def _stop_dataset_job() -> str:
    global _DS_PROC
    with _DS_LOG_LOCK:
        if _DS_PROC is None or _DS_PROC.poll() is not None:
            return "実行中のジョブはありません。"
        _DS_PROC.terminate()
        return f"ジョブ (PID {_DS_PROC.pid}) に停止シグナルを送信しました。"


def _read_dataset_log() -> str:
    with _DS_LOG_LOCK:
        path = _DS_LOG_PATH
        proc = _DS_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        rc = proc.returncode
        text += f"\n\n--- 完了 (returncode={rc}) ---"
    lines = text.splitlines()
    if len(lines) > 300:
        text = "... （先頭省略、末尾300行表示）\n" + "\n".join(lines[-300:])
    return text




# ─────────────────────────────
# 絵文字キャプション ロジック
# ─────────────────────────────

_EMOJI_API_KEYS = {
    "LM Studio（ローカル）": "lm_studio",
    "Groq": "groq",
    "OpenAI（ChatGPT）": "openai",
    "Together AI": "together",
}
_EMOJI_DEFAULT_MODELS = {
    "lm_studio": "",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}


def _append_emoji_to_ds_log(log_path: Path|None, message: str) -> None:
    if log_path != None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
                f.flush()
        except Exception:
            pass


def _run_emoji_caption_inline(
    csv_path: str,
    wav_dir: str,
    api_label: str,
    api_key: str = "",
) -> None:
    global _DS_PROC, _DS_LOG_PATH

    api_key_str = _EMOJI_API_KEYS.get(api_label, "lm_studio")

    cmd = [
        sys.executable, str(cnf.DATASET_TOOLS), "emoji_caption",
        "--csv",     str(csv_path).strip(),
        "--wav-dir", str(wav_dir).strip(),
        "--api",     api_key_str,
    ]
    if api_key_str != "lm_studio" and str(api_key).strip():
        cmd += ["--api-key", str(api_key).strip()]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with _DS_LOG_LOCK:
        log_path = _DS_LOG_PATH
        _append_emoji_to_ds_log(log_path, f"\n{'='*60}")
        _append_emoji_to_ds_log(log_path, "🎭 絵文字キャプション開始")
        _append_emoji_to_ds_log(log_path, f"{'='*60}")
        proc = subprocess.Popen(
            cmd, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _DS_PROC = proc

    def _stream():
        with open(log_path, "a", encoding="utf-8") as f: # type: ignore
            for line in proc.stdout: # type: ignore
                f.write(line)
                f.flush()
        proc.wait()
        rc = proc.returncode
        _append_emoji_to_ds_log(log_path, f"\n--- 絵文字キャプション完了 (returncode={rc}) ---")

    threading.Thread(target=_stream, daemon=True).start()

# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx: UIContext):
    with gr.Tab("🎙️ Dataset作成"):
        gr.Markdown(
            "## データセット作成\n"
            "長尺音声の**無音区間スライス**と**Whisperキャプション**を行い、"
            "学習用 manifest（CSV / JSONL）を生成します。\n\n"
            "> 必要ライブラリ: `pip install librosa soundfile faster-whisper`"
        )

        ds_mode = gr.Radio(
            label="実行モード",
            choices=["スライスのみ", "キャプションのみ", "パイプライン（スライス→キャプション）"],
            value="パイプライン（スライス→キャプション）",
        )

        with gr.Accordion("スライス設定", open=True) as slice_accordion:
            gr.Markdown(
                "*Silero VAD（ニューラルネット音声活動検出）で発話区間を検出してスライスします。連続発話・キャラクター音声に対応。*"
            )
            with gr.Row():
                ds_input = gr.Textbox(
                    label="入力パス（ファイルまたはフォルダ）",
                    value=str(cnf.BASE_DIR / "input"),
                    placeholder="/path/to/audio_or_folder",
                    scale=3,
                )
                ds_recursive_slice = gr.Checkbox(label="サブフォルダも検索", value=False, scale=1)
            ds_slice_output = gr.Textbox(
                label="スライス済み音声の保存先フォルダ",
                value=str(cnf.DEFAULT_DATASET_DIR),
                placeholder=str(cnf.DEFAULT_DATASET_DIR),
            )
            with gr.Row():
                ds_min_sec = gr.Number(
                    label="最小セグメント長（秒）", value=2.0, info="これより短いセグメントは破棄"
                )
                ds_max_sec = gr.Number(
                    label="最大セグメント長（秒）",
                    value=30.0,
                    info="超えた場合は最近傍の無音点で分割",
                )
                ds_top_db = gr.Slider(
                    label="VAD 発話判定閾値（threshold）",
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    info="大きいほど厳しく検出（0.5推奨）",
                )
            with gr.Row():
                ds_frame_length = gr.Number(
                    label="無音最短継続時間（ms）",
                    value=300,
                    precision=0,
                    info="この時間以上の無音でないと区切らない",
                )
                ds_hop_length = gr.Number(
                    label="発話前後パディング（ms）",
                    value=30,
                    precision=0,
                    info="発話区間の前後に追加する余白",
                )
            with gr.Row():
                ds_target_sr_enabled = gr.Checkbox(label="リサンプルを有効化", value=False)
                ds_target_sr = gr.Number(
                    label="リサンプル先サンプリングレート（Hz）", value=44100, precision=0
                )

        with gr.Accordion("キャプション設定", open=True) as caption_accordion:
            gr.Markdown(
                "*faster-whisper で音声を文字起こしします。精度重視設定（large-v3 + beam=5）がデフォルトです。*"
            )
            ds_caption_input = gr.Textbox(
                label="キャプション対象フォルダ（キャプションのみモード時に使用）",
                value=str(cnf.DEFAULT_DATASET_DIR),
                placeholder=str(cnf.DEFAULT_DATASET_DIR),
                info="パイプラインモード時はスライス出力先が自動的に使われます。",
            )
            with gr.Row():
                ds_whisper_model = gr.Dropdown(
                    label="Whisperモデル",
                    choices=["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
                    value="medium",
                    info="精度: large-v3 > large-v2 > medium > small（VRAMも同順で多く必要）",
                )
                ds_language = gr.Dropdown(
                    label="言語",
                    choices=["ja", "en", "zh", "ko", "自動検出"],
                    value="ja",
                )
                ds_device = gr.Dropdown(
                    label="使用デバイス",
                    choices=["自動", "cuda", "cpu"],
                    value="自動",
                )
            with gr.Row():
                ds_speaker_id = gr.Textbox(
                    label="話者ID（省略可・全ファイルに付与）",
                    value="",
                    placeholder="例: SPEAKER_A",
                    scale=2,
                )
                ds_recursive_caption = gr.Checkbox(label="サブフォルダも検索", value=False, scale=1)
            ds_model_cache_dir = gr.Textbox(
                label="Whisperモデルキャッシュフォルダ",
                value=str(cnf.CHECKPOINTS_DIR / "whisper"),
                info="モデルが存在しない場合は自動ダウンロードされます。空欄にするとHFデフォルト (~/.cache/huggingface/hub) に保存されます。",
            )

        with gr.Accordion("絵文字キャプション設定（オプション）", open=False):
            gr.Markdown(
                "有効にすると、Whisperキャプション完了後に音響特徴量とLLMを使って"
                "**Irodori-TTS互換の絵文字キャプション**を自動生成します。\n\n"
                "> 必要ライブラリ: `pip install librosa openai`  \n"
                "> Manifest出力設定が **CSV形式** の場合のみ動作します。"
            )
            with gr.Row():
                ec_enabled = gr.Checkbox(
                    label="絵文字キャプションを有効にする",
                    value=False,
                    scale=1,
                    info="チェックを入れると通常キャプション完了後に絵文字キャプションを続けて実行します。",
                )
                ec_api = gr.Dropdown(
                    label="APIプロバイダー",
                    choices=[
                        "LM Studio（ローカル）",
                        "Groq",
                        "OpenAI（ChatGPT）",
                        "Together AI",
                    ],
                    value="LM Studio（ローカル）",
                    scale=2,
                    interactive=False,
                    info="LM StudioはAPIキー不要。起動してモデルをロードしておいてください。",
                )

            ec_api_key = gr.Textbox(
                label="APIキー（Groq / OpenAI / Together AI）",
                value="",
                placeholder="sk-... または gsk_...",
                type="password",
                visible=False,
                info="LM Studio以外を選択した場合に必要です。",
            )

            def _ec_on_enabled(checked):
                return gr.update(interactive=checked)

            def _ec_on_api_change(api_label):
                need_key = api_label not in ("LM Studio（ローカル）",)
                return gr.update(visible=need_key)

            ec_enabled.change(_ec_on_enabled, inputs=[ec_enabled], outputs=[ec_api])
            ec_api.change(_ec_on_api_change, inputs=[ec_api], outputs=[ec_api_key])

        with gr.Accordion("Manifest出力設定", open=True):
            with gr.Row():
                ds_manifest_output_dir = gr.Textbox(
                    label="manifest保存先フォルダ",
                    value=str(cnf.DEFAULT_DATASET_DIR),
                    placeholder=str(cnf.DEFAULT_DATASET_DIR),
                    scale=3,
                )
                ds_manifest_filename = gr.Textbox(
                    label="ファイル名（拡張子なし）",
                    value="metadata",
                    scale=2,
                )
                ds_output_format = gr.Dropdown(
                    label="出力フォーマット",
                    choices=["CSV", "JSONL"],
                    value="CSV",
                    scale=1,
                )
            gr.Markdown(
                "**フォーマット補足**\n"
                "- **CSV**: `audio_path,text,speaker_id` — Excelや各種ツールで開きやすい汎用形式\n"
                '- **JSONL**: `{"text":"...","audio_path":"..."}` — `prepare_manifest.py` への入力前段として使用可能'
            )

        gr.Markdown("### 実行コマンドプレビュー")
        ds_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

        with gr.Row():
            ds_start_btn = gr.Button("実行", variant="primary", size="lg")
            ds_stop_btn = gr.Button("停止", variant="stop")
        ds_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

        gr.Markdown("### 実行ログ")
        with gr.Row():
            ds_log_interval = gr.Slider(
                label="自動更新間隔（秒）",
                minimum=2,
                maximum=30,
                value=3,
                step=1,
                scale=3,
            )
            ds_log_refresh_btn = gr.Button("手動更新", scale=1)
        ds_log_text = gr.Textbox(
            label="ログ出力",
            interactive=False,
            lines=20,
            max_lines=20,
            elem_id="ds_log_text",
        )
        gr.HTML("""
<script>
(function() {
    function attachDsLogScroll() {
        var el = document.getElementById('ds_log_text');
        if (!el) { setTimeout(attachDsLogScroll, 500); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(attachDsLogScroll, 500); return; }
        var lastVal = ta.value;
        setInterval(function() {
            if (ta.value !== lastVal) { lastVal = ta.value; ta.scrollTop = ta.scrollHeight; }
        }, 300);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachDsLogScroll);
    } else { attachDsLogScroll(); }
})();
</script>
""")

        def _on_mode_change(mode):
            show_slice = mode in ("スライスのみ", "パイプライン（スライス→キャプション）")
            show_caption = mode in ("キャプションのみ", "パイプライン（スライス→キャプション）")
            return gr.update(open=show_slice), gr.update(open=show_caption)

        ds_mode.change(
            _on_mode_change,
            inputs=[ds_mode],
            outputs=[slice_accordion, caption_accordion],
        )

        _ds_all_inputs = [
            ds_mode,
            ds_input,
            ds_slice_output,
            ds_min_sec,
            ds_max_sec,
            ds_top_db,
            ds_frame_length,
            ds_hop_length,
            ds_target_sr_enabled,
            ds_target_sr,
            ds_recursive_slice,
            ds_caption_input,
            ds_manifest_output_dir,
            ds_manifest_filename,
            ds_output_format,
            ds_whisper_model,
            ds_language,
            ds_speaker_id,
            ds_recursive_caption,
            ds_device,
            ds_model_cache_dir,
        ]

        def _update_ds_cmd(*args):
            try:
                return " ".join(_build_dataset_command(*args))
            except Exception as e:
                return f"(プレビュー生成エラー: {e})"

        for comp in _ds_all_inputs:
            comp.change(_update_ds_cmd, inputs=_ds_all_inputs, outputs=[ds_cmd_preview])

        _ds_exec_inputs = _ds_all_inputs + [ec_enabled, ec_api, ec_api_key]

        def _start_dataset_job_with_emoji(*args):
            base_args = args[:-3]
            ec_enabled_ = bool(args[-3])
            ec_api_ = str(args[-2])
            ec_api_key_ = str(args[-1]).strip()

            status, cmd = _start_dataset_job(*base_args)

            if ec_enabled_:
                output_fmt = str(args[14]).strip()
                if output_fmt != "CSV":
                    status += "\n⚠️ 絵文字キャプションはCSV形式のみ対応です。出力形式をCSVに変更してください。"
                else:
                    manifest_dir = str(args[12]).strip()
                    manifest_name = str(args[13]).strip()
                    csv_path = str(Path(manifest_dir) / f"{manifest_name}.csv")
                    mode_ = str(args[0])
                    wav_dir_ = (
                        str(args[11]).strip()
                        if mode_ == "キャプションのみ"
                        else str(args[2]).strip()
                    )

                    def _wait_and_emoji(
                        csv_path=csv_path,
                        wav_dir_=wav_dir_,
                        ec_api_=ec_api_,
                        ec_api_key_=ec_api_key_,
                    ):
                        while True:
                            with _DS_LOG_LOCK:
                                proc = _DS_PROC
                            if proc is None or proc.poll() is not None:
                                break
                            time.sleep(1)
                        with _DS_LOG_LOCK:
                            rc = _DS_PROC.returncode if _DS_PROC else -1
                        if rc == 0:
                            _run_emoji_caption_inline(csv_path, wav_dir_, ec_api_, ec_api_key_)

                    threading.Thread(target=_wait_and_emoji, daemon=True).start()
                    status += (
                        f"\n絵文字キャプション: 通常処理完了後に自動実行します（API: {ec_api_}）"
                    )

            return status, cmd

        ds_start_btn.click(
            _start_dataset_job_with_emoji,
            inputs=_ds_exec_inputs,
            outputs=[ds_status, ds_cmd_preview],
        )
        ds_stop_btn.click(_stop_dataset_job, outputs=[ds_status])

        def _ds_refresh():
            return _read_dataset_log()

        ds_log_refresh_btn.click(_ds_refresh, outputs=[ds_log_text])

        _ds_timer = gr.Timer(value=3, active=True)
        _ds_timer.tick(_ds_refresh, outputs=[ds_log_text])
        ds_log_interval.change(
            lambda v: float(v),
            inputs=[ds_log_interval],
            outputs=[_ds_timer],
        )
