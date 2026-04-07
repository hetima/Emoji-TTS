import subprocess
import sys
import gradio as gr
from ui.common import *
import gradio_conf as cnf

# ─────────────────────────────
# Convert タブ ロジック
# ─────────────────────────────

def _run_convert(input_pt: str) -> str:
    if not str(input_pt).strip():
        return "エラー: 変換対象の .pt ファイルを選択してください。"
    p = Path(input_pt)
    if not p.is_file():
        return f"エラー: ファイルが見つかりません: {p}"
    cmd = f"{sys.executable} {cnf.REPO_DIR / 'convert_checkpoint_to_safetensors.py'} {p}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        out = result.stdout + result.stderr
        if result.returncode == 0:
            safetensors_path = p.with_suffix(".safetensors")
            return f"変換完了: {safetensors_path}\n\n{out}"
        else:
            return f"変換失敗 (returncode={result.returncode}):\n{out}"
    except subprocess.TimeoutExpired:
        return "エラー: タイムアウト (300秒超過)"
    except Exception as e:
        return f"エラー: {e}"


def _run_lora_convert(input_full_dir: str, force: bool = False) -> str:
    if not str(input_full_dir).strip():
        return "エラー: 変換対象の _full フォルダを選択してください。"
    p = Path(input_full_dir)
    if not p.is_dir():
        return f"エラー: フォルダが存在しません: {p}"
    cmd = [sys.executable, str(cnf.REPO_DIR / "convert_lora_checkpoint.py"), str(p)]
    if force:
        cmd += ["--force"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return f"変換完了\n\n{out}"
    else:
        return f"変換失敗 (returncode={result.returncode}):\n{out}"

# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx):
    with gr.Tab("🔄 チェックポイント変換"):
        with gr.Tab("通常チェックポイント変換"):
            gr.Markdown(
                "## .pt → .safetensors 変換\n"
                "学習チェックポイント（`.pt`）を推論用の `.safetensors` 形式に変換します。\n"
                "変換後のファイルは元の `.pt` と同じフォルダに保存されます。"
            )

            with gr.Row():
                conv_input = gr.Dropdown(
                    label="変換対象の .pt ファイル",
                    choices=ctx.initial_train_ckpts,
                    value=ctx.initial_train_ckpts[-1] if ctx.initial_train_ckpts else None,
                    allow_custom_value=True, scale=4,
                )
                conv_refresh_btn = gr.Button("更新", scale=1)

            conv_btn    = gr.Button("変換実行", variant="primary", size="lg")
            conv_status = gr.Textbox(label="変換結果", interactive=False, lines=6)

            conv_refresh_btn.click(
                lambda: gr.Dropdown(choices=scan_train_checkpoints(), value=(scan_train_checkpoints() or [None])[-1]),
                outputs=[conv_input],
            )
            conv_btn.click(_run_convert, inputs=[conv_input], outputs=[conv_status])

        with gr.Tab("🚀 LoRA変換"):
            gr.Markdown(
                "## LoRA Full版 → EMA版 変換\n"
                "`_full` フォルダの EMA shadow 重みから、推論用の `_ema` フォルダを生成します。\n\n"
                "> **必要条件**: LoRA学習時に `--save-full` と `--ema-decay` を指定して保存したチェックポイント"
            )

            with gr.Row():
                lora_conv_input = gr.Dropdown(
                    label="変換対象の _full フォルダ",
                    choices=scan_lora_full_adapters(),
                    value=(scan_lora_full_adapters() or [None])[-1],
                    allow_custom_value=True, scale=4,
                )
                lora_conv_refresh_btn = gr.Button("更新", scale=1)

            lora_conv_force = gr.Checkbox(label="既存の出力を上書き (--force)", value=False)
            lora_conv_btn = gr.Button("LoRA変換実行", variant="primary", size="lg")
            lora_conv_status = gr.Textbox(label="変換結果", interactive=False, lines=8)

            lora_conv_refresh_btn.click(
                lambda: gr.Dropdown(choices=scan_lora_full_adapters(), value=(scan_lora_full_adapters() or [None])[-1]),
                outputs=[lora_conv_input],
            )
            lora_conv_btn.click(
                _run_lora_convert,
                inputs=[lora_conv_input, lora_conv_force],
                outputs=[lora_conv_status],
            )
