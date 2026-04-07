import gradio as gr
from ui.common import *
from .setting import cnfg
from merge import get_default_base_path


# ─────────────────────────────
# モデルマージ タブ ロジック
# ─────────────────────────────


def _run_merge_ui(
    path_a,
    path_b,
    method,
    alpha,
    lambda_a,
    lambda_b,
    base_path_ta,
    use_partial,
    text_method,
    text_alpha,
    text_lam_a,
    text_lam_b,
    speaker_method,
    speaker_alpha,
    speaker_lam_a,
    speaker_lam_b,
    diffusion_method,
    diffusion_alpha,
    diffusion_lam_a,
    diffusion_lam_b,
    io_method,
    io_alpha,
    io_lam_a,
    io_lam_b,
    use_lora,
    lora_base,
    lora_donor,
    lora_scale,
    lora_grp_text,
    lora_grp_speaker,
    lora_grp_diffusion,
    lora_grp_io,
    output_format,
    output_dir,
) -> str:
    def _norm(a, b):
        t = float(a) + float(b)
        return (float(a) / t, float(b) / t) if t > 0 else (0.5, 0.5)

    group_methods = None
    if use_partial:

        def _group_cfg(meth, al, la, lb):
            if meth == "task_arithmetic":
                na, nb = _norm(la, lb)
                return {"method": meth, "lambda_a": na, "lambda_b": nb}
            return {"method": meth, "alpha": float(al)}

        group_methods = {
            "text": _group_cfg(text_method, text_alpha, text_lam_a, text_lam_b),
            "speaker": _group_cfg(speaker_method, speaker_alpha, speaker_lam_a, speaker_lam_b),
            "diffusion_core": _group_cfg(
                diffusion_method, diffusion_alpha, diffusion_lam_a, diffusion_lam_b
            ),
            "io": _group_cfg(io_method, io_alpha, io_lam_a, io_lam_b),
        }

    lora_targets = []
    if lora_grp_text:
        lora_targets.append("text")
    if lora_grp_speaker:
        lora_targets.append("speaker")
    if lora_grp_diffusion:
        lora_targets.append("diffusion_core")
    if lora_grp_io:
        lora_targets.append("io")

    la_norm, lb_norm = _norm(lambda_a, lambda_b)

    success, message = run_merge(
        path_a=str(path_a),
        path_b=str(path_b),
        method=str(method),
        alpha=float(alpha),
        lambda_a=la_norm,
        lambda_b=lb_norm,
        base_path=str(base_path_ta) if base_path_ta else None,
        use_partial=bool(use_partial),
        group_methods=group_methods,
        use_lora_inject=bool(use_lora),
        lora_base_path=str(lora_base) if lora_base else None,
        lora_donor_path=str(lora_donor) if lora_donor else None,
        lora_scale=float(lora_scale),
        lora_target_groups=lora_targets if lora_targets else None,
        output_format="safetensors" if output_format == ".safetensors" else "pt",
        output_dir=str(output_dir) if output_dir else None,
    )
    return message


# ─────────────────────────────
# UI 生成
# ─────────────────────────────


def build(ctx):
    with gr.Tab("🔀 モデルマージ"):
        gr.Markdown(
            "## モデルマージ\n"
            "推論用モデル（EMA .pt / .safetensors）同士をマージして新しいモデルを生成します。\n\n"
            "> **対応形式**: `_ema.pt` / `.safetensors`（推論用のみ）"
        )

        initial_merge_ckpts = merge_scan()
        default_base_path = get_default_base_path()

        with gr.Row():
            merge_ckpt_a = gr.Dropdown(
                label="モデルA",
                choices=initial_merge_ckpts,
                value=initial_merge_ckpts[-1] if initial_merge_ckpts else None,
                allow_custom_value=True,
                scale=4,
            )
            merge_refresh_a = gr.Button("更新", scale=1)

        with gr.Row():
            merge_ckpt_b = gr.Dropdown(
                label="モデルB",
                choices=initial_merge_ckpts,
                value=initial_merge_ckpts[0] if len(initial_merge_ckpts) > 1 else None,
                allow_custom_value=True,
                scale=4,
            )
            merge_refresh_b = gr.Button("更新", scale=1)

        with gr.Accordion("基本マージ設定", open=True):
            merge_method = gr.Dropdown(
                label="マージ手法",
                choices=["weighted_average", "slerp", "task_arithmetic"],
                value="weighted_average",
                info="weighted_average: 安定・高速 / slerp: ノルム保持 / task_arithmetic: ベースモデル必要",
            )

            with gr.Row():
                merge_alpha = gr.Slider(
                    label="α（モデルAの割合）",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    info="Weighted Average / SLERP で使用",
                )

            with gr.Group() as ta_group:
                gr.Markdown("**Task Arithmetic 設定**")
                with gr.Row():
                    merge_lambda_a = gr.Slider(
                        label="λA（モデルAタスクベクトルの重み）",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.01,
                    )
                    merge_lambda_b = gr.Slider(
                        label="λB（モデルBタスクベクトルの重み）",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.01,
                        info="λA + λB は自動的に合計1.0に正規化されます",
                    )
                with gr.Row():
                    merge_base_ta = gr.Dropdown(
                        label="ベースモデル（Task Arithmetic用）",
                        choices=initial_merge_ckpts,
                        value=default_base_path
                        if default_base_path in initial_merge_ckpts
                        else (initial_merge_ckpts[0] if initial_merge_ckpts else None),
                        allow_custom_value=True,
                        scale=4,
                    )
                    merge_refresh_base = gr.Button("更新", scale=1)

                def _on_method_change(method):
                    visible = method == "task_arithmetic"
                    return gr.update(visible=visible)

                merge_method.change(_on_method_change, inputs=[merge_method], outputs=[ta_group])

        with gr.Accordion("部分マージ（グループごとに手法を選択）", open=False):
            gr.Markdown(
                "有効にすると、レイヤーグループごとに異なるマージ手法を設定できます。\n"
                "- **text**: テキストエンコーダ・TextBlock・JointAttentionのテキストKV\n"
                "- **speaker**: 話者エンコーダ・JointAttentionの話者KV\n"
                "- **diffusion_core**: DiffusionBlock本体（Attention/MLP/AdaLN）・cond_module\n"
                "- **io**: in_proj / out_norm / out_proj"
            )
            use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)

            _method_choices = ["weighted_average", "slerp", "task_arithmetic"]

            with gr.Group():
                gr.Markdown("#### テキスト応答性グループ（text）")
                with gr.Row():
                    pg_text_method = gr.Dropdown(
                        choices=_method_choices, value="weighted_average", label="手法"
                    )
                    pg_text_alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α"
                    )
                    pg_text_lam_a = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA"
                    )
                    pg_text_lam_b = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB"
                    )

                gr.Markdown("#### 話者表現グループ（speaker）")
                with gr.Row():
                    pg_spk_method = gr.Dropdown(
                        choices=_method_choices, value="weighted_average", label="手法"
                    )
                    pg_spk_alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α"
                    )
                    pg_spk_lam_a = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA"
                    )
                    pg_spk_lam_b = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB"
                    )

                gr.Markdown("#### 拡散コアグループ（diffusion_core）")
                with gr.Row():
                    pg_diff_method = gr.Dropdown(
                        choices=_method_choices, value="weighted_average", label="手法"
                    )
                    pg_diff_alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α"
                    )
                    pg_diff_lam_a = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA"
                    )
                    pg_diff_lam_b = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB"
                    )

                gr.Markdown("#### 入出力グループ（io）")
                with gr.Row():
                    pg_io_method = gr.Dropdown(
                        choices=_method_choices, value="weighted_average", label="手法"
                    )
                    pg_io_alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α"
                    )
                    pg_io_lam_a = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA"
                    )
                    pg_io_lam_b = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB"
                    )

        with gr.Accordion("LoRA的差分注入（オプション）", open=False):
            gr.Markdown(
                "ベースモデルに対してドナーモデルの差分を指定スケールで注入します。\n"
                "`result = base + scale × (donor − base)`\n\n"
                "注入対象グループを選択してください。"
            )
            use_lora = gr.Checkbox(label="LoRA的差分注入を有効にする", value=False)

            with gr.Row():
                lora_base = gr.Dropdown(
                    label="ベースモデル",
                    choices=initial_merge_ckpts,
                    value=default_base_path
                    if default_base_path in initial_merge_ckpts
                    else (initial_merge_ckpts[0] if initial_merge_ckpts else None),
                    allow_custom_value=True,
                    scale=4,
                )
                lora_refresh_base = gr.Button("更新", scale=1)

            with gr.Row():
                lora_donor = gr.Dropdown(
                    label="ドナーモデル（差分元）",
                    choices=initial_merge_ckpts,
                    value=initial_merge_ckpts[-1] if initial_merge_ckpts else None,
                    allow_custom_value=True,
                    scale=4,
                )
                lora_refresh_donor = gr.Button("更新", scale=1)

            lora_scale = gr.Slider(
                label="注入スケール（0=ベースのみ、1=ドナーに完全置換）",
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.01,
            )

            gr.Markdown("**注入対象グループ**")
            with gr.Row():
                lora_grp_text = gr.Checkbox(label="text（テキスト応答性）", value=True)
                lora_grp_speaker = gr.Checkbox(label="speaker（話者表現）", value=True)
                lora_grp_diffusion = gr.Checkbox(label="diffusion_core（拡散コア）", value=False)
                lora_grp_io = gr.Checkbox(label="io（入出力）", value=False)

        with gr.Accordion("出力設定", open=True):
            with gr.Row():
                merge_output_format = gr.Dropdown(
                    label="保存形式",
                    choices=[".safetensors", ".pt"],
                    value=".safetensors",
                    info=".safetensors=推論用（推奨）/ .pt=PyTorch標準形式",
                    scale=1,
                )
                merge_output_dir = gr.Textbox(
                    label="保存先フォルダ（空欄=" + str(cnfg.checkpoints_dir) + "/merged/）",
                    value="",
                    placeholder=str(cnfg.checkpoints_dir / "merged"),
                    scale=3,
                )

        merge_run_btn = gr.Button("マージ実行", variant="primary", size="lg")
        merge_status = gr.Textbox(label="実行結果", interactive=False, lines=10)

        def _rescan_merge():
            ckpts = merge_scan()
            val = ckpts[-1] if ckpts else None
            return gr.Dropdown(choices=ckpts, value=val)

        merge_refresh_a.click(_rescan_merge, outputs=[merge_ckpt_a])
        merge_refresh_b.click(_rescan_merge, outputs=[merge_ckpt_b])
        merge_refresh_base.click(_rescan_merge, outputs=[merge_base_ta])
        lora_refresh_base.click(_rescan_merge, outputs=[lora_base])
        lora_refresh_donor.click(_rescan_merge, outputs=[lora_donor])

        _merge_inputs = [
            merge_ckpt_a,
            merge_ckpt_b,
            merge_method,
            merge_alpha,
            merge_lambda_a,
            merge_lambda_b,
            merge_base_ta,
            use_partial,
            pg_text_method,
            pg_text_alpha,
            pg_text_lam_a,
            pg_text_lam_b,
            pg_spk_method,
            pg_spk_alpha,
            pg_spk_lam_a,
            pg_spk_lam_b,
            pg_diff_method,
            pg_diff_alpha,
            pg_diff_lam_a,
            pg_diff_lam_b,
            pg_io_method,
            pg_io_alpha,
            pg_io_lam_a,
            pg_io_lam_b,
            use_lora,
            lora_base,
            lora_donor,
            lora_scale,
            lora_grp_text,
            lora_grp_speaker,
            lora_grp_diffusion,
            lora_grp_io,
            merge_output_format,
            merge_output_dir,
        ]
        merge_run_btn.click(_run_merge_ui, inputs=_merge_inputs, outputs=[merge_status])
