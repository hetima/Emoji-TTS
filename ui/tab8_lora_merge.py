import gradio as gr
from ui.common import *
import gradio_conf as cnf
from lora_merge import run_lora_merge, run_lora_lora_merge, scan_lora_adapters_for_merge, peek_adapter_version
from merge import get_default_base_path

# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx):
    with gr.Tab("🧬 LoRAマージ"):
        gr.Markdown("## LoRAマージ")

        initial_lm_adapters = scan_lora_adapters_for_merge()
        initial_lm_ckpts    = merge_scan()
        default_lm_base     = get_default_base_path()

        with gr.Tabs():

            with gr.Tab("通常LoRAマージ"):
                gr.Markdown(
                    "## 通常LoRAマージ\n"
                    "LoRAアダプタ同士をマージして新しいLoRAアダプタを生成します。\n"
                    "ベースモデルへの焼き込みは行いません。\n\n"
                    "> **出力**: `lora/lora_merged_*/` フォルダ（推論タブで直接使用可）"
                )

                with gr.Row():
                    ll_adapter_a = gr.Dropdown(
                        label="アダプタA",
                        choices=["（なし）"] + initial_lm_adapters,
                        value=initial_lm_adapters[0] if initial_lm_adapters else "（なし）",
                        allow_custom_value=True, scale=4,
                    )
                    ll_ver_a   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                    ll_ref_a   = gr.Button("更新", scale=1)

                with gr.Row():
                    ll_adapter_b = gr.Dropdown(
                        label="アダプタB",
                        choices=["（なし）"] + initial_lm_adapters,
                        value=initial_lm_adapters[1] if len(initial_lm_adapters) > 1 else "（なし）",
                        allow_custom_value=True, scale=4,
                    )
                    ll_ver_b   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                    ll_ref_b   = gr.Button("更新", scale=1)

                with gr.Accordion("マージ設定", open=True):
                    ll_method = gr.Dropdown(
                        label="マージ手法",
                        choices=["weighted_average", "slerp", "task_arithmetic"],
                        value="weighted_average",
                        info="weighted_average: 安定 / slerp: ノルム保持 / task_arithmetic: ベースアダプタ必要",
                    )
                    with gr.Row():
                        ll_alpha    = gr.Slider(label="α（アダプタAの割合）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                    with gr.Group() as ll_ta_group:
                        gr.Markdown("**Task Arithmetic 設定**")
                        with gr.Row():
                            ll_lambda_a = gr.Slider(label="λA", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                            ll_lambda_b = gr.Slider(label="λB（自動正規化）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            ll_base_adapter = gr.Dropdown(
                                label="ベースアダプタ（Task Arithmetic用）",
                                choices=["（なし）"] + initial_lm_adapters,
                                value="（なし）",
                                allow_custom_value=True, scale=4,
                            )
                            ll_ref_base = gr.Button("更新", scale=1)

                with gr.Accordion("部分マージ（グループ別手法）", open=False):
                    gr.Markdown(
                        "有効にすると、レイヤーグループごとに異なる手法でマージできます。\n"
                        "上の「マージ設定」より優先されます。"
                    )
                    ll_use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)
                    _ll_mc = ["weighted_average", "slerp", "task_arithmetic"]
                    with gr.Group():
                        gr.Markdown("#### text グループ")
                        with gr.Row():
                            ll_pg_text_m = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                            ll_pg_text_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            ll_pg_text_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            ll_pg_text_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                        gr.Markdown("#### speaker グループ")
                        with gr.Row():
                            ll_pg_spk_m  = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                            ll_pg_spk_a  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            ll_pg_spk_la = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            ll_pg_spk_lb = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                        gr.Markdown("#### diffusion_core グループ")
                        with gr.Row():
                            ll_pg_diff_m = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                            ll_pg_diff_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            ll_pg_diff_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            ll_pg_diff_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                        gr.Markdown("#### io グループ")
                        with gr.Row():
                            ll_pg_io_m   = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                            ll_pg_io_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            ll_pg_io_la  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            ll_pg_io_lb  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                with gr.Accordion("出力設定", open=True):
                    ll_output_dir = gr.Textbox(
                        label="保存先フォルダ（空欄=" + str(cnf.LORA_DIR) + "/lora_merged_*/）",
                        value="",
                        placeholder=str(cnf.LORA_DIR / "lora_merged_*"),
                    )

                ll_run_btn = gr.Button("LoRAマージ実行", variant="primary", size="lg")
                ll_status  = gr.Textbox(label="実行結果", interactive=False, lines=10)

                def _ll_rescan():
                    ads = scan_lora_adapters_for_merge()
                    return gr.Dropdown(choices=["（なし）"] + ads)

                def _ll_ver(v):
                    if not v or str(v).strip() in ("", "（なし）"):
                        return ""
                    return peek_adapter_version(str(v).strip())

                def _ll_method_change(m):
                    return gr.update(visible=(m == "task_arithmetic"))

                ll_ref_a.click(_ll_rescan, outputs=[ll_adapter_a])
                ll_ref_b.click(_ll_rescan, outputs=[ll_adapter_b])
                ll_ref_base.click(_ll_rescan, outputs=[ll_base_adapter])
                ll_adapter_a.change(_ll_ver, inputs=[ll_adapter_a], outputs=[ll_ver_a])
                ll_adapter_b.change(_ll_ver, inputs=[ll_adapter_b], outputs=[ll_ver_b])
                ll_method.change(_ll_method_change, inputs=[ll_method], outputs=[ll_ta_group])

                def _run_ll_merge_ui(
                    adapter_a, adapter_b,
                    method, alpha, lambda_a, lambda_b, base_adapter,
                    use_partial,
                    pg_text_m, pg_text_a, pg_text_la, pg_text_lb,
                    pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb,
                    pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb,
                    pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb,
                    output_dir,
                ) -> str:
                    if str(adapter_a).strip() in ("", "（なし）"):
                        return "❌ アダプタAを選択してください。"
                    if str(adapter_b).strip() in ("", "（なし）"):
                        return "❌ アダプタBを選択してください。"

                    def _norm(a, b):
                        t = float(a) + float(b)
                        return (float(a)/t, float(b)/t) if t > 0 else (0.5, 0.5)

                    gm = None
                    if use_partial:
                        def _gc(m, al, la, lb):
                            if m == "task_arithmetic":
                                na, nb = _norm(la, lb)
                                return {"method": m, "lambda_a": na, "lambda_b": nb}
                            return {"method": m, "alpha": float(al)}
                        gm = {
                            "text":          _gc(pg_text_m, pg_text_a, pg_text_la, pg_text_lb),
                            "speaker":       _gc(pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb),
                            "diffusion_core":_gc(pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb),
                            "io":            _gc(pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb),
                        }

                    ba = str(base_adapter).strip()
                    ba = None if ba in ("", "（なし）") else ba
                    la_n, lb_n = _norm(lambda_a, lambda_b)

                    _, msg = run_lora_lora_merge(
                        adapter_dir_a=str(adapter_a).strip(),
                        adapter_dir_b=str(adapter_b).strip(),
                        method=str(method),
                        alpha=float(alpha),
                        lambda_a=la_n,
                        lambda_b=lb_n,
                        base_adapter_dir=ba,
                        use_partial=bool(use_partial),
                        group_methods=gm,
                        output_dir=str(output_dir).strip() or None,
                    )
                    return msg

                _ll_inputs = [
                    ll_adapter_a, ll_adapter_b,
                    ll_method, ll_alpha, ll_lambda_a, ll_lambda_b, ll_base_adapter,
                    ll_use_partial,
                    ll_pg_text_m, ll_pg_text_a, ll_pg_text_la, ll_pg_text_lb,
                    ll_pg_spk_m,  ll_pg_spk_a,  ll_pg_spk_la,  ll_pg_spk_lb,
                    ll_pg_diff_m, ll_pg_diff_a, ll_pg_diff_la, ll_pg_diff_lb,
                    ll_pg_io_m,   ll_pg_io_a,   ll_pg_io_la,   ll_pg_io_lb,
                    ll_output_dir,
                ]
                ll_run_btn.click(_run_ll_merge_ui, inputs=_ll_inputs, outputs=[ll_status])

            with gr.Tab("本体モデルマージ（焼き込み）"):
                gr.Markdown(
                    "## 本体モデルマージ（焼き込み）\n"
                    "LoRAアダプタをベースモデルに焼き込み、マージ済みモデルを生成します。\n\n"
                    "> **出力**: `checkpoints/lora_merged/` フォルダ（推論タブで直接使用可）"
                )

                with gr.Row():
                    lm_base_model = gr.Dropdown(
                        label="ベースモデル (.pt / .safetensors)",
                        choices=initial_lm_ckpts,
                        value=default_lm_base if default_lm_base in initial_lm_ckpts else (initial_lm_ckpts[-1] if initial_lm_ckpts else None),
                        allow_custom_value=True, scale=4,
                    )
                    lm_refresh_base = gr.Button("更新", scale=1)

                with gr.Accordion("アダプタA（必須）", open=True):
                    with gr.Row():
                        lm_adapter_a1 = gr.Dropdown(
                            label="アダプタA-1",
                            choices=["（なし）"] + initial_lm_adapters,
                            value=initial_lm_adapters[0] if initial_lm_adapters else "（なし）",
                            allow_custom_value=True, scale=4,
                        )
                        lm_scale_a1 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                        lm_ver_a1   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                        lm_ref_a1   = gr.Button("更新", scale=1)
                    with gr.Row():
                        lm_adapter_a2 = gr.Dropdown(
                            label="アダプタA-2（省略可）",
                            choices=["（なし）"] + initial_lm_adapters,
                            value="（なし）",
                            allow_custom_value=True, scale=4,
                        )
                        lm_scale_a2 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                        lm_ver_a2   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                        lm_ref_a2   = gr.Button("更新", scale=1)
                    with gr.Accordion("🧩 部分焼き込みA", open=False):
                        lm_use_pbake_a = gr.Checkbox(label="部分焼き込みAを有効にする", value=False)
                        with gr.Row():
                            lm_pbake_a_text  = gr.Checkbox(label="text",           value=True)
                            lm_pbake_a_spk   = gr.Checkbox(label="speaker",        value=True)
                            lm_pbake_a_diff  = gr.Checkbox(label="diffusion_core", value=True)
                            lm_pbake_a_io    = gr.Checkbox(label="io",             value=True)

                with gr.Accordion("焼き込み後マージ（オプション）", open=False):
                    lm_post_method = gr.Dropdown(
                        label="マージ手法",
                        choices=["none", "weighted_average", "slerp", "task_arithmetic"],
                        value="none",
                        info="none=焼き込みのみ",
                    )
                    with gr.Group() as lm_post_group:
                        with gr.Row():
                            lm_post_alpha = gr.Slider(label="α（アダプタA側の割合）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            lm_post_lam_a = gr.Slider(label="λA", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                            lm_post_lam_b = gr.Slider(label="λB", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            lm_post_base = gr.Dropdown(
                                label="Task Arithmetic用ベースモデル（省略時=ベースモデルと同じ）",
                                choices=["（省略）"] + initial_lm_ckpts,
                                value="（省略）",
                                allow_custom_value=True, scale=4,
                            )
                            lm_ref_post_base = gr.Button("更新", scale=1)

                    with gr.Group() as lm_adp_b_group:
                        gr.Markdown("**アダプタB**")
                        with gr.Row():
                            lm_adapter_b1 = gr.Dropdown(
                                label="アダプタB-1",
                                choices=["（なし）"] + initial_lm_adapters,
                                value="（なし）",
                                allow_custom_value=True, scale=4,
                            )
                            lm_scale_b1 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                            lm_ver_b1   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                            lm_ref_b1   = gr.Button("更新", scale=1)
                        with gr.Row():
                            lm_adapter_b2 = gr.Dropdown(
                                label="アダプタB-2（省略可）",
                                choices=["（なし）"] + initial_lm_adapters,
                                value="（なし）",
                                allow_custom_value=True, scale=4,
                            )
                            lm_scale_b2 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                            lm_ver_b2   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                            lm_ref_b2   = gr.Button("更新", scale=1)
                        with gr.Accordion("部分焼き込みB", open=False):
                            lm_use_pbake_b = gr.Checkbox(label="部分焼き込みBを有効にする", value=False)
                            with gr.Row():
                                lm_pbake_b_text = gr.Checkbox(label="text",           value=True)
                                lm_pbake_b_spk  = gr.Checkbox(label="speaker",        value=True)
                                lm_pbake_b_diff = gr.Checkbox(label="diffusion_core", value=True)
                                lm_pbake_b_io   = gr.Checkbox(label="io",             value=True)

                    with gr.Accordion("部分マージ（焼き込み後・グループ別）", open=False):
                        lm_use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)
                        _lm_mc = ["weighted_average", "slerp", "task_arithmetic"]
                        with gr.Group():
                            gr.Markdown("#### text グループ")
                            with gr.Row():
                                lm_pg_text_m = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                lm_pg_text_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                lm_pg_text_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                lm_pg_text_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                            gr.Markdown("#### speaker グループ")
                            with gr.Row():
                                lm_pg_spk_m  = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                lm_pg_spk_a  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                lm_pg_spk_la = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                lm_pg_spk_lb = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                            gr.Markdown("#### diffusion_core グループ")
                            with gr.Row():
                                lm_pg_diff_m = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                lm_pg_diff_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                lm_pg_diff_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                lm_pg_diff_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                            gr.Markdown("#### io グループ")
                            with gr.Row():
                                lm_pg_io_m   = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                lm_pg_io_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                lm_pg_io_la  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                lm_pg_io_lb  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                with gr.Accordion("出力設定", open=True):
                    with gr.Row():
                        lm_output_format = gr.Dropdown(
                            label="保存形式",
                            choices=[".safetensors", ".pt"],
                            value=".safetensors",
                            scale=1,
                        )
                        lm_output_dir = gr.Textbox(
                            label="保存先フォルダ（空欄=checkpoints/lora_merged/）",
                            value="",
                            placeholder=str(cnf.CHECKPOINTS_DIR / "lora_merged"),
                            scale=3,
                        )

                lm_run_btn = gr.Button("焼き込みマージ実行", variant="primary", size="lg")
                lm_status  = gr.Textbox(label="実行結果", interactive=False, lines=12)

                def _lm_rescan_ckpts():
                    ckpts = merge_scan()
                    return gr.Dropdown(choices=ckpts, value=ckpts[-1] if ckpts else None)

                def _lm_rescan_adapters():
                    ads = scan_lora_adapters_for_merge()
                    return gr.Dropdown(choices=["（なし）"] + ads)

                def _lm_ver(v):
                    if not v or str(v).strip() in ("", "（なし）"):
                        return ""
                    return peek_adapter_version(str(v).strip())

                def _lm_post_method_change(m):
                    vis = m != "none"
                    return gr.update(visible=vis), gr.update(visible=vis)

                lm_refresh_base.click(_lm_rescan_ckpts, outputs=[lm_base_model])
                lm_ref_a1.click(_lm_rescan_adapters, outputs=[lm_adapter_a1])
                lm_ref_a2.click(_lm_rescan_adapters, outputs=[lm_adapter_a2])
                lm_ref_b1.click(_lm_rescan_adapters, outputs=[lm_adapter_b1])
                lm_ref_b2.click(_lm_rescan_adapters, outputs=[lm_adapter_b2])
                lm_ref_post_base.click(_lm_rescan_ckpts, outputs=[lm_post_base])
                lm_adapter_a1.change(_lm_ver, inputs=[lm_adapter_a1], outputs=[lm_ver_a1])
                lm_adapter_a2.change(_lm_ver, inputs=[lm_adapter_a2], outputs=[lm_ver_a2])
                lm_adapter_b1.change(_lm_ver, inputs=[lm_adapter_b1], outputs=[lm_ver_b1])
                lm_adapter_b2.change(_lm_ver, inputs=[lm_adapter_b2], outputs=[lm_ver_b2])
                lm_post_method.change(
                    _lm_post_method_change, inputs=[lm_post_method],
                    outputs=[lm_post_group, lm_adp_b_group],
                )

                def _run_lm_bake_ui(
                    base_model,
                    adapter_a1, scale_a1, adapter_a2, scale_a2,
                    use_pbake_a, pbake_a_text, pbake_a_spk, pbake_a_diff, pbake_a_io,
                    post_method, post_alpha, post_lam_a, post_lam_b, post_base,
                    adapter_b1, scale_b1, adapter_b2, scale_b2,
                    use_pbake_b, pbake_b_text, pbake_b_spk, pbake_b_diff, pbake_b_io,
                    use_partial,
                    pg_text_m, pg_text_a, pg_text_la, pg_text_lb,
                    pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb,
                    pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb,
                    pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb,
                    output_format, output_dir,
                ) -> str:
                    def _norm(a, b):
                        t = float(a) + float(b)
                        return (float(a)/t, float(b)/t) if t > 0 else (0.5, 0.5)

                    dirs_a, scales_a = [], []
                    for ad, sc in [(adapter_a1, scale_a1), (adapter_a2, scale_a2)]:
                        if str(ad).strip() not in ("", "（なし）"):
                            dirs_a.append(str(ad).strip()); scales_a.append(float(sc))
                    if not dirs_a:
                        return "❌ アダプタA-1を選択してください。"

                    dirs_b, scales_b = [], []
                    for ad, sc in [(adapter_b1, scale_b1), (adapter_b2, scale_b2)]:
                        if str(ad).strip() not in ("", "（なし）"):
                            dirs_b.append(str(ad).strip()); scales_b.append(float(sc))

                    gb_a = {
                        "text": bool(pbake_a_text), "speaker": bool(pbake_a_spk),
                        "diffusion_core": bool(pbake_a_diff), "io": bool(pbake_a_io),
                    } if use_pbake_a else None
                    gb_b = {
                        "text": bool(pbake_b_text), "speaker": bool(pbake_b_spk),
                        "diffusion_core": bool(pbake_b_diff), "io": bool(pbake_b_io),
                    } if use_pbake_b else None

                    gm = None
                    if use_partial:
                        def _gc(m, al, la, lb):
                            if m == "task_arithmetic":
                                na, nb = _norm(la, lb)
                                return {"method": m, "lambda_a": na, "lambda_b": nb}
                            return {"method": m, "alpha": float(al)}
                        gm = {
                            "text":          _gc(pg_text_m, pg_text_a, pg_text_la, pg_text_lb),
                            "speaker":       _gc(pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb),
                            "diffusion_core":_gc(pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb),
                            "io":            _gc(pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb),
                        }

                    pb_str = str(post_base).strip()
                    pb = None if pb_str in ("", "（省略）") else pb_str
                    lam_a_n, lam_b_n = _norm(post_lam_a, post_lam_b)

                    _, msg = run_lora_merge(
                        base_path=str(base_model),
                        adapter_dirs_a=dirs_a, adapter_scales_a=scales_a,
                        adapter_dirs_b=dirs_b or None,
                        adapter_scales_b=scales_b or None,
                        post_merge_method=str(post_method),
                        post_alpha=float(post_alpha),
                        post_lambda_a=lam_a_n, post_lambda_b=lam_b_n,
                        post_base_path=pb,
                        use_partial=bool(use_partial), group_methods=gm,
                        use_partial_bake_a=bool(use_pbake_a), group_bake_a=gb_a,
                        use_partial_bake_b=bool(use_pbake_b), group_bake_b=gb_b,
                        output_format="safetensors" if output_format == ".safetensors" else "pt",
                        output_dir=str(output_dir).strip() or None,
                    )
                    return msg

                _lm_inputs = [
                    lm_base_model,
                    lm_adapter_a1, lm_scale_a1, lm_adapter_a2, lm_scale_a2,
                    lm_use_pbake_a, lm_pbake_a_text, lm_pbake_a_spk, lm_pbake_a_diff, lm_pbake_a_io,
                    lm_post_method, lm_post_alpha, lm_post_lam_a, lm_post_lam_b, lm_post_base,
                    lm_adapter_b1, lm_scale_b1, lm_adapter_b2, lm_scale_b2,
                    lm_use_pbake_b, lm_pbake_b_text, lm_pbake_b_spk, lm_pbake_b_diff, lm_pbake_b_io,
                    lm_use_partial,
                    lm_pg_text_m, lm_pg_text_a, lm_pg_text_la, lm_pg_text_lb,
                    lm_pg_spk_m,  lm_pg_spk_a,  lm_pg_spk_la,  lm_pg_spk_lb,
                    lm_pg_diff_m, lm_pg_diff_a, lm_pg_diff_la, lm_pg_diff_lb,
                    lm_pg_io_m,   lm_pg_io_a,   lm_pg_io_la,   lm_pg_io_lb,
                    lm_output_format, lm_output_dir,
                ]
                lm_run_btn.click(_run_lm_bake_ui, inputs=_lm_inputs, outputs=[lm_status])
