#!/usr/bin/env python3
from __future__ import annotations

import argparse

import gradio as gr

import gradio_conf as cnf

from ui.common import (
    default_model_device,
    ensure_default_model,
    load_yaml_config,
    precision_choices_for_device,
    scan_checkpoints,
    scan_configs,
    scan_manifests,
    scan_train_checkpoints,
)
from ui.context import UIContext
from ui.tab1_inference import build as build_tab1
from ui.tab2_prepare_manifest import build as build_tab2
from ui.tab3_training import build as build_tab3
from ui.tab4_lora_training import build as build_tab4
from ui.tab5_dataset import build as build_tab5
from ui.tab6_checkpoint_convert import build as build_tab6
from ui.tab7_model_merge import build as build_tab7
from ui.tab8_lora_merge import build as build_tab8


def build_ui(args: argparse.Namespace) -> gr.Blocks:
    ensure_default_model()

    initial_checkpoints = scan_checkpoints()
    default_checkpoint = initial_checkpoints[-1] if initial_checkpoints else ""
    default_model_device_ = default_model_device()
    default_codec_device_ = default_model_device()
    device_choices = scan_checkpoints()
    from ui.common import list_available_runtime_devices

    device_choices = list_available_runtime_devices()
    model_precision_choices = precision_choices_for_device(default_model_device_)
    codec_precision_choices = precision_choices_for_device(default_codec_device_)
    initial_configs = scan_configs()
    default_config = next(
        (c for c in initial_configs if cnf.DEFAULT_CONFIG in c),
        initial_configs[-1] if initial_configs else "",
    )
    initial_manifests = scan_manifests()
    initial_train_ckpts = scan_train_checkpoints()

    default_cfg = load_yaml_config(default_config).get("train", {}) if default_config else {}

    ctx = UIContext(
        args=args,
        initial_checkpoints=initial_checkpoints,
        default_checkpoint=default_checkpoint,
        default_model_device=default_model_device_,
        default_codec_device=default_codec_device_,
        device_choices=device_choices,
        model_precision_choices=model_precision_choices,
        codec_precision_choices=codec_precision_choices,
        initial_configs=initial_configs,
        default_config=default_config,
        initial_manifests=initial_manifests,
        initial_train_ckpts=initial_train_ckpts,
        default_cfg=default_cfg,
    )

    with gr.Blocks(title="Irodori-TTS GUI") as demo:
        gr.Markdown("# Irodori-TTS GUI")

        with gr.Tabs():
            build_tab1(ctx)
            build_tab2(ctx)
            build_tab3(ctx)
            build_tab4(ctx)
            build_tab5(ctx)
            build_tab6(ctx)
            build_tab7(ctx)
            build_tab8(ctx)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS GUI")
    parser.add_argument(
        "--checkpoint", default=None, help="model path (.pt/.safetensors) or model directory"
    )
    parser.add_argument(
        "--output-dir", default="gradio_outputs", help="output directory (default: gradio_outputs)"
    )
    parser.add_argument("--lora-dir", default="lora", help="lora directory (default: lora)")
    parser.add_argument(
        "--output-prefix", default="sample", help="output file name prefix (default: sample)"
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from pathlib import Path

    allowed_paths = []
    if args.checkpoint:
        cnf.CHECKPOINTS_DIR = Path(args.checkpoint)
    if args.output_dir:
        cnf.OUTPUTS_DIR = Path(args.output_dir)
        allowed_paths = [args.output_dir]
    if args.lora_dir:
        cnf.LORA_DIR = Path(args.lora_dir)

    demo = build_ui(args)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
