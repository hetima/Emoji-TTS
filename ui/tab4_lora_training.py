import subprocess
import os
import sys
import threading
from datetime import datetime
import gradio as gr
from ui.common import *
import gradio_conf as cnf

_LORA_TRAIN_LOG_LOCK = threading.Lock()
_LORA_TRAIN_PROC: subprocess.Popen | None = None
_LORA_TRAIN_LOG_PATH: Path | None = None
# ETA推定用: {"speed": steps/sec, "eta_sec": float, "step": int, "max_steps": int}
_LORA_ETA_INFO: dict = {}

# ─────────────────────────────
# LoRAプリセット用ユーティリティ  ← 追加
# ─────────────────────────────

def _scan_lora_configs() -> list[str]:
    """configs/ 配下のYAMLのうち 'lora' セクションを持つものを列挙。"""
    cnf.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(cnf.CONFIGS_DIR.glob("*.yaml")) + sorted(cnf.CONFIGS_DIR.glob("*.yml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if "lora" in data:
                result.append(str(p))
        except Exception:
            pass
    return result


def _lora_config_from_ui(
    base_model, manifest, output_dir, run_name,
    lora_rank, lora_alpha, lora_dropout, target_modules,
    save_mode, attention_backend,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_lora_path,
    batch_size, grad_accum, lr, optimizer,
    lr_scheduler, warmup_steps,
    max_steps, save_every, log_every,
    valid_ratio, valid_every,
    wandb_enabled, wandb_project, wandb_run_name,
    seed,
) -> dict:
    return {
        "lora": {
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "target_modules": str(target_modules),
            "save_mode": str(save_mode),
            "attention_backend": str(attention_backend),
            "use_early_stopping": bool(use_early_stopping),
            "es_patience": int(es_patience),
            "es_min_delta": float(es_min_delta),
            "use_ema": bool(use_ema),
            "ema_decay": float(ema_decay),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "lr": float(lr),
            "optimizer": str(optimizer),
            "lr_scheduler": str(lr_scheduler),
            "warmup_steps": int(warmup_steps),
            "max_steps": int(max_steps),
            "save_every": int(save_every),
            "log_every": int(log_every),
            "valid_ratio": float(valid_ratio),
            "valid_every": int(valid_every),
            "wandb_enabled": bool(wandb_enabled),
            "wandb_project": str(wandb_project) if wandb_project else "",
            "wandb_run_name": str(wandb_run_name) if wandb_run_name else "",
            "seed": int(seed),
        }
    }


def _save_lora_config(config_name: str, data: dict) -> str:
    cnf.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(config_name)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        p = cnf.CONFIGS_DIR / p.name
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"保存しました: {p}"


def _load_lora_config(config_path: str) -> dict:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("lora", {}) # type: ignore


def _load_lora_preset(config_path: str):
    if not config_path:
        return (
            16, 32.0, 0.05, "wq,wk,wv,wo",
            "EMAのみ", "sdpa",
            False, 3, 0.01,
            True, 0.9999,
            4, 1, 1e-4, "adamw",
            "none", 0,
            1000, 100, 10,
            0.0, 100,
            False, "", "",
            0,
        )
    cfg = _load_lora_config(config_path)
    def g(k, fb):
        return cfg.get(k, fb)
    return (
        g("lora_rank", 16),
        g("lora_alpha", 32.0),
        g("lora_dropout", 0.05),
        g("target_modules", "wq,wk,wv,wo"),
        g("save_mode", "EMAのみ"),
        g("attention_backend", "sdpa"),
        g("use_early_stopping", False),
        g("es_patience", 3),
        g("es_min_delta", 0.01),
        g("use_ema", True),
        g("ema_decay", 0.9999),
        g("batch_size", 4),
        g("grad_accum", 1),
        g("lr", 1e-4),
        g("optimizer", "adamw"),
        g("lr_scheduler", "none"),
        g("warmup_steps", 0),
        g("max_steps", 1000),
        g("save_every", 100),
        g("log_every", 10),
        g("valid_ratio", 0.0),
        g("valid_every", 100),
        g("wandb_enabled", False),
        g("wandb_project", "") or "",
        g("wandb_run_name", "") or "",
        g("seed", 0),
    )


def _save_lora_preset(name: str, *cfg_args):
    cfg_data = _lora_config_from_ui(*cfg_args)
    return _save_lora_config(name, cfg_data)



# ─────────────────────────────
# LoRA 学習タブ ロジック
# ─────────────────────────────

def _build_lora_train_command(
    base_model, manifest, output_dir, run_name,
    lora_rank, lora_alpha, lora_dropout, target_modules,
    save_mode, attention_backend,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_lora_path,
    batch_size, grad_accum, lr, optimizer, lr_scheduler, warmup_steps,
    max_steps, save_every, log_every,
    valid_ratio, valid_every,
    wandb_enabled, wandb_project, wandb_run_name,
    seed,
) -> list[str]:
    cmd = [sys.executable, str(cnf.REPO_DIR / "lora_train.py")]
    cmd += ["--base-model", str(base_model)]
    cmd += ["--manifest", str(manifest)]

    _run_name = str(run_name).strip() if str(run_name).strip() else ""
    if _run_name:
        cmd += ["--run-name", _run_name]

    if str(output_dir).strip():
        cmd += ["--output-dir", str(output_dir).strip()]

    cmd += [
        "--lora-rank", str(int(lora_rank)),
        "--lora-alpha", str(float(lora_alpha)),
        "--lora-dropout", str(float(lora_dropout)),
        "--target-modules", str(target_modules).strip(),
        "--batch-size", str(int(batch_size)),
        "--gradient-accumulation-steps", str(int(grad_accum)),
        "--lr", str(float(lr)),
        "--optimizer", str(optimizer),
        "--lr-scheduler", str(lr_scheduler),
        "--warmup-steps", str(int(warmup_steps)),
        "--max-steps", str(int(max_steps)),
        "--save-every", str(int(save_every)),
        "--log-every", str(int(log_every)),
        "--seed", str(int(seed)),
    ]

    if str(attention_backend) != "sdpa":
        cmd += ["--attention-backend", str(attention_backend)]

    if str(save_mode) == "EMA + Full両方":
        cmd += ["--save-full"]

    if use_ema:
        cmd += ["--ema-decay", str(float(ema_decay))]

    if float(valid_ratio) > 0.0:
        cmd += ["--valid-ratio", str(float(valid_ratio))]
        if int(valid_every) > 0:
            cmd += ["--valid-every", str(int(valid_every))]

    if use_early_stopping and float(valid_ratio) > 0.0:
        cmd += [
            "--early-stopping",
            "--early-stopping-patience", str(int(es_patience)),
            "--early-stopping-min-delta", str(float(es_min_delta)),
        ]

    if wandb_enabled:
        cmd += ["--wandb"]
        if str(wandb_project).strip():
            cmd += ["--wandb-project", str(wandb_project).strip()]
        if str(wandb_run_name).strip():
            cmd += ["--wandb-run-name", str(wandb_run_name).strip()]

    if resume_enabled and str(resume_lora_path).strip():
        cmd += ["--resume-lora", str(resume_lora_path).strip()]

    return cmd


def _start_lora_train(*args) -> tuple[str, str]:
    global _LORA_TRAIN_PROC, _LORA_TRAIN_LOG_PATH, _LORA_ETA_INFO

    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is not None and _LORA_TRAIN_PROC.poll() is None:
            return "LoRA学習が既に実行中です。停止してから再実行してください。", ""

    cmd_list = _build_lora_train_command(*args)
    cmd_str = " ".join(cmd_list)

    # _build_lora_train_command のシグネチャ順 (0-indexed):
    # 0:base_model 1:manifest 2:output_dir 3:run_name 4:lora_rank 5:lora_alpha
    # 6:lora_dropout 7:target_modules 8:save_mode 9:attention_backend
    # 10:use_early_stopping 11:es_patience 12:es_min_delta 13:use_ema 14:ema_decay
    # 15:resume_enabled 16:resume_lora_path 17:batch_size 18:grad_accum 19:lr
    # 20:optimizer 21:lr_scheduler 22:warmup_steps 23:max_steps ...
    try:
        _max_steps = int(args[23])
    except (IndexError, ValueError, TypeError):
        _max_steps = 0

    cnf.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cnf.LOGS_DIR / f"lora_train_{stamp}.log"

    _LORA_ETA_INFO.clear()

    with _LORA_TRAIN_LOG_LOCK:
        _LORA_TRAIN_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _LORA_TRAIN_PROC = proc

    import re as _re_eta
    _STEP_RE = _re_eta.compile(r"step=(\d+)")
    import re as _re_speed
    _SPEED_RE = _re_speed.compile(r"speed=([0-9.]+)steps/s")
    _ETA_STR_RE = _re_speed.compile(r"eta=(.+)")

    def _eta_str_to_sec(eta_str: str) -> float:
        """lora_train.py が出力する eta= 文字列を秒数に変換する。
        フォーマット: 〇時間〇分 / 〇分〇秒 / 〇秒
        """
        import re as _r
        s = eta_str.strip()
        total = 0.0
        m = _r.search(r"(\d+)時間", s)
        if m:
            total += int(m.group(1)) * 3600
        m = _r.search(r"(\d+)分", s)
        if m:
            total += int(m.group(1)) * 60
        m = _r.search(r"(\d+)秒", s)
        if m:
            total += int(m.group(1))
        return total

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout: # type: ignore
                f.write(line)
                f.flush()
                # ステップ行からstep/speed/etaを直接パースして更新
                # lora_train.py が計算した値をそのまま使うことで二重計算のズレを排除
                if "step=" in line and "loss=" in line:
                    m_step = _STEP_RE.search(line)
                    m_speed = _SPEED_RE.search(line)
                    m_eta = _ETA_STR_RE.search(line)
                    if m_step and _max_steps > 0:
                        current_step = int(m_step.group(1))
                        speed = float(m_speed.group(1)) if m_speed else 0.0
                        eta_sec = _eta_str_to_sec(m_eta.group(1)) if m_eta else 0.0
                        _LORA_ETA_INFO.update({
                            "step": current_step,
                            "max_steps": _max_steps,
                            "speed": speed,
                            "eta_sec": eta_sec,
                        })
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()

    warning = ""
    if bool(args[10]) and float(args[26]) <= 0.0:
        warning = "\n⚠️ Early Stopping は valid_ratio=0 のため無効化されました。"
    return f"LoRA学習開始 (PID {proc.pid})\nログ: {log_path}{warning}", cmd_str


def _stop_lora_train() -> str:
    global _LORA_TRAIN_PROC
    import signal as _signal
    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is None or _LORA_TRAIN_PROC.poll() is not None:
            return "実行中のLoRA学習プロセスはありません。"
        pid = _LORA_TRAIN_PROC.pid
        proc = _LORA_TRAIN_PROC
    # SIGINTでグレースフルシャットダウンを試みる（DataLoaderが正常終了できる）
    try:
        import os as _os
        _os.kill(pid, _signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    # 最大5秒待機し、まだ生きていれば強制終了
    def _deferred_kill():
        import time as _t
        _t.sleep(5)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    import threading as _thr
    _thr.Thread(target=_deferred_kill, daemon=True).start()
    return f"LoRA学習プロセス (PID {pid}) に停止シグナルを送信しました（最大5秒でシャットダウン）。"


def _read_lora_train_log() -> str:
    with _LORA_TRAIN_LOG_LOCK:
        path = _LORA_TRAIN_LOG_PATH
        proc = _LORA_TRAIN_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        text += f"\n\n--- LoRA学習終了 (returncode={proc.returncode}) ---"
    lines = text.splitlines()
    if len(lines) > 200:
        text = "... （先頭省略、末尾200行表示）\n" + "\n".join(lines[-200:])
    # ETA情報を末尾に付加（学習中のみ）
    eta = _LORA_ETA_INFO
    if eta and proc is not None and proc.poll() is None:
        step = eta.get("step", 0)
        max_steps = eta.get("max_steps", 0)
        speed = eta.get("speed", 0.0)
        eta_sec = int(eta.get("eta_sec", 0.0))
        h, rem = divmod(eta_sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            eta_str = f"{h}時間{m}分"
        elif m > 0:
            eta_str = f"{m}分{s}秒"
        else:
            eta_str = f"{s}秒"
        progress_pct = (step / max_steps * 100) if max_steps > 0 else 0.0
        text += (
            f"\n\n--- ETA: 残り約 {eta_str}"
            f"  ({step}/{max_steps} steps, {progress_pct:.1f}%,"
            f" {speed:.3f} steps/sec) ---"
        )
    return text

# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx):
    with gr.Tab("🚀 LoRA学習"):
        gr.Markdown(
            "## LoRA 差分学習\n"
            "ベースモデルに対して LoRA アダプタを学習します。\n\n"
            "> **必要ライブラリ**: `pip install peft`"
        )

        with gr.Accordion("プリセット管理（configs/ フォルダ）", open=True):
            with gr.Row():
                lora_preset_dropdown = gr.Dropdown(
                    label="プリセット選択",
                    choices=_scan_lora_configs(),
                    value=None,
                    scale=3,
                )
                lora_preset_refresh_btn = gr.Button("更新", scale=1)
            with gr.Row():
                lora_preset_name_input = gr.Textbox(
                    label="保存ファイル名（例: my_lora.yaml）",
                    value="my_lora.yaml",
                    scale=3,
                )
                lora_preset_save_btn = gr.Button("保存", scale=1)
            lora_preset_status = gr.Textbox(label="プリセット操作結果", interactive=False, lines=1)

        with gr.Row():
            lora_base_model = gr.Dropdown(
                label="ベースモデル (.pt / .safetensors)",
                choices=ctx.initial_checkpoints,
                value=(
                    str(cnf.CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors")
                    if (cnf.CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors").exists()
                    else (ctx.initial_checkpoints[-1] if ctx.initial_checkpoints else None)
                ),
                allow_custom_value=True, scale=4,
            )
            lora_base_refresh_btn = gr.Button("更新", scale=1)

        with gr.Row():
            lora_manifest = gr.Dropdown(
                label="マニフェストファイル (.jsonl)",
                choices=ctx.initial_manifests,
                value=ctx.initial_manifests[-1] if ctx.initial_manifests else None,
                allow_custom_value=True, scale=4,
            )
            lora_manifest_refresh_btn = gr.Button("更新", scale=1)

        with gr.Row():
            lora_run_name = gr.Textbox(
                label="実行名 (run_name)（空欄=タイムスタンプ自動生成）",
                value="", placeholder="my_lora_run", scale=2,
            )
            lora_output_dir = gr.Textbox(
                label="LoRA出力フォルダ（空欄=lora/{run_name}/）",
                value="", placeholder=str(cnf.LORA_DIR / "{run_name}"), scale=2,
            )

        with gr.Row():
            lora_save_mode = gr.Dropdown(
                label="保存モード",
                choices=["EMAのみ", "EMA + Full両方"],
                value="EMAのみ",
                info="EMAのみ=推論専用 / EMA+Full=Resume前提",
            )
            lora_attention_backend = gr.Dropdown(
                label="Attention Backend",
                choices=["sdpa", "flash2", "sage", "eager"],
                value="sdpa",
                info="sdpa=推奨 / flash2・sage=別途インストール必要",
            )

        with gr.Accordion("LoRA設定", open=True):
            with gr.Row():
                lora_rank = gr.Slider(label="LoRAランク", minimum=1, maximum=128, value=16, step=1)
                lora_alpha = gr.Number(label="lora_alpha", value=32.0)
                lora_dropout = gr.Slider(label="lora_dropout", minimum=0.0, maximum=0.5, value=0.05, step=0.01)
            lora_target_modules = gr.Textbox(
                label="ターゲットモジュール（カンマ区切り）",
                value="wq,wk,wv,wo",
                info="デフォルト: wq,wk,wv,wo / 拡張: wq,wk,wv,wo,wk_text,wv_text,wk_speaker,wv_speaker,w1,w2,w3",
            )

        with gr.Accordion("Resume設定", open=False):
            lora_resume_enabled = gr.Checkbox(label="Resume（既存LoRAから再開）", value=False)
            with gr.Row():
                lora_resume_path = gr.Dropdown(
                    label="既存LoRAフォルダ（_full推奨）",
                    choices=scan_lora_adapters(),
                    value=None, allow_custom_value=True, scale=4,
                )
                lora_resume_refresh_btn = gr.Button("更新", scale=1)
            lora_resume_warning = gr.Markdown(visible=False)

            def _on_lora_resume_path_change(path):
                if path and "_ema" in str(path):
                    return gr.update(visible=True, value=(
                        "⚠️ **_ema フォルダを選択しています。**\n\n"
                        "EMA版にはoptimizer状態・step数が含まれないため、"
                        "学習は step=0 から再スタートします。\n"
                        "学習率ウォームアップが再度かかり、学習曲線が不連続になります。\n\n"
                        "中断した学習を完全に再開する場合は **_full フォルダ** を選択してください。"
                    ))
                return gr.update(visible=False)

            lora_resume_path.change(
                _on_lora_resume_path_change,
                inputs=[lora_resume_path], outputs=[lora_resume_warning],
            )
            lora_resume_refresh_btn.click(
                lambda: gr.Dropdown(choices=scan_lora_adapters()),
                outputs=[lora_resume_path],
            )

        with gr.Accordion("学習パラメータ", open=True):
            with gr.Row():
                lora_batch_size = gr.Slider(label="バッチサイズ", minimum=1, maximum=32, value=4, step=1)
                lora_grad_accum = gr.Slider(label="勾配蓄積ステップ", minimum=1, maximum=16, value=1, step=1)
            with gr.Row():
                lora_lr = gr.Number(label="学習率", value=1e-4)
                lora_optimizer = gr.Dropdown(
                    label="オプティマイザ", choices=["adamw", "muon", "lion", "ademamix"],
                    value="adamw",
                )
                lora_lr_scheduler = gr.Dropdown(
                    label="スケジューラ", choices=["none", "cosine", "wsd"], value="none",
                )
                lora_warmup_steps = gr.Number(label="ウォームアップステップ", value=0, precision=0)
            with gr.Row():
                lora_max_steps = gr.Number(label="最大学習ステップ", value=1000, precision=0)
                lora_save_every = gr.Number(label="保存間隔", value=100, precision=0)
                lora_log_every = gr.Number(label="ログ間隔", value=10, precision=0)

        with gr.Accordion("EMA設定", open=False):
            with gr.Row():
                lora_use_ema = gr.Checkbox(label="EMAを有効化", value=True)
                lora_ema_decay = gr.Number(label="EMA減衰率", value=0.9999)

        with gr.Accordion("バリデーション設定", open=False):
            with gr.Row():
                lora_valid_ratio = gr.Slider(label="バリデーション分割比率", minimum=0.0, maximum=0.5, value=0.0, step=0.01)
                lora_valid_every = gr.Number(label="バリデーション実行間隔", value=100, precision=0)

        with gr.Accordion("Early Stopping設定", open=False):
            with gr.Row():
                lora_early_stopping = gr.Checkbox(label="Early Stoppingを有効化", value=False)
                lora_es_patience = gr.Number(label="パティエンス", value=3, precision=0)
                lora_es_min_delta = gr.Number(label="最小悪化量", value=0.01)

        with gr.Accordion("W&B設定", open=False):
            with gr.Row():
                lora_wandb_enabled = gr.Checkbox(label="W&Bを有効化", value=False)
                lora_wandb_project = gr.Textbox(label="W&Bプロジェクト名", value="")
                lora_wandb_run_name = gr.Textbox(label="W&B実行名（省略可）", value="")

        lora_seed = gr.Number(label="乱数シード", value=0, precision=0)

        gr.Markdown("### 実行コマンドプレビュー")
        lora_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

        with gr.Row():
            lora_start_btn = gr.Button("LoRA学習開始", variant="primary", size="lg")
            lora_stop_btn = gr.Button("停止", variant="stop")
        lora_train_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

        gr.Markdown("### 学習ログ")
        with gr.Row():
            lora_log_interval = gr.Slider(label="自動更新間隔（秒）", minimum=2, maximum=60, value=5, step=1, scale=3)
            lora_log_refresh_btn = gr.Button("手動更新", scale=1)
        lora_log_text = gr.Textbox(label="LoRA学習ログ（末尾200行）", interactive=False, lines=15, max_lines=15)

        _lora_exec_inputs = [
            lora_base_model, lora_manifest, lora_output_dir, lora_run_name,
            lora_rank, lora_alpha, lora_dropout, lora_target_modules,
            lora_save_mode, lora_attention_backend,
            lora_early_stopping, lora_es_patience, lora_es_min_delta,
            lora_use_ema, lora_ema_decay,
            lora_resume_enabled, lora_resume_path,
            lora_batch_size, lora_grad_accum, lora_lr, lora_optimizer,
            lora_lr_scheduler, lora_warmup_steps,
            lora_max_steps, lora_save_every, lora_log_every,
            lora_valid_ratio, lora_valid_every,
            lora_wandb_enabled, lora_wandb_project, lora_wandb_run_name,
            lora_seed,
        ]

        _lora_preset_outputs = [
            lora_rank, lora_alpha, lora_dropout, lora_target_modules,
            lora_save_mode, lora_attention_backend,
            lora_early_stopping, lora_es_patience, lora_es_min_delta,
            lora_use_ema, lora_ema_decay,
            lora_batch_size, lora_grad_accum, lora_lr, lora_optimizer,
            lora_lr_scheduler, lora_warmup_steps,
            lora_max_steps, lora_save_every, lora_log_every,
            lora_valid_ratio, lora_valid_every,
            lora_wandb_enabled, lora_wandb_project, lora_wandb_run_name,
            lora_seed,
        ]

        def _update_lora_cmd(*args):
            try:
                return " ".join(_build_lora_train_command(*args))
            except Exception as e:
                return f"(プレビュー生成エラー: {e})"

        for comp in [lora_base_model, lora_manifest, lora_output_dir, lora_run_name,
                      lora_rank, lora_alpha, lora_dropout, lora_target_modules,
                      lora_save_mode, lora_attention_backend,
                      lora_use_ema, lora_ema_decay, lora_max_steps]:
            comp.change(_update_lora_cmd, inputs=_lora_exec_inputs, outputs=[lora_cmd_preview])

        lora_preset_dropdown.change(
            _load_lora_preset,
            inputs=[lora_preset_dropdown],
            outputs=_lora_preset_outputs,
        )
        lora_preset_refresh_btn.click(
            lambda: gr.Dropdown(choices=_scan_lora_configs(), value=None),
            outputs=[lora_preset_dropdown],
        )
        lora_preset_save_btn.click(
            _save_lora_preset,
            inputs=[lora_preset_name_input] + _lora_exec_inputs,
            outputs=[lora_preset_status],
        )

        lora_base_refresh_btn.click(
            lambda: gr.Dropdown(choices=scan_checkpoints(), value=(scan_checkpoints() or [None])[-1]),
            outputs=[lora_base_model],
        )
        lora_manifest_refresh_btn.click(
            lambda: gr.Dropdown(choices=scan_manifests(), value=(scan_manifests() or [None])[-1]),
            outputs=[lora_manifest],
        )
        lora_start_btn.click(_start_lora_train, inputs=_lora_exec_inputs,
                             outputs=[lora_train_status, lora_cmd_preview])
        lora_stop_btn.click(_stop_lora_train, outputs=[lora_train_status])
        lora_log_refresh_btn.click(_read_lora_train_log, outputs=[lora_log_text])
        _lora_timer = gr.Timer(value=5, active=True)
        _lora_timer.tick(_read_lora_train_log, outputs=[lora_log_text])
        lora_log_interval.change(lambda v: float(v), inputs=[lora_log_interval], outputs=[_lora_timer])
