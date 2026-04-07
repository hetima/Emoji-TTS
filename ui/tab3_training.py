from __future__ import annotations
import subprocess
import threading
import sys
import os
from datetime import datetime
import gradio as gr
from ui.common import *
import gradio_conf as cnf



# ─────────────────────────────
# 学習タブ ロジック
# ─────────────────────────────

_TRAIN_LOG_PATH: Path | None = None
_TRAIN_PROC: subprocess.Popen | None = None
_TRAIN_LOG_LOCK = threading.Lock()
# ETA推定用: {"speed": steps/sec, "eta_sec": float, "step": int, "max_steps": int}
_TRAIN_ETA_INFO: dict = {}


def _save_yaml_config(config_path: str, data: dict) -> str:
    cnf.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(config_path)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        p = cnf.CONFIGS_DIR / p.name
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"保存しました: {p}"


def _config_from_ui(
    manifest, output_dir,
    batch_size, grad_accum, num_workers, persistent_workers, prefetch_factor,
    allow_tf32, compile_model, precision,
    optimizer, muon_momentum, learning_rate, weight_decay,
    adam_beta1, adam_beta2, adam_eps,
    lr_scheduler, warmup_steps, stable_steps, min_lr_scale,
    max_steps, max_text_len,
    text_dropout, speaker_dropout, timestep_stratified,
    max_latent_steps, fixed_target_latent_steps, fixed_target_full_mask,
    log_every, save_every,
    wandb_enabled, wandb_project, wandb_run_name,
    valid_ratio, valid_every,
    early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    seed,
) -> dict:
    return {
        "train": {
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "num_workers": int(num_workers),
            "dataloader_persistent_workers": bool(persistent_workers),
            "dataloader_prefetch_factor": int(prefetch_factor),
            "allow_tf32": bool(allow_tf32),
            "compile_model": bool(compile_model),
            "precision": str(precision),
            "optimizer": str(optimizer),
            "muon_momentum": float(muon_momentum),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "adam_beta1": float(adam_beta1),
            "adam_beta2": float(adam_beta2),
            "adam_eps": float(adam_eps),
            "lr_scheduler": str(lr_scheduler),
            "warmup_steps": int(warmup_steps),
            "stable_steps": int(stable_steps),
            "min_lr_scale": float(min_lr_scale),
            "max_steps": int(max_steps),
            "max_text_len": int(max_text_len),
            "text_condition_dropout": float(text_dropout),
            "speaker_condition_dropout": float(speaker_dropout),
            "timestep_stratified": bool(timestep_stratified),
            "max_latent_steps": int(max_latent_steps),
            "fixed_target_latent_steps": int(fixed_target_latent_steps),
            "fixed_target_full_mask": bool(fixed_target_full_mask),
            "log_every": int(log_every),
            "save_every": int(save_every),
            "wandb_enabled": bool(wandb_enabled),
            "wandb_project": str(wandb_project) if wandb_project else None,
            "wandb_run_name": str(wandb_run_name) if wandb_run_name else None,
            "valid_ratio": float(valid_ratio),
            "valid_every": int(valid_every),
            "seed": int(seed),
        }
    }


def _build_train_command(
    manifest, output_dir, config_path,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_checkpoint,
    save_mode,
    num_gpus,
    attention_backend="sdpa",
) -> list[str]:
    if int(num_gpus) > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={num_gpus}", str(cnf.REPO_DIR / "train.py")]
    else:
        cmd = [sys.executable, str(cnf.REPO_DIR / "train.py")]

    cmd += [
        "--config", str(config_path),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    if use_early_stopping:
        cmd += ["--early-stopping",
                "--early-stopping-patience", str(es_patience),
                "--early-stopping-min-delta", str(es_min_delta)]
    if use_ema:
        cmd += ["--ema-decay", str(ema_decay)]

    if resume_enabled and str(resume_checkpoint).strip():
        cmd += ["--resume", str(resume_checkpoint)]

    if str(save_mode) in ("Fullのみ", "EMA + Full両方"):
        cmd += ["--save-full"]

    if str(attention_backend) and str(attention_backend) != "sdpa":
        cmd += ["--attention-backend", str(attention_backend)]

    return cmd


def _start_train(
    manifest, output_dir, config_path,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay, resume_enabled, resume_checkpoint, save_mode, num_gpus,
    attention_backend="sdpa",
    *ui_cfg_args,
) -> tuple[str, str]:
    global _TRAIN_LOG_PATH, _TRAIN_PROC, _TRAIN_ETA_INFO

    with _TRAIN_LOG_LOCK:
        if _TRAIN_PROC is not None and _TRAIN_PROC.poll() is None:
            return "学習が既に実行中です。停止してから再実行してください。", ""

    cfg_data = _config_from_ui(*ui_cfg_args)
    tmp_config = cnf.CONFIGS_DIR / "_train_tmp.yaml"
    cnf.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml_config(str(config_path)) if Path(config_path).is_file() else {}
    base_cfg.update(cfg_data)
    with open(tmp_config, "w", encoding="utf-8") as f:
        yaml.dump(base_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    cmd_list = _build_train_command(
        manifest, output_dir, tmp_config,
        use_early_stopping, es_patience, es_min_delta,
        use_ema, ema_decay, resume_enabled, resume_checkpoint, save_mode, num_gpus,
        attention_backend,
    )
    cmd = " ".join(cmd_list)

    cnf.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cnf.LOGS_DIR / f"train_{stamp}.log"

    _TRAIN_ETA_INFO.clear()

    with _TRAIN_LOG_LOCK:
        _TRAIN_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _TRAIN_PROC = proc

    import re as _re_train_eta
    _TRAIN_STEP_RE = _re_train_eta.compile(r"step=(\d+)")
    _TRAIN_SPEED_RE = _re_train_eta.compile(r"speed=([0-9.]+)steps/s")
    _TRAIN_ETA_RE = _re_train_eta.compile(r"eta=(.+)")

    def _train_eta_str_to_sec(eta_str: str) -> float:
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
                # train.py のログ行から speed= / eta= を直接パース
                if "step=" in line and "loss=" in line:
                    m_step = _TRAIN_STEP_RE.search(line)
                    m_speed = _TRAIN_SPEED_RE.search(line)
                    m_eta = _TRAIN_ETA_RE.search(line)
                    if m_step:
                        current_step = int(m_step.group(1))
                        speed = float(m_speed.group(1)) if m_speed else 0.0
                        eta_sec = _train_eta_str_to_sec(m_eta.group(1)) if m_eta else 0.0
                        _TRAIN_ETA_INFO.update({
                            "step": current_step,
                            "speed": speed,
                            "eta_sec": eta_sec,
                        })
        proc.wait()
        _write_tensorboard_events(log_path)

    threading.Thread(target=_stream, daemon=True).start()
    return f"学習開始 (PID {proc.pid})\nログ: {log_path}", cmd


def _stop_train() -> str:
    global _TRAIN_PROC
    import signal as _signal
    with _TRAIN_LOG_LOCK:
        if _TRAIN_PROC is None or _TRAIN_PROC.poll() is not None:
            return "実行中の学習プロセスはありません。"
        pid = _TRAIN_PROC.pid
        proc = _TRAIN_PROC
    try:
        import os as _os
        _os.kill(pid, _signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError):
        pass
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
    return f"学習プロセス (PID {pid}) に停止シグナルを送信しました（最大5秒でシャットダウン）。"


def _read_train_log() -> str:
    with _TRAIN_LOG_LOCK:
        path = _TRAIN_LOG_PATH
        proc = _TRAIN_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        rc = proc.returncode
        text += f"\n\n--- 学習終了 (returncode={rc}) ---"
    lines = text.splitlines()
    if len(lines) > 200:
        text = f"... （先頭省略、末尾200行表示）\n" + "\n".join(lines[-200:])
    # ETA情報を末尾に付加（学習中のみ）
    eta = _TRAIN_ETA_INFO
    if eta and proc is not None and proc.poll() is None:
        step = eta.get("step", 0)
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
        text += (
            f"\n\n--- ETA: 残り約 {eta_str}"
            f"  (step={step}, {speed:.3f} steps/sec) ---"
        )
    return text


def _parse_train_log_metrics():
    if not PANDAS_AVAILABLE:
        return None
    with _TRAIN_LOG_LOCK:
        path = _TRAIN_LOG_PATH
    if path is None or not path.exists():
        return pandas.DataFrame({"step": [], "loss": [], "lr": []}) # type: ignore

    import re as _re_metrics
    # 各フィールドを個別に正規表現で抽出（speed= や eta= が混在しても壊れない）
    _RE_STEP = _re_metrics.compile(r"\bstep=(\d+)")
    _RE_LOSS = _re_metrics.compile(r"\bloss=([0-9.eE+\-]+)")
    _RE_LR   = _re_metrics.compile(r"\blr=([0-9.eE+\-]+)")

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "step=" not in line or "loss=" not in line:
            continue
        # valid行・EarlyStopping行などのメトリクス以外の行を除外
        stripped = line.lstrip()
        if stripped.startswith("valid") or stripped.startswith("EarlyStopping"):
            continue
        try:
            m_step = _RE_STEP.search(line)
            m_loss = _RE_LOSS.search(line)
            m_lr   = _RE_LR.search(line)
            if not m_step or not m_loss:
                continue
            step = int(m_step.group(1))
            loss = float(m_loss.group(1))
            lr   = float(m_lr.group(1)) if m_lr else 0.0
            rows.append({"step": step, "loss": loss, "lr": lr})
        except (ValueError, AttributeError):
            continue
    if not rows:
        return pandas.DataFrame({"step": [], "loss": [], "lr": []}) # type: ignore
    return pandas.DataFrame(rows) # type: ignore


def _write_tensorboard_events(log_path: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = cnf.LOGS_DIR / "tensorboard" / log_path.stem
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        df = _parse_train_log_metrics()
        for _, row in df.iterrows(): # type: ignore
            writer.add_scalar("train/loss", row["loss"], int(row["step"]))
            writer.add_scalar("train/lr",   row["lr"],   int(row["step"]))
        writer.close()
        print(f"[gradio] TensorBoardイベント保存: {tb_dir}", flush=True)
    except ImportError:
        pass
    finally:
        df = _parse_train_log_metrics()
        if not df.empty: # type: ignore
            csv_path = cnf.LOGS_DIR / f"{log_path.stem}_metrics.csv"
            df.to_csv(csv_path, index=False) # type: ignore
            print(f"[gradio] メトリクスCSV保存: {csv_path}", flush=True)


# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx):
    with gr.Tab("🏋️ 学習"):
        gr.Markdown("## 学習設定")

        with gr.Accordion("プリセット管理（configs/ フォルダ）", open=True):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="プリセット選択", choices=ctx.initial_configs,
                    value=ctx.default_config or None, scale=3,
                )
                preset_refresh_btn = gr.Button("更新", scale=1)
            with gr.Row():
                preset_name_input = gr.Textbox(
                    label="保存ファイル名（例: my_config.yaml）",
                    value="my_config.yaml", scale=3,
                )
                preset_save_btn = gr.Button("保存", scale=1)
            preset_status = gr.Textbox(label="プリセット操作結果", interactive=False, lines=1)

        with gr.Row():
            train_manifest = gr.Dropdown(
                label="マニフェストファイル (.jsonl)",
                choices=ctx.initial_manifests,
                value=ctx.initial_manifests[-1] if ctx.initial_manifests else None,
                allow_custom_value=True, scale=3,
            )
            train_manifest_refresh = gr.Button("更新", scale=1)
        train_output_dir = gr.Textbox(
            label="学習出力フォルダ（チェックポイント保存先）",
            value=str(cnf.OUTPUTS_DIR / "irodori_tts"),
        )

        with gr.Row():
            num_gpus = gr.Slider(label="GPU数（1=単体, 複数=DDP）", minimum=1, maximum=8, value=1, step=1)
            save_mode = gr.Dropdown(
                label="保存ファイル形式",
                choices=["EMAのみ", "Fullのみ", "EMA + Full両方"],
                value="EMAのみ",
                info="EMA=推論用軽量版、Full=追加学習用（optimizer状態含む）",
            )
            train_attention_backend = gr.Dropdown(
                label="Attention Backend",
                choices=["sdpa", "flash2", "sage", "eager"],
                value="sdpa",
                info="sdpa=推奨 / flash2=FlashAttention2要インストール / sage=SageAttention要インストール",
            )

        with gr.Accordion("ベースモデル・追加学習設定", open=True):
            _default_safetensors = str(
                cnf.CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors"
            )
            gr.Markdown(
                "**--resume オプション設定**\n\n"
                "- **オフ（スクラッチ学習）**: モデルを最初からランダム初期化して学習します。\n"
                "- **オン・パス未入力**: `checkpoints/Aratako_Irodori-TTS-500M-v2/model.safetensors` が"
                "存在すれば自動でロードして追加学習します（デフォルト動作）。\n"
                "- **オン・パス入力**: 指定したファイルをベースに追加学習します。"
                "`.safetensors` を指定するとstep=0から学習開始、`.pt` チェックポイントを指定すると"
                "step/optimizer状態を引き継いで再開します。"
            )
            with gr.Row():
                resume_enabled = gr.Checkbox(
                    label="--resume を有効にする（追加学習 / チェックポイント再開）",
                    value=True,
                    scale=1,
                )
            resume_checkpoint = gr.Textbox(
                label="ベースモデルパス（空欄 = デフォルト自動参照）",
                value="",
                placeholder=_default_safetensors,
                info=f"空欄の場合、resume有効時は {_default_safetensors} を自動参照します。",
            )

        with gr.Accordion("バッチ・精度設定", open=True):
            gr.Markdown("*バッチサイズと勾配蓄積ステップの積が実効バッチサイズになります。*")
            with gr.Row():
                t_batch_size  = gr.Slider(label="バッチサイズ（GPUメモリに合わせて調整）", minimum=1, maximum=64, value=ctx.v("batch_size", 4), step=1)
                t_grad_accum  = gr.Slider(label="勾配蓄積ステップ数（実効バッチを増やす）", minimum=1, maximum=32, value=ctx.v("gradient_accumulation_steps", 2), step=1)
                t_num_workers = gr.Slider(label="DataLoaderワーカー数", minimum=0, maximum=16, value=ctx.v("num_workers", 4), step=1)
            with gr.Row():
                t_persistent_workers = gr.Checkbox(label="ワーカーの永続化（起動高速化）", value=ctx.v("dataloader_persistent_workers", True))
                t_prefetch_factor    = gr.Slider(label="プリフェッチ係数", minimum=1, maximum=8, value=ctx.v("dataloader_prefetch_factor", 2), step=1)
                t_allow_tf32         = gr.Checkbox(label="TF32を許可（Ampere以降GPU向け高速化）", value=ctx.v("allow_tf32", True))
                t_compile_model      = gr.Checkbox(label="torch.compileを使用（初回遅延あり）", value=ctx.v("compile_model", False))
                t_precision          = gr.Dropdown(label="学習精度", choices=["bf16", "fp32", "fp16"], value=ctx.v("precision", "bf16"))

        with gr.Accordion("オプティマイザ設定", open=True):
            gr.Markdown("*Muon: 行列重み向けの高性能オプティマイザ。AdamW: 汎用的で安定。*")
            with gr.Row():
                t_optimizer    = gr.Dropdown(label="オプティマイザ", choices=["muon", "adamw", "lion", "ademamix", "sgd"], value=ctx.v("optimizer", "muon"))
                t_learning_rate= gr.Number(label="学習率", value=ctx.v("learning_rate", 3e-4))
                t_weight_decay = gr.Number(label="重み減衰（L2正則化）", value=ctx.v("weight_decay", 0.01))
            with gr.Row():
                t_muon_momentum= gr.Number(label="Muonモメンタム（Muon使用時のみ有効）", value=ctx.v("muon_momentum", 0.95))
                t_adam_beta1   = gr.Number(label="Adam β1（AdamW使用時）", value=ctx.v("adam_beta1", 0.9))
                t_adam_beta2   = gr.Number(label="Adam β2（AdamW使用時）", value=ctx.v("adam_beta2", 0.999))
                t_adam_eps     = gr.Number(label="Adam ε（AdamW使用時）", value=ctx.v("adam_eps", 1e-8))

        with gr.Accordion("学習率スケジューラ", open=True):
            gr.Markdown("*wsd: warmup→stable→decay の3段階スケジュール。cosine: コサインアニーリング。*")
            with gr.Row():
                t_lr_scheduler  = gr.Dropdown(label="スケジューラ種別", choices=["wsd", "cosine", "none"], value=ctx.v("lr_scheduler", "wsd"))
                t_warmup_steps  = gr.Number(label="ウォームアップステップ数", value=ctx.v("warmup_steps", 300), precision=0)
                t_stable_steps  = gr.Number(label="安定期ステップ数（wsdのみ）", value=ctx.v("stable_steps", 2100), precision=0)
                t_min_lr_scale  = gr.Number(label="最小学習率スケール比率（0〜1）", value=ctx.v("min_lr_scale", 0.01))

        with gr.Accordion("学習ステップ・テキスト設定", open=True):
            with gr.Row():
                t_max_steps             = gr.Number(label="最大学習ステップ数", value=ctx.v("max_steps", 3000), precision=0)
                t_max_text_len          = gr.Number(label="テキスト最大トークン長", value=ctx.v("max_text_len", 256), precision=0)
                t_max_latent_steps      = gr.Number(label="ラテント最大フレーム数", value=ctx.v("max_latent_steps", 750), precision=0)
                t_fixed_target_latent_steps = gr.Number(label="固定ターゲットラテント長", value=ctx.v("fixed_target_latent_steps", 750), precision=0)
                t_fixed_target_full_mask= gr.Checkbox(label="固定ターゲット全マスク", value=ctx.v("fixed_target_full_mask", True))

        with gr.Accordion("Conditioningドロップアウト・タイムステップ", open=False):
            gr.Markdown("*ドロップアウト率を高めると過学習防止。小データセットでは0.1〜0.2推奨。*")
            with gr.Row():
                t_text_dropout      = gr.Slider(label="テキスト条件ドロップアウト率（0=無効）", minimum=0.0, maximum=0.5, value=ctx.v("text_condition_dropout", 0.15), step=0.01)
                t_speaker_dropout   = gr.Slider(label="話者条件ドロップアウト率（0=無効）", minimum=0.0, maximum=0.5, value=ctx.v("speaker_condition_dropout", 0.15), step=0.01)
                t_timestep_stratified= gr.Checkbox(label="タイムステップ層化サンプリング（安定化に有効）", value=ctx.v("timestep_stratified", True))

        with gr.Accordion("グラフ更新・チェックポイント保存設定", open=False):
            with gr.Row():
                t_log_every  = gr.Number(label="グラフ描画間隔（ステップ数）", value=ctx.v("log_every", 10), precision=0,
                                                 info="この間隔でloss/lrをログ出力→グラフに反映。ファイル保存とは無関係。")
                t_save_every = gr.Number(label="チェックポイント保存間隔（ステップ数）", value=ctx.v("save_every", 100), precision=0)

        with gr.Accordion("Weights & Biases 設定", open=False):
            gr.Markdown("*wandb_enabledをオンにするとクラウドでリアルタイム学習曲線を確認できます。*")
            with gr.Row():
                t_wandb_enabled  = gr.Checkbox(label="W&B を有効化", value=ctx.v("wandb_enabled", False))
                t_wandb_project  = gr.Textbox(label="W&B プロジェクト名", value=ctx.v("wandb_project", "") or "")
                t_wandb_run_name = gr.Textbox(label="W&B 実行名（省略可）", value=ctx.v("wandb_run_name", "") or "")

        with gr.Accordion("バリデーション設定", open=False):
            gr.Markdown("*valid_ratioを0より大きくするとバリデーションlossを監視できます。early_stoppingには必須。*")
            with gr.Row():
                t_valid_ratio= gr.Slider(label="バリデーション分割比率（0=無効）", minimum=0.0, maximum=0.5, value=ctx.v("valid_ratio", 0.0), step=0.01)
                t_valid_every= gr.Number(label="バリデーション実行間隔（ステップ数）", value=ctx.v("valid_every", 100), precision=0)

        with gr.Accordion("オプション機能", open=False):
            gr.Markdown("*Early Stoppingはvalid_ratio > 0 のときのみ有効。EMAは推論品質向上に有効。*")
            with gr.Row():
                t_early_stopping = gr.Checkbox(label="Early Stopping を有効化（valid lossが改善しなくなったら自動停止）", value=False)
                t_es_patience    = gr.Number(label="Early Stopping: 悪化を許容する回数", value=3, precision=0)
                t_es_min_delta   = gr.Number(label="Early Stopping: カウント最小悪化量", value=0.01)
            with gr.Row():
                t_use_ema  = gr.Checkbox(label="EMA（指数移動平均）を有効化（推論品質向上）", value=False)
                t_ema_decay= gr.Number(label="EMA減衰率（0に近いほど追従速度が速い）", value=0.9999)
            t_seed = gr.Number(label="乱数シード（再現性のために固定推奨）", value=ctx.v("seed", 0), precision=0)

        gr.Markdown("### 実行コマンドプレビュー")
        train_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

        with gr.Row():
            train_start_btn = gr.Button("▶学習開始", variant="primary", size="lg")
            train_stop_btn  = gr.Button("⏹学習停止", variant="stop")
        train_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

        gr.Markdown("### 学習ログ・グラフ")
        with gr.Row():
            auto_refresh_interval = gr.Slider(
                label="自動更新間隔（秒）",
                minimum=2, maximum=60, value=5, step=1,
                info="学習中にログ・グラフを自動更新する間隔です。",
                scale=3,
            )
            train_log_refresh_btn = gr.Button("手動更新", scale=1)

        train_log_text = gr.Textbox(label="学習ログ（末尾200行）", interactive=False, lines=15, max_lines=15, elem_id="train_log_text")

        gr.HTML("""
<script>
(function() {
    function attachScrollWatcher() {
        var el = document.getElementById('train_log_text');
        if (!el) { setTimeout(attachScrollWatcher, 500); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(attachScrollWatcher, 500); return; }
        var lastVal = ta.value;
        setInterval(function() {
            if (ta.value !== lastVal) {
                lastVal = ta.value;
                ta.scrollTop = ta.scrollHeight;
            }
        }, 300);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachScrollWatcher);
    } else {
        attachScrollWatcher();
    }
})();
</script>
""")

        if PANDAS_AVAILABLE:
            gr.Markdown("*ログ・グラフは自動更新されます（手動更新ボタンでも即時反映可）。*")
            _empty_df = pandas.DataFrame({"step": [], "loss": [], "lr": []})
            with gr.Row():
                loss_plot = gr.LinePlot(
                    value=_empty_df,
                    label="Loss曲線",
                    x="step", y="loss",
                    height=300,
                )
                lr_plot = gr.LinePlot(
                    value=_empty_df,
                    label="学習率曲線",
                    x="step", y="lr",
                    height=300,
                )

            def _do_refresh():
                log = _read_train_log()
                df = _parse_train_log_metrics()
                return log, df, df

            train_log_refresh_btn.click(_do_refresh, outputs=[train_log_text, loss_plot, lr_plot])

            _auto_timer = gr.Timer(value=5, active=True)
            _auto_timer.tick(_do_refresh, outputs=[train_log_text, loss_plot, lr_plot])
            auto_refresh_interval.change(
                lambda v: float(v),
                inputs=[auto_refresh_interval],
                outputs=[_auto_timer],
            )

        else:
            gr.Markdown(
                "⚠️ **グラフ表示には `pandas` が必要です。**\n"
                "`pip install pandas` または `uv add pandas` を実行後に再起動してください。"
            )
            metrics_text = gr.Textbox(label="メトリクス（step / loss / lr）", interactive=False, lines=6)

            def _do_refresh_nopd():
                log = _read_train_log()
                df = _parse_train_log_metrics()
                if df is None:
                    metrics = "（pandas未インストールのためグラフ非表示）"
                elif df.empty:
                    metrics = "（データなし）"
                else:
                    lines = [f"step={int(r['step'])}  loss={r['loss']:.4f}  lr={r['lr']:.2e}"
                             for _, r in df.tail(10).iterrows()]
                    metrics = "\n".join(lines)
                return log, metrics

            train_log_refresh_btn.click(_do_refresh_nopd, outputs=[train_log_text, metrics_text])

            _auto_timer = gr.Timer(value=5, active=True)
            _auto_timer.tick(_do_refresh_nopd, outputs=[train_log_text, metrics_text])
            auto_refresh_interval.change(
                lambda v: float(v),
                inputs=[auto_refresh_interval],
                outputs=[_auto_timer],
            )

        _train_cfg_inputs = [
            train_manifest, train_output_dir,
            t_batch_size, t_grad_accum, t_num_workers, t_persistent_workers, t_prefetch_factor,
            t_allow_tf32, t_compile_model, t_precision,
            t_optimizer, t_muon_momentum, t_learning_rate, t_weight_decay,
            t_adam_beta1, t_adam_beta2, t_adam_eps,
            t_lr_scheduler, t_warmup_steps, t_stable_steps, t_min_lr_scale,
            t_max_steps, t_max_text_len,
            t_text_dropout, t_speaker_dropout, t_timestep_stratified,
            t_max_latent_steps, t_fixed_target_latent_steps, t_fixed_target_full_mask,
            t_log_every, t_save_every,
            t_wandb_enabled, t_wandb_project, t_wandb_run_name,
            t_valid_ratio, t_valid_every,
            t_early_stopping, t_es_patience, t_es_min_delta,
            t_use_ema, t_ema_decay, t_seed,
        ]
        _train_exec_inputs = [
            train_manifest, train_output_dir, preset_dropdown,
            t_early_stopping, t_es_patience, t_es_min_delta,
            t_use_ema, t_ema_decay,
            resume_enabled, resume_checkpoint, save_mode,
            num_gpus,
            train_attention_backend,
        ] + _train_cfg_inputs

        def _update_train_cmd(manifest, output_dir, config_path,
                              use_early_stopping, es_patience, es_min_delta,
                              use_ema, ema_decay,
                              resume_enabled, resume_checkpoint, save_mode,
                              num_gpus, attention_backend, *_rest):
            return _build_train_command(manifest, output_dir, config_path,
                                       use_early_stopping, es_patience, es_min_delta,
                                       use_ema, ema_decay,
                                       resume_enabled, resume_checkpoint, save_mode,
                                       num_gpus, attention_backend)

        for comp in [train_manifest, train_output_dir, preset_dropdown,
                      t_early_stopping, t_es_patience, t_es_min_delta,
                      t_use_ema, t_ema_decay,
                      resume_enabled, resume_checkpoint, save_mode,
                      num_gpus, train_attention_backend]:
            comp.change(_update_train_cmd, inputs=_train_exec_inputs, outputs=[train_cmd_preview])

        def _load_preset(config_path: str):
            cfg = load_yaml_config(config_path).get("train", {})
            def g(k, fb): return cfg.get(k, fb)
            return (
                g("batch_size", 4), g("gradient_accumulation_steps", 2),
                g("num_workers", 4), g("dataloader_persistent_workers", True),
                g("dataloader_prefetch_factor", 2), g("allow_tf32", True),
                g("compile_model", False), g("precision", "bf16"),
                g("optimizer", "muon"), g("muon_momentum", 0.95),
                g("learning_rate", 3e-4), g("weight_decay", 0.01),
                g("adam_beta1", 0.9), g("adam_beta2", 0.999), g("adam_eps", 1e-8),
                g("lr_scheduler", "wsd"), g("warmup_steps", 300),
                g("stable_steps", 2100), g("min_lr_scale", 0.01),
                g("max_steps", 3000), g("max_text_len", 256),
                g("text_condition_dropout", 0.15), g("speaker_condition_dropout", 0.15),
                g("timestep_stratified", True),
                g("max_latent_steps", 750), g("fixed_target_latent_steps", 750),
                g("fixed_target_full_mask", True),
                g("log_every", 10), g("save_every", 100),
                g("wandb_enabled", False), g("wandb_project", "") or "",
                g("wandb_run_name", "") or "",
                g("valid_ratio", 0.0), g("valid_every", 100),
                False, 3, 0.01, False, 0.9999,
                g("seed", 0),
            )

        _preset_outputs = [
            t_batch_size, t_grad_accum, t_num_workers, t_persistent_workers, t_prefetch_factor,
            t_allow_tf32, t_compile_model, t_precision,
            t_optimizer, t_muon_momentum, t_learning_rate, t_weight_decay,
            t_adam_beta1, t_adam_beta2, t_adam_eps,
            t_lr_scheduler, t_warmup_steps, t_stable_steps, t_min_lr_scale,
            t_max_steps, t_max_text_len,
            t_text_dropout, t_speaker_dropout, t_timestep_stratified,
            t_max_latent_steps, t_fixed_target_latent_steps, t_fixed_target_full_mask,
            t_log_every, t_save_every,
            t_wandb_enabled, t_wandb_project, t_wandb_run_name,
            t_valid_ratio, t_valid_every,
            t_early_stopping, t_es_patience, t_es_min_delta,
            t_use_ema, t_ema_decay, t_seed,
        ]

        preset_dropdown.change(_load_preset, inputs=[preset_dropdown], outputs=_preset_outputs)
        preset_refresh_btn.click(
            lambda: gr.Dropdown(choices=scan_configs(), value=ctx.default_config or None),
            outputs=[preset_dropdown],
        )

        def _save_preset(name, *cfg_args):
            cfg_data = _config_from_ui(*cfg_args)
            return _save_yaml_config(name, cfg_data)

        preset_save_btn.click(
            _save_preset,
            inputs=[preset_name_input] + _train_cfg_inputs,
            outputs=[preset_status],
        )

        train_manifest_refresh.click(
            lambda: gr.Dropdown(choices=scan_manifests(), value=(scan_manifests() or [None])[-1]),
            outputs=[train_manifest],
        )

        train_start_btn.click(
            _start_train, inputs=_train_exec_inputs,
            outputs=[train_status, train_cmd_preview],
        )
        train_stop_btn.click(_stop_train, outputs=[train_status])
