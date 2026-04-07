from __future__ import annotations

from datetime import datetime
import json
import gradio as gr
import gradio_conf as cnf

from ui.common import *

# ─────────────────────────────
# スピーカーライブラリ ユーティリティ
# ─────────────────────────────

def _scan_speakers() -> list[str]:
    """speakers/ 配下のキャラクター名を列挙（ref.pt が存在するフォルダのみ）。"""
    cnf.SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    return ["（使用しない）"] + sorted(
        d.name for d in cnf.SPEAKERS_DIR.iterdir()
        if d.is_dir() and (d / "ref.pt").exists()
    )


def _run_create_speaker(
    char_name: str,
    wav_path: str,
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> str:
    """参照WAVをDACVAEエンコードして speakers/{char_name}/ に3ファイルを生成する。"""
    import shutil
    import torch
    import torchaudio

    char_name = str(char_name).strip()
    if not char_name:
        return "エラー: キャラクター名を入力してください。"
    if not wav_path or not Path(wav_path).is_file():
        return "エラー: WAVファイルを選択してください。"

    # ── ガードレール: キャッシュ済み runtime を直接使用 ──────────────
    # get_cached_runtime() による暗黙の再ロードを防止する。
    # キャッシュ未ロードの場合はエラーを返す。
    # キャッシュ済みの場合は codec_repo も含めてそのまま流用する。
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    _cached_runtime = _RUNTIME_CACHE_VALUE
    _cached_key = _RUNTIME_CACHE_KEY

    if _cached_runtime is None:
        return (
            "エラー: モデルが読み込まれていません。\n"
            "先に「モデル読み込み」ボタンを押してください。"
        )

    # checkpoint / device / precision の一致確認（codec_repo はキャッシュから自動使用）
    try:
        runtime_key = _build_runtime_key(
            checkpoint, model_device, model_precision,
            codec_device, codec_precision, False, "（なし）",
            codec_repo=_cached_key.codec_repo, # type: ignore
        )
    except Exception as e:
        return f"エラー: チェックポイントパスが無効です。\n{e}"

    _codec_fields = ("checkpoint", "model_device", "codec_repo", "model_precision",
                     "codec_device", "codec_precision", "enable_watermark")
    _mismatch = [
        f for f in _codec_fields
        if getattr(_cached_key, f, None) != getattr(runtime_key, f, None)
    ]
    if _mismatch:
        return (
            f"エラー: 現在読み込み中のモデル設定と登録パネルの設定が一致しません。\n"
            f"不一致フィールド: {', '.join(_mismatch)}\n"
            "推論タブの設定と一致させてから「モデル読み込み」を実行してください。"
        )

    codec = _cached_runtime.codec

    # モデルバージョン情報を取得してログに含める
    ldim = int(_cached_runtime.model_cfg.latent_dim)
    version_label = "v2" if ldim == 32 else ("v1" if ldim == 128 else f"unknown(dim={ldim})")

    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        return f"エラー: WAV読み込み失敗: {e}"

    # モノラル化
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 最大30秒トリム
    max_samples = int(30.0 * sr)
    trimmed = wav.shape[1] > max_samples
    if trimmed:
        wav = wav[:, :max_samples]

    duration_sec = round(wav.shape[1] / sr, 2)

    try:
        with torch.inference_mode():
            latent = codec.encode_waveform(wav.unsqueeze(0), sample_rate=sr).cpu()
    except Exception as e:
        return f"エラー: DACVAEエンコード失敗: {e}"

    out_dir = cnf.SPEAKERS_DIR / char_name
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(wav_path, out_dir / "ref.wav")
    torch.save(latent, out_dir / "ref.pt")
    (out_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": char_name,
                "duration_sec": duration_sec,
                "latent_shape": list(latent.shape),
                "source_wav": str(wav_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    msg = f"✅ 登録完了: speakers/{char_name}/\n"
    msg += f"  ref.wav / ref.pt / profile.json\n"
    msg += f"  潜在 shape: {tuple(latent.shape)}  ({duration_sec}秒)\n"
    msg += f"  使用モデル: {version_label} (latent_dim={ldim}, codec={_cached_key.codec_repo})" # type: ignore
    if trimmed:
        msg += "\n  （30秒にトリム済み）"
    return msg



# ─────────────────────────────
# 推論タブ ロジック
# ─────────────────────────────


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return float(text)
    except ValueError as exc: raise ValueError(f"{label} must be a float or blank.") from exc

def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return int(text)
    except ValueError as exc: raise ValueError(f"{label} must be an int or blank.") from exc

def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)

def _resolve_checkpoint_path_infer(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("チェックポイントが選択されていません。")
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint}")
        return checkpoint
    raise ValueError(f"サポートされていないファイル形式: {suffix}")

def _peek_latent_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """チェックポイントを軽量に読み取りlatent_dimを返す。失敗時はNone。"""
    try:
        from pathlib import Path as _Path
        import json as _json
        p = _Path(checkpoint_path)
        if p.suffix.lower() == ".safetensors":
            from safetensors import safe_open as _safe_open
            with _safe_open(str(p), framework="pt", device="cpu") as h:
                meta = h.metadata() or {}
            cfg_raw = meta.get("config_json")
            if cfg_raw:
                cfg = _json.loads(cfg_raw)
                return int(cfg["latent_dim"])
        else:
            import torch as _torch
            ckpt = _torch.load(str(p), map_location="cpu", weights_only=True)
            model_cfg = ckpt.get("model_config", {})
            if "latent_dim" in model_cfg:
                return int(model_cfg["latent_dim"])
    except Exception:
        pass
    return None

def _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim"):
    checkpoint_path = _resolve_checkpoint_path_infer(checkpoint)
    lora_path = None
    if str(lora_adapter).strip() and str(lora_adapter).strip() != "（なし）":
        lp = Path(lora_adapter)
        if lp.is_dir() and (lp / "adapter_config.json").exists():
            lora_path = str(lp)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=str(codec_repo),
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
        lora_path=lora_path,
    )

def _load_model(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）") -> tuple[str, str, bool]:
    """モデルをロードしてステータスと自動選択された codec_repo を返す。"""
    # ロード前にlatent_dimを先読みして正しいcodec_repoを決定する
    _raw_cp = _resolve_checkpoint_path_infer(str(checkpoint).strip())
    _ldim_pre = _peek_latent_dim_from_checkpoint(_raw_cp)
    _initial_codec_repo = _codec_repo_for_latent_dim(_ldim_pre) if _ldim_pre is not None else "Aratako/Semantic-DACVAE-Japanese-32dim"
    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter, codec_repo=_initial_codec_repo)
    _, reloaded = get_cached_runtime(runtime_key)
    status = "モデルを読み込みました" if reloaded else "モデルは既にロード済みです（再利用）"

    # ロード済みモデルからバージョン情報を取得
    info = _detect_model_version_from_runtime()
    version_str = ""
    auto_codec_repo = runtime_key.codec_repo
    if info is not None:
        version_label, codec_repo_used, ldim = info
        version_str = f"\nモデルバージョン: {version_label} (latent_dim={ldim})"
        auto_codec_repo = codec_repo_used

    lora_info = f"\nLoRAアダプタ: {runtime_key.lora_path}" if runtime_key.lora_path else ""
    voice_design_enabled = _runtime_uses_voice_design()
    vd_line = "\nvoice_design: enabled (caption conditioning)" if voice_design_enabled else "\nvoice_design: disabled"
    status_text = (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}"
        f"{version_str}\n"
        f"model_device: {runtime_key.model_device} / {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device} / {runtime_key.codec_precision}\n"
        f"codec_repo: {auto_codec_repo}"
        f"{vd_line}"
        f"{lora_info}"
    )
    return status_text, auto_codec_repo, voice_design_enabled

def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "モデルをメモリから解放しました"


def _clear_runtime_cache_ui():
    return (
        _clear_runtime_cache(),
    )

def _run_generation(
    checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
    lora_adapter, lora_scale, lora_disabled_modules_raw,
    text, caption_text, uploaded_audio, spk_ref_latent_path,
    num_steps, seed_raw, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker,
    cfg_scale_caption,
    cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache,
    max_caption_len_raw,
    truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
    num_candidates: int = 1,
    filename_prefix: str = "",
) -> tuple[list[tuple[str, str]], str, str]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    # ロード済みモデルの codec_repo を優先使用（v1/v2 自動対応）
    info = _detect_model_version_from_runtime()
    auto_codec_repo = (
        info[1] if info is not None
        else _codec_repo_for_latent_dim(32)  # フォールバック: v2
    )
    runtime_key = _build_runtime_key(
        checkpoint, model_device, model_precision,
        codec_device, codec_precision, enable_watermark,
        lora_adapter, codec_repo=auto_codec_repo,
    )
    if str(text).strip() == "":
        raise ValueError("テキストを入力してください。")
    
    if not filename_prefix:
        filename_prefix = Path(checkpoint).name.split("_")[0]

    cfg_scale        = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_caption_len  = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor= _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k        = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma    = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    # lora_disabled_modules: カンマ区切り文字列 → tuple[str, ...]
    _disabled_raw = str(lora_disabled_modules_raw).strip() if lora_disabled_modules_raw else ""
    lora_disabled_modules: tuple[str, ...] = (
        tuple(m.strip() for m in _disabled_raw.split(",") if m.strip())
        if _disabled_raw else ()
    )

    # 参照音声の優先順位: スピーカーライブラリ > 直接アップロード > no-reference
    _spk_pt = str(spk_ref_latent_path).strip() if spk_ref_latent_path else ""
    ref_latent_path: str | None = None
    ref_wav: str | None = None

    if _spk_pt and Path(_spk_pt).is_file():
        ref_latent_path = _spk_pt
        no_ref = False
    elif uploaded_audio and str(uploaded_audio).strip():
        ref_wav = str(uploaded_audio)
        no_ref = False
    else:
        no_ref = True

    num_candidates = max(1, int(num_candidates))

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    use_voice_design = bool(getattr(runtime.model_cfg, "use_caption_condition", False))
    caption_value = str(caption_text).strip() if use_voice_design and caption_text is not None else ""

    # モデルバージョン情報をログに記録
    _ver_info = _detect_model_version_from_runtime()
    _ver_str = f"{_ver_info[0]} (latent_dim={_ver_info[2]})" if _ver_info else "unknown"
    stdout_log(f"[gradio] model_version: {_ver_str}")

    cnf.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    gallery_items: list[tuple[str, str]] = []
    all_detail_lines: list[str] = [
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"model_version: {_ver_str}",
    ]
    if runtime_key.lora_path:
        all_detail_lines.append(f"lora: {Path(runtime_key.lora_path).name}")
    last_timing_text = ""

    # ── 共通のsynthesize呼び出しヘルパー ───────────────────────────
    def _synthesize_line(line_text: str, line_seed) -> object:
        return runtime.synthesize(
            SamplingRequest(
                text=str(line_text), ref_wav=ref_wav, ref_latent=ref_latent_path, no_ref=bool(no_ref),
                seconds=cnf.FIXED_SECONDS, max_ref_seconds=30.0, max_text_len=None,
                caption=caption_value or None,
                max_caption_len=max_caption_len,
                num_steps=int(num_steps),
                seed=line_seed,
                cfg_guidance_mode=str(cfg_guidance_mode),
                cfg_scale_text=float(cfg_scale_text),
                cfg_scale_caption=float(cfg_scale_caption),
                cfg_scale_speaker=float(cfg_scale_speaker),
                cfg_scale=cfg_scale, cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma,
                context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale, speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers, trim_tail=True,
                lora_scale=float(lora_scale) if runtime_key.lora_path else 1.0,
                lora_disabled_modules=lora_disabled_modules if runtime_key.lora_path else (),
            ),
            log_fn=stdout_log,
        )

    for i in range(num_candidates):
        candidate_seed = None if seed is None else (seed + i)
        stdout_log(f"[gradio] generating candidate {i + 1}/{num_candidates} ...")

        result = _synthesize_line(str(text), candidate_seed)

        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = save_wav(
            cnf.OUTPUTS_DIR / f"{filename_prefix}_{stamp}_c{i + 1}.wav",
            result.audio.float(), # type: ignore
            result.sample_rate, # type: ignore
        )
        caption = f"候補 {i + 1}  seed={result.used_seed}" # type: ignore
        gallery_items.append((str(out_path), caption))

        all_detail_lines.append(
            f"[候補 {i + 1}] seed={result.used_seed}  saved={out_path}" # type: ignore
        )
        for msg in result.messages: # type: ignore
            all_detail_lines.append(f"  {msg}")

        last_timing_text = _format_timings(result.stage_timings, result.total_to_decode) # type: ignore
        stdout_log(f"[gradio] candidate {i + 1} saved: {out_path}")

    detail_text = "\n".join(all_detail_lines)
    return gallery_items, detail_text, last_timing_text


def _codec_repo_for_latent_dim(latent_dim: int) -> str:
    """latent_dim からデフォルト codec_repo を返す。"""
    if latent_dim == 32:
        return "Aratako/Semantic-DACVAE-Japanese-32dim"
    return "facebook/dacvae-watermarked"


def _detect_model_version_from_runtime() -> tuple[str, str, int] | None:
    """キャッシュ済み runtime からモデルバージョン情報を取得する。
    戻り値: (version_label, codec_repo, latent_dim) または None"""
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE
    runtime = _RUNTIME_CACHE_VALUE
    if runtime is None:
        return None
    ldim = int(runtime.model_cfg.latent_dim)
    version = "v2" if ldim == 32 else ("v1" if ldim == 128 else f"unknown(dim={ldim})")
    return version, runtime.key.codec_repo, ldim


def _validate_lora_compat_ui(lora_adapter: str) -> str:
    """
    LoRAアダプタとロード済みモデルの互換性をUIから検証する。
    戻り値: 状態メッセージ文字列（エラー時は ❌ プレフィックス）
    """
    import json as _json
    if not lora_adapter or lora_adapter.strip() in ("", "（なし）"):
        return ""

    lp = Path(lora_adapter.strip())
    if not lp.is_dir():
        return f"❌ フォルダが存在しません: {lp}"

    adapter_config_path = lp / "adapter_config.json"
    if not adapter_config_path.is_file():
        return "❌ adapter_config.json が見つかりません。"

    _ADAPTER_STATES = ("adapter_model.safetensors", "adapter_model.bin")
    if not any((lp / s).is_file() for s in _ADAPTER_STATES):
        return "❌ adapter_model.safetensors / adapter_model.bin が見つかりません。"

    # 本家版 vs フォーク版の識別
    try:
        adapter_cfg = _json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"❌ adapter_config.json の読み込みに失敗: {e}"

    has_metadata = (lp / "irodori_lora_metadata.json").is_file()
    target_modules = adapter_cfg.get("target_modules")
    is_upstream = has_metadata or (
        isinstance(target_modules, str) and target_modules.startswith("^")
    )
    origin_label = "本家版" if is_upstream else "フォーク版"

    # ロード済みモデルとの latent_dim 照合
    info = _detect_model_version_from_runtime()
    if info is None:
        return f"⚠️ {origin_label}アダプタ検出。モデル未読み込みのため latent_dim 照合をスキップ。"

    version_label, _, ldim = info
    adapter_st_path = lp / "adapter_model.safetensors"
    if adapter_st_path.is_file():
        try:
            from safetensors import safe_open as _safe_open
            from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE
            runtime = _RUNTIME_CACHE_VALUE
            expected_patched = ldim * int(runtime.model_cfg.latent_patch_size) # type: ignore

            with _safe_open(str(adapter_st_path), framework="pt", device="cpu") as _h:
                all_keys = list(_h.keys())
            in_proj_key = next(
                (k for k in all_keys if "in_proj" in k and "lora_A" in k), None
            )
            if in_proj_key is not None:
                with _safe_open(str(adapter_st_path), framework="pt", device="cpu") as _h:
                    t = _h.get_tensor(in_proj_key)
                adapter_in = int(t.shape[1])
                if adapter_in != expected_patched:
                    return (
                        f"❌ 互換性エラー ({origin_label}): "
                        f"アダプタ in_features={adapter_in} ≠ "
                        f"モデル patched_latent_dim={expected_patched}。"
                        f"このLoRAはモデル {version_label} と非互換です。"
                    )
        except Exception as e:
            return f"⚠️ {origin_label}アダプタ / shape 検証スキップ: {e}"

    return f"✅ {origin_label}アダプタ ({version_label} モデルと互換)"


def _runtime_uses_voice_design() -> bool:
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE

    runtime = _RUNTIME_CACHE_VALUE
    if runtime is None:
        return False
    return bool(getattr(runtime.model_cfg, "use_caption_condition", False))



# ─────────────────────────────
# UI 生成
# ─────────────────────────────

def build(ctx):
    with gr.Tab("🔊 推論"):
        gr.Markdown("## モデル設定")

        with gr.Row():
            infer_checkpoint = gr.Dropdown(
                label="チェックポイント (.pt / .safetensors)",
                choices=ctx.initial_checkpoints,
                value=ctx.default_checkpoint or None,
                scale=4, allow_custom_value=False,
            )
            infer_refresh_btn = gr.Button("更新", scale=1)

        with gr.Row():
            infer_lora_adapter = gr.Dropdown(
                label="LoRAアダプタ（なし=ベースモデルのみ）",
                choices=["（なし）"] + scan_lora_adapters(),
                value="（なし）",
                scale=4, allow_custom_value=False,
            )
            infer_lora_refresh_btn = gr.Button("更新", scale=1)
        infer_lora_compat_status = gr.Textbox(
            label="LoRA互換性チェック",
            value="",
            interactive=False,
            lines=1,
            visible=False,
        )
        infer_lora_scale = gr.Slider(
            label="LoRAスケール（0.0=LoRA無効 / 1.0=通常 / >1.0=強調）",
            minimum=0.0, maximum=2.0, value=1.0, step=0.05, visible=False,
        )
        infer_lora_disabled_modules = gr.Textbox(
            label="LoRA無効モジュール（カンマ区切り、空=全て有効）",
            value="",
            placeholder="例: blocks.0.attention, blocks.1.attention",
            visible=False,
            info="指定したモジュールのLoRAをスケール0で無効化します。",
        )
        with gr.Accordion("モデル詳細設定", open=False):
            with gr.Row():
                model_device = gr.Dropdown(label="モデルデバイス", choices=ctx.device_choices, value=ctx.default_model_device, scale=1)
                model_precision = gr.Dropdown(label="モデル精度", choices=ctx.model_precision_choices, value=ctx.model_precision_choices[0], scale=1)
                codec_device = gr.Dropdown(label="コーデックデバイス", choices=ctx.device_choices, value=ctx.default_codec_device, scale=1)
                codec_precision = gr.Dropdown(label="コーデック精度", choices=ctx.codec_precision_choices, value=ctx.codec_precision_choices[0], scale=1)
                enable_watermark = gr.Checkbox(label="ウォーターマーク", value=False, scale=1)

            infer_codec_repo = gr.Dropdown(
                label="コーデックリポジトリ（モデル読み込み時に自動設定）",
                choices=cnf.PREPARE_CODEC_REPO_CHOICES,
                value="Aratako/Semantic-DACVAE-Japanese-32dim",
                info="v2(dim32) / v1(dim128) — モデル読み込み後に自動切替されます。",
                interactive=True,
            )
            with gr.Row():
                load_model_btn  = gr.Button("モデル読み込み", variant="secondary")
                unload_model_btn= gr.Button("モデル解放",    variant="secondary")
            
            model_status = gr.Textbox(label="モデルステータス", interactive=False, lines=4)

        gr.Markdown("## 音声生成")

        with gr.Accordion("参照音声 | Voice Design", open=False):
            spk_ref_latent_path = gr.Textbox(visible=False, value="")

            with gr.Tabs():
                with gr.Tab("直接アップロード"):
                    infer_audio = gr.Audio(label="参照音声", type="filepath")
                    infer_audio.change(
                        lambda v: "",
                        inputs=[infer_audio],
                        outputs=[spk_ref_latent_path],
                    )

                with gr.Tab("スピーカーライブラリ"):
                    with gr.Row():
                        spk_select = gr.Dropdown(
                            label="キャラクター",
                            choices=_scan_speakers(),
                            value="（使用しない）",
                            scale=4,
                        )
                        spk_lib_refresh = gr.Button("更新", scale=1)
                    spk_info = gr.Textbox(
                        label="登録情報", interactive=False, lines=2
                    )

                    def _on_spk_select(name):
                        if not name or name == "（使用しない）":
                            return "", ""
                        pt = cnf.SPEAKERS_DIR / name / "ref.pt"
                        profile_p = cnf.SPEAKERS_DIR / name / "profile.json"
                        pt_str = str(pt) if pt.exists() else ""
                        info = ""
                        if profile_p.exists():
                            try:
                                p = json.loads(profile_p.read_text(encoding="utf-8"))
                                info = (
                                    f"duration: {p.get('duration_sec', '?')}秒  "
                                    f"latent: {p.get('latent_shape', '?')}"
                                )
                            except Exception:
                                pass
                        return pt_str, info

                    spk_select.change(
                        _on_spk_select,
                        inputs=[spk_select],
                        outputs=[spk_ref_latent_path, spk_info],
                    )
                    spk_lib_refresh.click(
                        lambda: gr.Dropdown(choices=_scan_speakers()),
                        outputs=[spk_select],
                    )

                with gr.Tab("スピーカー登録"):
                    gr.Markdown(
                        "参照WAVをDACVAEエンコードして `speakers/{名前}/` に\n"
                        "`ref.wav` / `ref.pt` / `profile.json` の3ファイルを生成します。\n\n"
                        "> **事前条件**: 上部の「モデル読み込み」を完了してから実行してください。"
                    )
                    spk_reg_name = gr.Textbox(
                        label="キャラクター名",
                        placeholder="alice",
                    )
                    spk_reg_wav = gr.Audio(
                        label="参照WAV（5〜30秒推奨、雑音なし）",
                        type="filepath",
                    )
                    spk_reg_btn = gr.Button(
                        "登録", variant="primary"
                    )
                    spk_reg_status = gr.Textbox(
                        label="結果", interactive=False, lines=4
                    )

                    spk_reg_btn.click(
                        _run_create_speaker,
                        inputs=[
                            spk_reg_name, spk_reg_wav,
                            infer_checkpoint,
                            model_device, model_precision,
                            codec_device, codec_precision,
                        ],
                        outputs=[spk_reg_status],
                    )
                    spk_reg_btn.click(
                        lambda: gr.Dropdown(choices=_scan_speakers()),
                        outputs=[spk_select],
                    )
                with gr.Tab("Voice Design Caption"):
                    caption_input_vd = gr.Textbox(
                        label="Caption (Voice Design)",
                        value="",
                        lines=3,
                        placeholder="e.g. calm, bright, energetic, whispering",
                        info="VoiceDesign 対応モデル読み込み時のみ有効",
                    )
                    with gr.Row():
                        cfg_scale_caption = gr.Slider(
                            label="Caption CFG Scale",
                            minimum=0.0, maximum=10.0, value=3.0, step=0.1,
                        )
                        max_caption_len_raw = gr.Textbox(
                            label="Max Caption Len (optional)",
                            value="",
                        )

        with gr.Accordion("感情スタイル", open=False):
            gr.Markdown(
                "プリセットボタンを押すと、下の各パラメータが自動設定されます。"
                "その後スライダーを手動調整することも可能です。"
            )
            with gr.Row():
                preset_normal  = gr.Button("ノーマル",   variant="secondary", scale=1)
                preset_strong  = gr.Button("力強く",     variant="secondary", scale=1)
                preset_calm    = gr.Button("おとなしく", variant="secondary", scale=1)
                preset_bright  = gr.Button("明るく",     variant="secondary", scale=1)
                preset_whisper = gr.Button("ひそやかに", variant="secondary", scale=1)

            gr.Markdown("##### スタイル調整パラメータ")
            with gr.Row():
                style_cfg_text = gr.Slider(
                    label="テキスト表現力（低：棒読み ↔ 高：抑揚強調）",
                    minimum=0.0, maximum=10.0, value=3.0, step=0.1, scale=2,
                )
                style_cfg_speaker = gr.Slider(
                    label="感情の強さ（低：ニュートラル ↔ 高：スタイル強調）",
                    minimum=0.0, maximum=10.0, value=5.0, step=0.1, scale=2,
                )
            with gr.Row():
                style_kv_scale = gr.Slider(
                    label="話者密着度（1.0=標準、高いほど参照音声の特徴を強く反映）",
                    minimum=1.0, maximum=4.0, value=1.0, step=0.1, scale=2,
                )
                style_trunc = gr.Slider(
                    label="表現の振れ幅（低：安定・平坦 ↔ 高：ダイナミック・不安定）",
                    minimum=0.7, maximum=1.0, value=1.0, step=0.01, scale=2,
                )

        with gr.Accordion("サンプリング設定", open=True):
            with gr.Row():
                num_steps = gr.Slider(
                    label="ステップ数（多いほど品質向上・低速）",
                    minimum=1, maximum=120, value=40, step=1,
                )
                seed_raw = gr.Textbox(label="シード（空白=ランダム）", value="")

        with gr.Accordion("CFG設定", open=False):
            gr.Markdown(
                "**CFG（Classifier-Free Guidance）** はモデルが条件（テキスト・話者）をどれだけ"
                "強く守るかを制御します。値が大きいほど条件に忠実になりますが、高すぎると"
                "不自然になる場合があります。"
            )
            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="ガイダンスモード",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                    info="independent=高品質・低速（推奨） / joint=バランス / alternating=高速",
                )
                cfg_scale_text = gr.Slider(
                    label="テキストCFG強度",
                    minimum=0.0, maximum=10.0, value=3.0, step=0.1,
                    info="テキスト内容への忠実度。感情スタイルと連動します。",
                )
                cfg_scale_speaker = gr.Slider(
                    label="話者CFG強度",
                    minimum=0.0, maximum=10.0, value=5.0, step=0.1,
                    info="参照音声の声質への忠実度。感情スタイルと連動します。",
                )
        with gr.Accordion("詳細設定（上級者向け）", open=False):
            gr.Markdown(
                "通常は変更不要です。動作確認・実験用途向けの項目です。"
            )
            cfg_scale_raw = gr.Textbox(
                label="CFGスケール一括上書き（テキスト・話者を同値に設定。空=無効）",
                value="",
            )
            with gr.Row():
                cfg_min_t = gr.Number(
                    label="CFG適用開始タイムステップ",
                    value=0.5,
                    info="拡散過程のどの時点からCFGを適用するか（0.0〜1.0）",
                )
                cfg_max_t = gr.Number(
                    label="CFG適用終了タイムステップ",
                    value=1.0,
                    info="拡散過程のどの時点までCFGを適用するか（0.0〜1.0）",
                )
                context_kv_cache = gr.Checkbox(
                    label="コンテキストKVキャッシュ（推論高速化）",
                    value=True,
                    info="テキスト・話者のKV射影を事前計算してステップ間で再利用します",
                )
            with gr.Row():
                rescale_k_raw = gr.Textbox(
                    label="スコア再スケールk（空=無効）",
                    value="",
                    info="Xu et al. 2025 の時間的スコアリスケール係数k",
                )
                rescale_sigma_raw = gr.Textbox(
                    label="スコア再スケールsigma（空=無効）",
                    value="",
                    info="rescale_k と合わせて設定します",
                )
            with gr.Row():
                speaker_kv_min_t_raw = gr.Textbox(
                    label="話者KVスケール適用閾値（デフォルト0.9）",
                    value="0.9",
                    info="この値以上のタイムステップでのみ話者KV強調を適用します",
                )
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="話者KVスケール適用レイヤー数上限（空=全レイヤー）",
                    value="",
                    info="拡散ブロックの先頭N層にのみ話者KV強調を適用します",
                )

            truncation_factor_raw = gr.Textbox(visible=False, value="")
            speaker_kv_scale_raw  = gr.Textbox(visible=False, value="")

        with gr.Row():
            num_candidates = gr.Slider(
                label="生成候補数 (Num Candidates)",
                minimum=1, maximum=8, value=1, step=1,
                scale=4,
                info="1回の生成で作成する数。シード指定時は seed, seed+1, seed+2...。改行分割モード時は1固定。",
            )
            filename_prefix = gr.Textbox(
                label="File name prefix",
                value=ctx.args.output_prefix,
                scale=4,
                info="If left blank, use the model name.",
            )

        infer_text = gr.Textbox(label="テキスト（合成したい文章）", lines=4)
        
        generate_btn = gr.Button("生成", variant="primary", size="lg")

        _MAX_CANDIDATES = 8
        gr.Markdown("### 生成結果")
        gr.Markdown(
            "ファイルは `" + str(cnf.OUTPUTS_DIR) + "` フォルダに保存されています。"
        )

        _cand_labels = []
        _cand_audios = []
        for _ci in range(_MAX_CANDIDATES):
            with gr.Row(visible=False) as _row:
                pass
            _lbl = gr.Textbox(
                value="",
                label=f"候補 {_ci + 1}",
                interactive=False,
                visible=False,
                max_lines=1,
                show_label=True,
                scale=1,
            )
            _aud = gr.Audio(
                value=None,
                label=f"候補 {_ci + 1} の音声",
                type="filepath",
                interactive=False,
                visible=False,
                scale=3,
            )
            _cand_labels.append(_lbl)
            _cand_audios.append(_aud)

        out_log    = gr.Textbox(label="実行ログ", lines=6)
        out_timing = gr.Textbox(label="タイミング情報", lines=6)

        def _run_generation_ui(*args):
            gallery_items, detail_text, timing_text = _run_generation(*args)
            label_updates = []
            audio_updates = []
            for i in range(_MAX_CANDIDATES):
                if i < len(gallery_items):
                    path, caption = gallery_items[i]
                    label_updates.append(gr.update(value=caption, visible=True))
                    audio_updates.append(gr.update(value=path, visible=True))
                else:
                    label_updates.append(gr.update(value="", visible=False))
                    audio_updates.append(gr.update(value=None, visible=False))
            return label_updates + audio_updates + [detail_text, timing_text]

        _PRESETS = {
            "normal":  (3.0, 5.0, 1.0, 1.0),
            "strong":  (5.0, 7.0, 1.8, 1.0),
            "calm":    (2.0, 3.0, 1.0, 0.80),
            "bright":  (4.5, 6.0, 1.5, 0.95),
            "whisper": (2.0, 2.0, 1.0, 0.75),
        }

        def _apply_preset(name):
            ct, cs, kv, tr = _PRESETS[name]
            kv_str = "" if kv == 1.0 else str(kv)
            tr_str = "" if tr == 1.0 else str(tr)
            return ct, cs, ct, cs, kv, tr, kv_str, tr_str

        _preset_outputs = [
            style_cfg_text, style_cfg_speaker,
            cfg_scale_text, cfg_scale_speaker,
            style_kv_scale, style_trunc,
            speaker_kv_scale_raw, truncation_factor_raw,
        ]

        preset_normal.click(
            lambda: _apply_preset("normal"), outputs=_preset_outputs,
        )
        preset_strong.click(
            lambda: _apply_preset("strong"), outputs=_preset_outputs,
        )
        preset_calm.click(
            lambda: _apply_preset("calm"), outputs=_preset_outputs,
        )
        preset_bright.click(
            lambda: _apply_preset("bright"), outputs=_preset_outputs,
        )
        preset_whisper.click(
            lambda: _apply_preset("whisper"), outputs=_preset_outputs,
        )

        def _sync_style_to_cfg(ct, cs, kv, tr):
            kv_str = "" if kv <= 1.0 else str(round(kv, 2))
            tr_str = "" if tr >= 1.0 else str(round(tr, 2))
            return ct, cs, kv_str, tr_str

        style_cfg_text.change(
            lambda v, cs, kv, tr: _sync_style_to_cfg(v, cs, kv, tr),
            inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
            outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
        )
        style_cfg_speaker.change(
            lambda ct, v, kv, tr: _sync_style_to_cfg(ct, v, kv, tr),
            inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
            outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
        )
        style_kv_scale.change(
            lambda ct, cs, v, tr: _sync_style_to_cfg(ct, cs, v, tr),
            inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
            outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
        )
        style_trunc.change(
            lambda ct, cs, kv, v: _sync_style_to_cfg(ct, cs, kv, v),
            inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
            outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
        )

        cfg_scale_text.change(
            lambda v: v, inputs=[cfg_scale_text], outputs=[style_cfg_text],
        )
        cfg_scale_speaker.change(
            lambda v: v, inputs=[cfg_scale_speaker], outputs=[style_cfg_speaker],
        )

        infer_refresh_btn.click(
            lambda: gr.Dropdown(choices=scan_checkpoints(), value=(scan_checkpoints() or [None])[-1]),
            outputs=[infer_checkpoint],
        )
        infer_lora_refresh_btn.click(
            lambda: gr.Dropdown(choices=["（なし）"] + scan_lora_adapters()),
            outputs=[infer_lora_adapter],
        )

        def _on_lora_adapter_change(v):
            is_active = str(v).strip() not in ("", "（なし）")
            compat_msg = _validate_lora_compat_ui(v) if is_active else ""
            return (
                gr.Slider(visible=is_active),
                gr.Textbox(visible=is_active),
                gr.Textbox(value=compat_msg, visible=is_active),
            )

        infer_lora_adapter.change(
            _on_lora_adapter_change,
            inputs=[infer_lora_adapter],
            outputs=[infer_lora_scale, infer_lora_disabled_modules, infer_lora_compat_status],
        )

        model_device.change(_on_model_device_change, inputs=[model_device], outputs=[model_precision])
        codec_device.change(_on_codec_device_change, inputs=[codec_device], outputs=[codec_precision])

        def _load_model_ui(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter, cur_lora_adapter):
            status_text, auto_codec, voice_design_enabled = _load_model(
                checkpoint, model_device, model_precision,
                codec_device, codec_precision, enable_watermark, lora_adapter,
            )
            compat_msg = _validate_lora_compat_ui(cur_lora_adapter) if (
                str(cur_lora_adapter).strip() not in ("", "（なし）")
            ) else ""
            return (
                status_text,
                gr.Dropdown(value=auto_codec),
                gr.Textbox(value=compat_msg),
            )

        load_model_btn.click(
            _load_model_ui,
            inputs=[infer_checkpoint, model_device, model_precision,
                    codec_device, codec_precision, enable_watermark,
                    infer_lora_adapter, infer_lora_adapter],
            outputs=[model_status, infer_codec_repo, infer_lora_compat_status],
        )
        unload_model_btn.click(_clear_runtime_cache_ui, outputs=[model_status])
        _ui_outputs = _cand_labels + _cand_audios + [out_log, out_timing]
        generate_btn.click(_run_generation_ui,
            inputs=[
                infer_checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
                infer_lora_adapter, infer_lora_scale, infer_lora_disabled_modules,
                infer_text, caption_input_vd, infer_audio, spk_ref_latent_path,
                num_steps, seed_raw, cfg_guidance_mode,
                cfg_scale_text, cfg_scale_speaker, cfg_scale_caption, cfg_scale_raw, cfg_min_t, cfg_max_t,
                context_kv_cache, max_caption_len_raw, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
                speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
                num_candidates,
                filename_prefix,
            ],
            outputs=_ui_outputs,
        )
