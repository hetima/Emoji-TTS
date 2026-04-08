from __future__ import annotations


import warnings
from pathlib import Path
import gradio as gr
import yaml

from .setting import cnfg

# torch.nn.utils.weight_norm deprecation warning (from upstream deps) is noisy
# but currently harmless for inference.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`\.",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)

try:
    import pandas as pandas

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pandas = None  # type: ignore

from merge import (
    run_merge,
    scan_checkpoints_for_merge,
    get_default_base_path,
    LAYER_GROUPS,
)

from lora_merge import (
    run_lora_merge,
    run_lora_lora_merge,
    scan_lora_adapters_for_merge,
    peek_adapter_version,
)

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)


# ─────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────


def default_model_device() -> str:
    return default_runtime_device()


def precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def scan_checkpoints() -> list[str]:
    cnfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [
            *cnfg.checkpoints_dir.glob("**/*.pt"),
            *cnfg.checkpoints_dir.glob("**/*.safetensors"),
        ]
    )
    result = []
    for p in candidates:
        parts = p.relative_to(cnfg.checkpoints_dir).parts
        if parts[0] in {"codecs", "tokenizers"}:
            continue
        result.append(str(p))
    return result


def scan_configs() -> list[str]:
    cnfg.configs_dir.mkdir(parents=True, exist_ok=True)
    return sorted(str(p) for p in cnfg.configs_dir.glob("*.yaml")) + sorted(
        str(p) for p in cnfg.configs_dir.glob("*.yml")
    )


def scan_manifests() -> list[str]:
    return sorted(str(p) for p in cnfg.data_root_dir.glob("**/*.jsonl"))


def scan_train_checkpoints() -> list[str]:
    result = []
    for p in cnfg.base_dir.glob("**/*.pt"):
        if p.stat().st_size > 1024 * 1024:
            result.append(str(p))
    return sorted(result)


def scan_lora_adapters() -> list[str]:
    """adapter_config.json と adapter_model.safetensors/.bin の両方が存在するフォルダを列挙。"""
    cnfg.lora_dir.mkdir(parents=True, exist_ok=True)
    _ADAPTER_STATES = ("adapter_model.safetensors", "adapter_model.bin")
    result = []
    for p in sorted(cnfg.lora_dir.rglob("adapter_config.json")):
        parent = p.parent
        if any((parent / s).is_file() for s in _ADAPTER_STATES):
            result.append(str(parent))
    return result


def scan_lora_full_adapters() -> list[str]:
    cnfg.lora_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(cnfg.lora_dir.rglob("adapter_config.json")):
        if p.parent.name.endswith("_full"):
            result.append(str(p.parent))
    return result


def load_yaml_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}  # type: ignore


def ensure_default_model() -> None:
    if scan_checkpoints():
        return
    print(f"[gradio] モデル未検出。{cnfg.default_hf_repo} を自動ダウンロードします...", flush=True)
    try:
        from huggingface_hub import hf_hub_download

        safe_name = cnfg.default_hf_repo.replace("/", "_")
        local_dir = cnfg.checkpoints_dir / safe_name
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=cnfg.default_hf_repo,
            filename="model.safetensors",
            local_dir=str(local_dir),
        )
        print(f"[gradio] 自動ダウンロード完了: {downloaded}", flush=True)
    except Exception as e:
        print(f"[gradio] 自動ダウンロード失敗: {e}", flush=True)


def merge_scan() -> list[str]:
    return scan_checkpoints_for_merge()

def make_unique_file_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 99):
        new_path = path.parent / f"{path.stem}_{i:02}{path.suffix}"
        if not new_path.exists():
            return new_path
    for i in range(100, 999):
        new_path = path.parent / f"{path.stem}_{i:03}{path.suffix}"
        if not new_path.exists():
            return new_path
    return path
