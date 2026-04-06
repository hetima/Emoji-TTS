from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# パス定数
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
CONFIGS_DIR      = BASE_DIR / "configs"
LOGS_DIR         = BASE_DIR / "logs"
OUTPUTS_DIR      = BASE_DIR / "gradio_outputs"
LORA_DIR         = BASE_DIR / "lora"
DEFAULT_HF_REPO  = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CONFIG   = "train_v2.yaml"
DEFAULT_PREPARE_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
PREPARE_CODEC_REPO_CHOICES = [
    "Aratako/Semantic-DACVAE-Japanese-32dim",  # v2 (dim32)
    "facebook/dacvae-watermarked",             # v1 (dim128)
]
FIXED_SECONDS    = 30.0
DATASET_TOOLS    = BASE_DIR / "dataset_tools.py"
DEFAULT_DATASET_DIR = BASE_DIR / "my_dataset"
SPEAKERS_DIR        = BASE_DIR / "speakers"

