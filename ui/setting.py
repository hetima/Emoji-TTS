import dataclasses
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CONFIGS_DIR = BASE_DIR / "configs"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "gradio_outputs"
OUTPU_PREFIX = ""
LORA_DIR = BASE_DIR / "lora"
DEFAULT_HF_REPO = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CONFIG = "train_v2.yaml"
DEFAULT_PREPARE_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
PREPARE_CODEC_REPO_CHOICES = [
    "Aratako/Semantic-DACVAE-Japanese-32dim",  # v2 (dim32)
    "facebook/dacvae-watermarked",  # v1 (dim128)
]
FIXED_SECONDS = 30.0
DATASET_TOOLS = REPO_DIR / "dataset_tools.py"
DEFAULT_DATASET_DIR = BASE_DIR / "my_dataset"
SPEAKERS_DIR = BASE_DIR / "speakers"


@dataclasses.dataclass
class Setting:
    _SAVABLE_SETTINGS: tuple[str, ...] = dataclasses.field(
        default=(
            "outputs_dir",
            "output_prefix",
            "speakers_dir",
            "checkpoints_dir",
            "lora_dir",
            "default_dataset_dir",
            "fixed_seconds",
        ),
        init=False,
        repr=False,
    )

    repo_dir: Path = REPO_DIR
    base_dir: Path = BASE_DIR
    outputs_dir: Path = OUTPUTS_DIR
    output_prefix: str = OUTPU_PREFIX
    setting_path: Path = REPO_DIR / "app_setting.json"
    speakers_dir: Path = SPEAKERS_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    configs_dir: Path = CONFIGS_DIR
    logs_dir: Path = LOGS_DIR
    lora_dir: Path = LORA_DIR
    default_hf_repo: str = DEFAULT_HF_REPO
    default_config: str = DEFAULT_CONFIG
    default_dataset_dir: Path = DEFAULT_DATASET_DIR
    dataset_tools: Path = DATASET_TOOLS
    fixed_seconds: float = FIXED_SECONDS
    default_prepare_codec_repo: str = DEFAULT_PREPARE_CODEC_REPO
    prepare_codec_repo_choices: list[str] = dataclasses.field(
        default_factory=lambda: list(PREPARE_CODEC_REPO_CHOICES)
    )

    def __post_init__(self):
        self.load()

    def save(self):
        def _serialize(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [_serialize(v) for v in value]
            return value

        data = {name: _serialize(getattr(self, name)) for name in self._SAVABLE_SETTINGS}
        self.setting_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load(self):
        if not self.setting_path.exists():
            return
        data = json.loads(self.setting_path.read_text(encoding="utf-8"))
        for name in self._SAVABLE_SETTINGS:
            if name not in data:
                continue
            value = data[name]
            field_type = None
            for f in dataclasses.fields(self):
                if f.name == name:
                    field_type = f.type
                    break
            if field_type is Path:
                value = Path(value)
            setattr(self, name, value)


cnfg = Setting()
