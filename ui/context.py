from __future__ import annotations

import argparse
import dataclasses


@dataclasses.dataclass
class UIContext:
    args: argparse.Namespace
    initial_checkpoints: list[str]
    default_checkpoint: str
    default_model_device: str
    default_codec_device: str
    device_choices: list[str]
    model_precision_choices: list[str]
    codec_precision_choices: list[str]
    initial_configs: list[str]
    default_config: str
    initial_manifests: list[str]
    initial_train_ckpts: list[str]
    default_cfg: dict

    def v(self, key, fallback=None):
        return self.default_cfg.get(key, fallback)
