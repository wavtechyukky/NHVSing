"""
prepare_finetune.py
-------------------
Create a fine-tuning snapshot from pretrained model weights.

The output snapshot can be passed directly to train.py via --resume_path.
Discriminator and optimizers are freshly initialized so training resumes
cleanly from epoch 0 without inheriting old momentum or adversarial state.

Usage:
    python prepare_finetune.py \
        --weights exported_models/model.pth \
        --config  config.yaml \
        --output  finetune_init.pth

    python train.py --resume_path finetune_init.pth

Supported input formats for --weights:
    - Full training snapshot (keys: model, discriminator, optimizer_g, ...)
    - Exported state_dict (model.pth produced by export.py --pytorch)
"""

import argparse
from pathlib import Path

import torch
import yaml

from discriminator import DiscriminatorWithComplexSTFT
from model import NHVSing, NHVSingV2


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning snapshot")
    parser.add_argument("--weights", required=True,
                        help="Source model weights (.pth): snapshot or exported state_dict")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml for the TARGET training (default: config.yaml)")
    parser.add_argument("--output", default="finetune_init.pth",
                        help="Output snapshot path (default: finetune_init.pth)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Build model from target config ---
    ltv_filter_cfg = cfg['model']['ltv_filter']
    ModelClass = NHVSingV2 if ltv_filter_cfg.get('use_shared_trunk', False) else NHVSing
    print(f"Model: {ModelClass.__name__}")
    model = ModelClass(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=ltv_filter_cfg,
    )

    # --- Load source weights (snapshot or raw state_dict) ---
    src = torch.load(args.weights, map_location='cpu')
    weights = src['model'] if isinstance(src, dict) and 'model' in src else src
    model.load_state_dict(weights)
    print(f"Loaded model weights from: {args.weights}")

    # --- Fresh discriminator and optimizers ---
    disc_cfg = cfg.get('discriminator', {})
    discriminator = DiscriminatorWithComplexSTFT(
        use_msd=disc_cfg.get('use_msd', True),
        stft_filters=disc_cfg.get('stft_filters', 32),
    )
    optimizer_g = torch.optim.RAdam(model.parameters(),
                                    lr=cfg['training']['lr_g'], eps=1e-4)
    optimizer_d = torch.optim.RAdam(discriminator.parameters(),
                                    lr=cfg['training']['lr_d'])

    # --- Save snapshot in train.py-compatible format ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":         model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g":   optimizer_g.state_dict(),
        "optimizer_d":   optimizer_d.state_dict(),
        "epoch":         0,
    }, output_path)
    print(f"Saved fine-tune snapshot: {output_path}")
    print("Run training with:")
    print(f"  python train.py --resume_path {output_path}")
    print(f"  # --config defaults to config.yaml; specify explicitly if using a different config")


if __name__ == "__main__":
    main()
