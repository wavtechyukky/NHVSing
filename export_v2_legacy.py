import argparse
import os
import yaml
import torch
import torch.nn as nn
import onnx

from model import NHVSingV2
from onnx_model import NHVConvsShared
from layers import F0Embedder
from dsp_rebuild.impulse_train_onnx import GenerateImpulseTrainONNX
from dsp_rebuild.complex_cepstrum_to_imp_onnx import ComplexCepstrumToImpONNX
from dsp_rebuild.ltv_fir_onnx import LTVFirONNX


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config):
    model = NHVSingV2(
        vocoder_cfg=config['model']['vocoder'],
        ltv_filter_cfg=config['model']['ltv_filter'],
    )
    snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(snapshot['model'] if 'model' in snapshot else snapshot)
    model.eval()
    return model


def export_pytorch_model(model, save_path):
    print(f"Exporting PyTorch state_dict to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Done.")


def export_jit_model(model, save_path):
    print(f"Exporting JIT model to {save_path}...")
    scripted = torch.jit.script(model)
    scripted.save(save_path)
    print("Done.")


# ---------------------------------------------------------------------------
# Full unified ONNX model (NHVSingV2 — always shared trunk + F0 embedder)
# ---------------------------------------------------------------------------

class FullVocoderONNX(nn.Module):
    """Unified ONNX-exportable vocoder for NHVSingV2.

    Inputs:
      log_melspc : (B, T, 128)         — log mel-spectrogram
      f0         : (B, 1, T)           — continuous F0 in Hz (interpolated)
      z          : (B, 1, T*hop_size)  — noise source

    Output:
      waveform   : (B, 1, T*hop_size)  — synthesized waveform in [-1, 1]
    """

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        self.hop_size = vocoder_cfg['hop_size']

        self.nn_core = NHVConvsShared(dict(ltv_filter_cfg))
        self.f0_embedder = F0Embedder(
            n_bins    = ltv_filter_cfg.get('f0_embed_bins', 256),
            embed_dim = ltv_filter_cfg.get('f0_embed_dim',  128),
            f0_min    = ltv_filter_cfg.get('f0_embed_fmin', 40.0),
            f0_max    = ltv_filter_cfg.get('f0_embed_fmax', 1200.0),
        )
        self.impulse_train = GenerateImpulseTrainONNX(200, vocoder_cfg['sample_rate'])
        self.ccep_to_imp   = ComplexCepstrumToImpONNX(ltv_filter_cfg['fft_size'], use_float64=True)
        self.ltv_fir       = LTVFirONNX(self.hop_size, filter_size=ltv_filter_cfg['fft_size'])

    def forward(self, log_melspc: torch.Tensor, f0: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        f0_embed = self.f0_embedder(f0)                      # (B, T, embed_dim)
        x = torch.cat([log_melspc, f0_embed], dim=-1)        # (B, T, mel+embed_dim)
        ccep_harm, ccep_noise = self.nn_core(x)

        cf0_resampled   = torch.repeat_interleave(f0, self.hop_size, dim=-1)
        harmonic_source = self.impulse_train(cf0_resampled)

        sig_harm  = self.ltv_fir(harmonic_source, self.ccep_to_imp(ccep_harm))
        sig_noise = self.ltv_fir(z,               self.ccep_to_imp(ccep_noise))
        return torch.clamp(sig_harm + sig_noise, -1.0, 1.0)


def export_full_onnx(model, config, save_path, opset=18):
    print("Creating FullVocoderONNX model...")
    full_model = FullVocoderONNX(
        vocoder_cfg=config['model']['vocoder'],
        ltv_filter_cfg=config['model']['ltv_filter'],
    )

    # Copy weights: NHVSingV2 uses self.convs (NHVConvsShared) and self.f0_embedder
    src_sd = model.state_dict()

    nn_core_sd = {k.replace('convs_onnx.', '', 1): v
                  for k, v in src_sd.items() if k.startswith('convs_onnx.')}
    full_model.nn_core.load_state_dict(nn_core_sd)
    print(f"  Copied {len(nn_core_sd)} tensors into nn_core.")

    f0_embed_sd = {k.replace('f0_embedder.', '', 1): v
                   for k, v in src_sd.items() if k.startswith('f0_embedder.')}
    full_model.f0_embedder.load_state_dict(f0_embed_sd)
    print(f"  Copied {len(f0_embed_sd)} tensors into f0_embedder.")

    full_model.eval()

    n_frames  = 100
    mel_dim   = config['preprocess']['mel_dim']
    hop_size  = config['model']['vocoder']['hop_size']
    n_samples = n_frames * hop_size

    dummy_mel = torch.randn(1, n_frames, mel_dim, dtype=torch.float32)
    dummy_f0  = torch.randn(1, 1, n_frames, dtype=torch.float32).abs() * 300 + 100
    dummy_z   = torch.randn(1, 1, n_samples, dtype=torch.float32) * 0.03

    print("Running test forward pass...")
    with torch.no_grad():
        y = full_model(dummy_mel, dummy_f0, dummy_z)
    print(f"  Output shape: {y.shape}, range: [{y.min():.4f}, {y.max():.4f}]")

    print(f"\nExporting to {save_path} (opset {opset})...")
    torch.onnx.export(
        full_model,
        (dummy_mel, dummy_f0, dummy_z),
        save_path,
        input_names=['log_melspc', 'f0', 'z'],
        output_names=['waveform'],
        dynamic_axes={
            'log_melspc': {0: 'batch', 1: 'n_frames'},
            'f0':         {0: 'batch', 2: 'n_frames'},
            'z':          {0: 'batch', 2: 'n_samples'},
            'waveform':   {0: 'batch', 2: 'n_samples'},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )

    print("Merging into single .onnx file...")
    model_proto = onnx.load(save_path)
    onnx.save_model(model_proto, save_path, save_as_external_data=False)
    data_file = save_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"  Exported: {save_path} ({os.path.getsize(save_path)/1024/1024:.1f} MB)")
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export NHVSingV2 for inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="config_v2.yaml")
    parser.add_argument("--output_dir", type=str, default="exported_models/v2")
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--pytorch",    action="store_true", help="Export state_dict.")
    parser.add_argument("--jit",        action="store_true", help="Export JIT-scripted model.")
    parser.add_argument("--full_onnx",  action="store_true", help="Export unified full ONNX.")
    parser.add_argument("--opset",      type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)

    print("Loading NHVSingV2 from checkpoint...")
    model = load_model(args.checkpoint, config)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.all or args.pytorch:
        export_pytorch_model(model, os.path.join(args.output_dir, "model.pth"))

    if args.all or args.jit:
        export_jit_model(model, os.path.join(args.output_dir, "model_jit.pt"))

    if args.all or args.full_onnx:
        export_full_onnx(model, config,
                         save_path=os.path.join(args.output_dir, "full_vocoder.onnx"),
                         opset=args.opset)


if __name__ == "__main__":
    main()
