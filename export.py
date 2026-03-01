import argparse
import os
import yaml
import torch
import torch.nn as nn
import onnx

from model import NHVSing
from onnx_model import NHVConvsONNX
from dsp_rebuild.impulse_train_onnx import GenerateImpulseTrainONNX
from dsp_rebuild.complex_cepstrum_to_imp_onnx import ComplexCepstrumToImpONNX
from dsp_rebuild.ltv_fir_onnx import LTVFirONNX


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config):
    """Load NHVSing from a training snapshot."""
    model = NHVSing(
        vocoder_cfg=config['model']['vocoder'],
        ltv_filter_cfg=config['model']['ltv_filter'],
    )
    snapshot = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot['model'])
    model.eval()
    return model


def export_pytorch_model(model, save_path):
    """Export model state_dict."""
    print(f"Exporting PyTorch model state_dict to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Done.")


def export_jit_model(model, save_path):
    """Export JIT-scripted model."""
    print(f"Exporting JIT model to {save_path}...")
    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path)
    print("Done.")


def export_onnx_core(model, save_path, dummy_x):
    """Export the conv-only core (NHVConvsONNX) to ONNX.
    Useful for partial ONNX pipelines or debugging.
    """
    print(f"Exporting ONNX conv core to {save_path}...")
    core_model = model.convs_onnx
    dynamic_axes = {
        'log_melspc': {0: 'batch_size', 1: 'time'},
        'ccep_harm':  {0: 'batch_size', 1: 'time'},
        'ccep_noise': {0: 'batch_size', 1: 'time'},
    }
    torch.onnx.export(
        core_model, dummy_x, save_path,
        opset_version=11,
        input_names=['log_melspc'],
        output_names=['ccep_harm', 'ccep_noise'],
        dynamic_axes=dynamic_axes,
    )
    print("Done.")


# ---------------------------------------------------------------------------
# Full unified ONNX model
# ---------------------------------------------------------------------------

class FullVocoderONNX(nn.Module):
    """Unified ONNX-exportable vocoder.

    Combines:
      - NHVConvsONNX       : mel → complex cepstrum (harmonic + noise)
      - GenerateImpulseTrainONNX : F0 → impulse train
      - ComplexCepstrumToImpONNX : complex cepstrum → impulse response
      - LTVFirONNX         : LTV-FIR filtering

    Inputs:
      log_melspc : (B, T, D)          — log mel-spectrogram
      f0         : (B, 1, T)          — continuous F0 (interpolated, no zeros)
      z          : (B, 1, T*hop_size) — noise source

    Output:
      waveform   : (B, 1, T*hop_size)
    """

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()

        self.hop_size   = vocoder_cfg['hop_size']
        sample_rate     = vocoder_cfg['sample_rate']
        fft_size        = ltv_filter_cfg['fft_size']

        ltv_params = {
            **ltv_filter_cfg,
            "in_channels":   vocoder_cfg['in_channels'],
            "conv_channels": vocoder_cfg['conv_channels'],
            "kernel_size":   vocoder_cfg['kernel_size'],
            "dilation_size": vocoder_cfg['dilation_size'],
            "group_size":    vocoder_cfg['group_size'],
            "n_ltv_layers":  ltv_filter_cfg.get("n_ltv_layers", 10),
            "use_causal":    vocoder_cfg['use_causal'],
            "conv_type":     vocoder_cfg['conv_type'],
            "hop_size":      vocoder_cfg['hop_size'],
        }
        self.nn_core       = NHVConvsONNX(ltv_params)
        self.impulse_train = GenerateImpulseTrainONNX(200, sample_rate)
        self.ccep_to_imp   = ComplexCepstrumToImpONNX(fft_size, use_float64=True)
        self.ltv_fir       = LTVFirONNX(self.hop_size, filter_size=fft_size)

    def forward(
        self,
        log_melspc: torch.Tensor,
        f0:         torch.Tensor,
        z:          torch.Tensor,
    ) -> torch.Tensor:
        # mel → complex cepstrum
        ccep_harm, ccep_noise = self.nn_core(log_melspc)

        # harmonic source from F0
        cf0_resampled  = torch.repeat_interleave(f0, self.hop_size, dim=-1)
        harmonic_source = self.impulse_train(cf0_resampled)

        # harmonic filtering
        imp_harm  = self.ccep_to_imp(ccep_harm)
        sig_harm  = self.ltv_fir(harmonic_source, imp_harm)

        # noise filtering
        imp_noise = self.ccep_to_imp(ccep_noise)
        sig_noise = self.ltv_fir(z, imp_noise)

        return torch.clamp(sig_harm + sig_noise, -1.0, 1.0)


def export_full_onnx(model, config, save_path, opset=18):
    """Export the complete vocoder as a single unified ONNX file.

    Args:
        model      : NHVSing (training snapshot already loaded)
        config     : full config dict
        save_path  : output .onnx file path
        opset      : ONNX opset version (default 18)
    """
    print("Creating FullVocoderONNX model...")
    full_model = FullVocoderONNX(
        vocoder_cfg=config['model']['vocoder'],
        ltv_filter_cfg=config['model']['ltv_filter'],
    )

    # Copy conv weights: convs_onnx.* → nn_core.*
    src_sd = model.state_dict()
    nn_core_sd = {
        k.replace('convs_onnx.', '', 1): v
        for k, v in src_sd.items()
        if k.startswith('convs_onnx.')
    }
    full_model.nn_core.load_state_dict(nn_core_sd)
    full_model.eval()
    print(f"  Copied {len(nn_core_sd)} weight tensors into nn_core.")

    # Dummy inputs
    n_frames  = 100
    mel_dim   = config['preprocess']['mel_dim']
    hop_size  = config['model']['vocoder']['hop_size']
    n_samples = n_frames * hop_size

    dummy_mel = torch.randn(1, n_frames, mel_dim, dtype=torch.float32)
    dummy_f0  = torch.randn(1, 1, n_frames, dtype=torch.float32).abs() * 300 + 100
    dummy_z   = torch.randn(1, 1, n_samples, dtype=torch.float32) * 0.03

    # Verify forward pass
    print("Running test forward pass...")
    with torch.no_grad():
        y = full_model(dummy_mel, dummy_f0, dummy_z)
    print(f"  Output shape: {y.shape}, range: [{y.min():.4f}, {y.max():.4f}]")

    # Export to ONNX
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

    # Merge external data into single self-contained .onnx file
    print("Merging into single .onnx file...")
    model_proto = onnx.load(save_path)
    onnx.save_model(model_proto, save_path, save_as_external_data=False)

    data_file = save_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"  Removed: {data_file}")

    file_size = os.path.getsize(save_path)
    print(f"\n  Exported: {save_path} ({file_size / 1024 / 1024:.1f} MB)")
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export trained NHVSing models for inference.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training snapshot (.pth).")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file.")
    parser.add_argument("--output_dir", type=str, default="exported_models",
                        help="Directory to save exported files.")
    parser.add_argument("--all",       action="store_true", help="Export all formats.")
    parser.add_argument("--pytorch",   action="store_true", help="Export state_dict.")
    parser.add_argument("--jit",       action="store_true", help="Export JIT-scripted model.")
    parser.add_argument("--onnx",      action="store_true", help="Export conv-core ONNX.")
    parser.add_argument("--full_onnx", action="store_true", help="Export unified full ONNX.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)

    print("Loading model from checkpoint...")
    model = load_model(args.checkpoint, config)
    print("Model loaded.")

    # Dummy input for conv-core ONNX export
    dummy_x = torch.randn(1, 100, config['preprocess']['mel_dim'], dtype=torch.float32)

    if args.all or args.pytorch:
        export_pytorch_model(model, os.path.join(args.output_dir, "model.pth"))

    if args.all or args.jit:
        export_jit_model(model, os.path.join(args.output_dir, "model_jit.pt"))

    if args.all or args.onnx:
        export_onnx_core(model, os.path.join(args.output_dir, "core_model.onnx"), dummy_x)

    if args.all or args.full_onnx:
        export_full_onnx(
            model, config,
            save_path=os.path.join(args.output_dir, "full_vocoder.onnx"),
            opset=args.opset,
        )


if __name__ == "__main__":
    main()
