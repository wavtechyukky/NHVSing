import torch
import torch.nn as nn
from torch.nn.utils import parametrize, parametrizations

# Import the ONNX-exportable convolution module
from onnx_model import NHVConvsONNX

# Import DSP functions and helpers from original files
from dsp import generate_impulse_train, complex_cepstrum_to_imp, ltv_fir


def repeat_interpolate(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    return torch.repeat_interleave(x, frame_size, dim=-1)


class NHVSing(nn.Module):
    """
    A refactored version of NHVSing that uses NHVConvsONNX as an internal module.
    This model is functionally identical to NHVSing for training purposes,
    but cleanly separates the convolutional parts from the DSP parts.
    """
    def __init__(
        self,
        vocoder_cfg: dict,
        ltv_filter_cfg: dict,
    ):
        super().__init__()
        # Store necessary parameters
        self.fs = vocoder_cfg['sample_rate']
        self.hop_size = vocoder_cfg['hop_size']
        self.fft_size = ltv_filter_cfg['fft_size']
        self.noise_std = vocoder_cfg['noise_std']
        
        # This is the ONNX-exportable part
        ltv_params = {
            **ltv_filter_cfg,
            "in_channels": vocoder_cfg['in_channels'],
            "conv_channels": vocoder_cfg['conv_channels'],
            "kernel_size": vocoder_cfg['kernel_size'],
            "dilation_size": vocoder_cfg['dilation_size'],
            "group_size": vocoder_cfg['group_size'],
            "n_ltv_layers": ltv_filter_cfg.get("n_ltv_layers", 10), # Ensure this key exists
            "use_causal": vocoder_cfg['use_causal'],
            "conv_type": vocoder_cfg['conv_type'],
            "hop_size": vocoder_cfg['hop_size'],
        }
        self.convs_onnx = NHVConvsONNX(ltv_params)

        # DSP functions (not part of the ONNX graph)
        self.impulse_generator = generate_impulse_train

    def forward(self, x, cf0):
        """
        Args:
            x: (B, T, D) - Mel-cepstrum
            cf0: (B, 1, T) - Continuous F0
        
        Returns:
            y: (B, T * hop_size) - Synthesized waveform
        """
        # Generate noise internally
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, self.noise_std, z_shape).to(x.device)

        # --- Part 1: ONNX-able convolutions ---
        ccep_harm, ccep_noise = self.convs_onnx(x)

        # --- Part 2: Non-exportable DSP processing ---
        # Generate harmonic source signal
        cf0_resampled = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = self.impulse_generator(cf0_resampled, 200, float(self.fs))

        # Process harmonic part
        imp_harm = complex_cepstrum_to_imp(ccep_harm, self.fft_size)
        sig_harm = ltv_fir(harmonic_source, imp_harm, self.hop_size)

        # Process noise part
        imp_noise = complex_cepstrum_to_imp(ccep_noise, self.fft_size)
        sig_noise = ltv_fir(z, imp_noise, self.hop_size)
        
        # Combine and clamp
        y = sig_harm + sig_noise
        y = torch.clamp(y, -1, 1)
        
        return y.reshape(x.size(0), -1)

    def remove_weight_norm(self):
        """Removes weight normalization from the convolutional layers."""
        def _remove(m):
            if parametrize.is_parametrized(m, "weight"):
                parametrize.remove_parametrizations(m, "weight")
        # Apply to the sub-module that contains the convolutions
        self.convs_onnx.apply(_remove)

    def _apply_weight_norm(self):
        """Applies weight normalization to the convolutional layers."""
        def _apply(m):
            if isinstance(m, torch.nn.Conv1d):
                parametrizations.weight_norm(m)
        # Apply to the sub-module that contains the convolutions
        self.convs_onnx.apply(_apply)
