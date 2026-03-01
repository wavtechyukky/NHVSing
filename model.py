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
        self.fft_size_harm = ltv_filter_cfg['fft_size']
        self.fft_size_noise = ltv_filter_cfg.get('fft_size_noise', self.fft_size_harm)
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

    def _forward_impl(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool, noise_std: float):
        """Common forward logic. Returns (y, sig_harm)."""
        actual_std = self.noise_std if noise_std < 0 else noise_std
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        if actual_std > 0:
            z = torch.normal(0.0, actual_std, z_shape).to(x.device)
        else:
            z = torch.zeros(z_shape, device=x.device)

        ccep_harm, ccep_noise = self.convs_onnx(x)

        if no_dsp_grad:
            ccep_harm = ccep_harm.detach()
            ccep_noise = ccep_noise.detach()

        cf0_resampled = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = self.impulse_generator(cf0_resampled, 200, float(self.fs))
        harmonic_source = harmonic_source * (cf0_resampled > 0).float()

        imp_harm = complex_cepstrum_to_imp(ccep_harm, self.fft_size_harm)
        sig_harm = ltv_fir(harmonic_source, imp_harm, self.hop_size)

        imp_noise = complex_cepstrum_to_imp(ccep_noise, self.fft_size_noise)
        sig_noise = ltv_fir(z, imp_noise, self.hop_size)

        y = torch.clamp(sig_harm + sig_noise, -1, 1)
        return y.reshape(x.size(0), -1), sig_harm

    def forward(self, x: torch.Tensor, cf0: torch.Tensor,
                no_dsp_grad: bool = False, noise_std: float = -1.0) -> torch.Tensor:
        """Inference-compatible forward. JIT-scriptable.

        Args:
            x:   (B, T, D) - Log mel-spectrogram
            cf0: (B, 1, T) - Continuous F0
            no_dsp_grad: If True, detach DSP outputs from graph (saves memory).
            noise_std: Override noise std. -1.0 → use self.noise_std.
        Returns:
            y: (B, T * hop_size) - Synthesized waveform
        """
        y, _ = self._forward_impl(x, cf0, no_dsp_grad, noise_std)
        return y

    def forward_train(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool = False,
                      noise_std: float = -1.0):
        """Training forward. Returns (y, sig_harm) for harmonic penalty loss.

        Returns:
            y:        (B, T * hop_size) - Synthesized waveform
            sig_harm: (B, 1, T * hop_size) - Harmonic component (for penalty)
        """
        return self._forward_impl(x, cf0, no_dsp_grad, noise_std)

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
