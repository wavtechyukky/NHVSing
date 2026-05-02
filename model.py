import torch
import torch.nn as nn
from torch.nn.utils import parametrize, parametrizations

from onnx_model import NHVConvsONNX, NHVConvsShared
from layers import F0Embedder
from dsp import generate_impulse_train, complex_cepstrum_to_imp, ltv_fir


def repeat_interpolate(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    return torch.repeat_interleave(x, frame_size, dim=-1)


class NHVSing(nn.Module):
    """Original NHVSing (dual-branch, no shared trunk, no F0 embedder).

    Kept for loading pre-trained checkpoints exported before NHVSingV2.
    Architecture: NHVConvsONNX → complex cepstrum → LTV FIR DSP.
    """

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        self.fs            = vocoder_cfg['sample_rate']
        self.hop_size      = vocoder_cfg['hop_size']
        self.fft_size_harm = ltv_filter_cfg['fft_size']
        self.fft_size_noise = ltv_filter_cfg.get('fft_size_noise', self.fft_size_harm)
        self.noise_std     = vocoder_cfg['noise_std']

        ltv_params = {
            **ltv_filter_cfg,
            "in_channels":   ltv_filter_cfg.get('in_channels',   vocoder_cfg['in_channels']),
            "conv_channels": ltv_filter_cfg.get('conv_channels', vocoder_cfg['conv_channels']),
            "kernel_size":   ltv_filter_cfg.get('kernel_size',   vocoder_cfg['kernel_size']),
            "dilation_size": ltv_filter_cfg.get('dilation_size', vocoder_cfg['dilation_size']),
            "group_size":    ltv_filter_cfg.get('group_size',    vocoder_cfg['group_size']),
            "n_ltv_layers":  ltv_filter_cfg.get('n_ltv_layers',  10),
            "use_causal":    ltv_filter_cfg.get('use_causal',    vocoder_cfg['use_causal']),
            "hop_size":      vocoder_cfg['hop_size'],
        }
        self.convs_onnx = NHVConvsONNX(ltv_params)
        # Zero-init output layers → flat spectrum at epoch 0
        for branch in [self.convs_onnx.conv_harmonic, self.convs_onnx.conv_noise]:
            last = branch.conv_layers[-1]
            nn.init.zeros_(last.conv1d.weight)
            nn.init.zeros_(last.conv1d.bias)

        self.impulse_generator = generate_impulse_train

    def _forward_impl(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool, noise_std: float):
        actual_std = self.noise_std if noise_std < 0 else noise_std
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, actual_std, z_shape).to(x.device) if actual_std > 0 \
            else torch.zeros(z_shape, device=x.device)

        ccep_harm, ccep_noise = self.convs_onnx(x)
        if no_dsp_grad:
            ccep_harm  = ccep_harm.detach()
            ccep_noise = ccep_noise.detach()

        cf0_resampled  = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = self.impulse_generator(cf0_resampled, 200, float(self.fs))
        harmonic_source = harmonic_source * (cf0_resampled > 0).float()

        sig_harm  = ltv_fir(harmonic_source, complex_cepstrum_to_imp(ccep_harm,  self.fft_size_harm),  self.hop_size)
        sig_noise = ltv_fir(z,               complex_cepstrum_to_imp(ccep_noise, self.fft_size_noise), self.hop_size)

        y = torch.clamp(sig_harm + sig_noise, -1, 1)
        return y.reshape(x.size(0), -1), sig_harm, sig_noise

    def forward(self, x: torch.Tensor, cf0: torch.Tensor,
                no_dsp_grad: bool = False, noise_std: float = -1.0) -> torch.Tensor:
        y, _, _ = self._forward_impl(x, cf0, no_dsp_grad, noise_std)
        return y

    def forward_train(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool = False, noise_std: float = -1.0):
        return self._forward_impl(x, cf0, no_dsp_grad, noise_std)

    def remove_weight_norm(self):
        def _remove(m):
            if parametrize.is_parametrized(m, "weight"):
                parametrize.remove_parametrizations(m, "weight")
        self.convs_onnx.apply(_remove)

    def _apply_weight_norm(self):
        def _apply(m):
            if isinstance(m, torch.nn.Conv1d):
                parametrizations.weight_norm(m)
        self.convs_onnx.apply(_apply)


class NHVSingV2(nn.Module):
    """NHVSing V2 — shared-trunk CNN with F0 embedder.

    Improvements over NHVSing:
      - Shared CNN trunk: one backbone feeds both harmonic and noise heads,
        learning a common harmonic/noise decomposition rather than having each
        branch rediscover it independently.
      - F0 embedder: continuous F0 is quantized to log2-scale bins and embedded
        to 128-dim, then concatenated with the mel before the CNN.  This gives
        the network an explicit, voiced/unvoiced-aware pitch signal rather than
        relying solely on the mel spectrogram.
      - quef_norm (alpha=0.3): soft 1/|n|^alpha scaling of the complex cepstrum
        encourages natural spectral envelope decay without over-constraining the
        network.
      - Amplitude augmentation during training (0.5–2.0×, log-uniform): makes
        the model robust to input volume variation.

    Config: use config_m4singer_ampaugment.yaml.
    """

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        self.fs             = vocoder_cfg['sample_rate']
        self.hop_size       = vocoder_cfg['hop_size']
        self.fft_size_harm  = ltv_filter_cfg['fft_size']
        self.fft_size_noise = ltv_filter_cfg.get('fft_size_noise', self.fft_size_harm)
        self.noise_std      = vocoder_cfg['noise_std']

        ltv_params = {
            **ltv_filter_cfg,
            "in_channels":   ltv_filter_cfg['in_channels'],   # 256 (mel 128 + f0_embed 128)
            "conv_channels": ltv_filter_cfg['conv_channels'],
            "kernel_size":   ltv_filter_cfg.get('kernel_size',   vocoder_cfg['kernel_size']),
            "dilation_size": ltv_filter_cfg.get('dilation_size', vocoder_cfg['dilation_size']),
            "group_size":    ltv_filter_cfg.get('group_size',    vocoder_cfg['group_size']),
            "n_ltv_layers":  ltv_filter_cfg.get('n_ltv_layers',  2),
            "use_causal":    ltv_filter_cfg.get('use_causal',    vocoder_cfg['use_causal']),
            "hop_size":      vocoder_cfg['hop_size'],
        }
        self.convs_onnx = NHVConvsShared(ltv_params)

        self.f0_embedder = F0Embedder(
            n_bins    = ltv_filter_cfg.get('f0_embed_bins', 256),
            embed_dim = ltv_filter_cfg.get('f0_embed_dim',  128),
            f0_min    = ltv_filter_cfg.get('f0_embed_fmin', 40.0),
            f0_max    = ltv_filter_cfg.get('f0_embed_fmax', 1200.0),
        )

        self.impulse_generator = generate_impulse_train

    def _forward_impl(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool, noise_std: float):
        actual_std = self.noise_std if noise_std < 0 else noise_std
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, actual_std, z_shape).to(x.device) if actual_std > 0 \
            else torch.zeros(z_shape, device=x.device)

        f0_embed = self.f0_embedder(cf0)              # (B, T, embed_dim)
        x = torch.cat([x, f0_embed], dim=-1)          # (B, T, in_channels)

        ccep_harm, ccep_noise = self.convs_onnx(x)
        if no_dsp_grad:
            ccep_harm  = ccep_harm.detach()
            ccep_noise = ccep_noise.detach()

        cf0_resampled   = repeat_interpolate(cf0, self.hop_size)
        harmonic_source = self.impulse_generator(cf0_resampled, 200, float(self.fs))
        harmonic_source = harmonic_source * (cf0_resampled > 0).float()

        sig_harm  = ltv_fir(harmonic_source, complex_cepstrum_to_imp(ccep_harm,  self.fft_size_harm),  self.hop_size)
        sig_noise = ltv_fir(z,               complex_cepstrum_to_imp(ccep_noise, self.fft_size_noise), self.hop_size)

        y = torch.clamp(sig_harm + sig_noise, -1, 1)
        return y.reshape(x.size(0), -1), sig_harm, sig_noise

    def forward(self, x: torch.Tensor, cf0: torch.Tensor,
                no_dsp_grad: bool = False, noise_std: float = -1.0) -> torch.Tensor:
        """
        Args:
            x:   (B, T, 128)  log mel-spectrogram
            cf0: (B, 1, T)    continuous F0 in Hz (interpolated, 0 = unvoiced)
        Returns:
            y:   (B, T * hop_size)  synthesized waveform in [-1, 1]
        """
        y, _, _ = self._forward_impl(x, cf0, no_dsp_grad, noise_std)
        return y

    def forward_train(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool = False, noise_std: float = -1.0):
        """Returns (y, sig_harm, sig_noise) for penalty losses."""
        return self._forward_impl(x, cf0, no_dsp_grad, noise_std)

    def remove_weight_norm(self):
        def _remove(m):
            if parametrize.is_parametrized(m, "weight"):
                parametrize.remove_parametrizations(m, "weight")
        self.convs_onnx.apply(_remove)

    def _apply_weight_norm(self):
        def _apply(m):
            if isinstance(m, torch.nn.Conv1d):
                parametrizations.weight_norm(m)
        self.convs_onnx.apply(_apply)

