import torch
import torch.nn as nn
from torch.nn.utils import parametrize, parametrizations

from onnx_model import NHVConvsONNX, NHVConvsShared
from layers import F0Embedder
from dsp import (generate_impulse_train, generate_impulse_train_v3beta,
                 complex_cepstrum_to_imp, ltv_fir)


def repeat_interpolate(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    return torch.repeat_interleave(x, frame_size, dim=-1)

def linear_interpolate(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    """1D 線形補間アップサンプル(論文著者実装)。x:[B,C,n_frame] -> [B,C,n_frame*frame_size]"""
    return torch.nn.functional.interpolate(x, scale_factor=frame_size,
                                           mode="linear", align_corners=False)


def upsample_f0(x: torch.Tensor, frame_size: int, mode: str = "repeat") -> torch.Tensor:
    """F0 アップサンプル切替。'linear'=滑らか(倍音をシャープ化)/ 'repeat'=ZOH(従来・段差)。"""
    if mode == "linear":
        return linear_interpolate(x, frame_size)
    return repeat_interpolate(x, frame_size)


def select_model_class(vocoder_cfg: dict, ltv_filter_cfg: dict):
    """config フラグから生成器クラスを選ぶ。

    優先順: use_v3x(V3X: hop512 入力)> use_v3/use_v6/use_hard_vuv(V3: 最終モデル)
    > use_shared_trunk(V2 legacy)> NHVSing(V1 legacy)。
    ※ use_v6 は既存 ckpt の config 互換のため V3 にマップ(公開版では中間世代 V4/V5 を削除)。
    """
    if ltv_filter_cfg.get('use_v3x', False):           # V3X: hop512 入力 → 内部で 256 補間
        return NHVSingV3X
    if (ltv_filter_cfg.get('use_v3', False) or ltv_filter_cfg.get('use_v6', False)
            or ltv_filter_cfg.get('use_hard_vuv', False)):
        return NHVSingV3                               # V3: 最終モデル(mel のみ・白色励起・uv gate)
    if ltv_filter_cfg.get('use_shared_trunk', False):  # V2 legacy: shared trunk + F0 embed
        return NHVSingV2
    return NHVSing                                     # V1 legacy


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
        self.f0_upsample   = vocoder_cfg.get('f0_upsample', 'repeat')   # 'linear'=滑らか(倍音シャープ)/'repeat'=ZOH(従来)
        self.n_harmonic    = vocoder_cfg.get('n_harmonic', 200)         # cos 重ね合わせ数(倍音数)。既定200=従来

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

        cf0_resampled  = upsample_f0(cf0, self.hop_size, self.f0_upsample)
        harmonic_source = self.impulse_generator(cf0_resampled, self.n_harmonic, float(self.fs))
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

    CONVS_CLS = NHVConvsShared   # V2/V3 共有トランク(harmonic/noise 2ヘッド)

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        self.fs             = vocoder_cfg['sample_rate']
        self.hop_size       = vocoder_cfg['hop_size']
        self.fft_size_harm  = ltv_filter_cfg['fft_size']
        self.fft_size_noise = ltv_filter_cfg.get('fft_size_noise', self.fft_size_harm)
        self.noise_std      = vocoder_cfg['noise_std']
        self.f0_upsample    = vocoder_cfg.get('f0_upsample', 'repeat')   # 'linear'=滑らか(倍音シャープ)/'repeat'=ZOH(従来)
        self.n_harmonic     = vocoder_cfg.get('n_harmonic', 200)         # cos 重ね合わせ数(倍音数)。既定200=従来

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
        self.convs_onnx = self.CONVS_CLS(ltv_params)

        # use_f0_embed=False で f0 を ccep フィルタに埋め込まない(オリジナル NHV 流。
        # f0 は harmonic source の impulse train にのみ使われる)。in_channels も config で
        # mel のみ(80)にすること。True(既定)は従来どおり mel+f0_embed を concat。
        self.use_f0_embed = ltv_filter_cfg.get('use_f0_embed', True)
        if self.use_f0_embed:
            self.f0_embedder = F0Embedder(
                n_bins    = ltv_filter_cfg.get('f0_embed_bins', 256),
                embed_dim = ltv_filter_cfg.get('f0_embed_dim',  128),
                f0_min    = ltv_filter_cfg.get('f0_embed_fmin', 40.0),
                f0_max    = ltv_filter_cfg.get('f0_embed_fmax', 1200.0),
            )
        else:
            self.f0_embedder = None

        # 励起インパルスの選択(既定は白色 soft sigmoid mask):
        #   use_impulse_v3beta : v3beta 励起(未採用実験)= soft mask + 1/k tilt(-6dB/oct)。
        if vocoder_cfg.get('use_impulse_v3beta', False):
            self.impulse_generator = generate_impulse_train_v3beta
        else:
            self.impulse_generator = generate_impulse_train

    def _forward_impl(self, x: torch.Tensor, cf0: torch.Tensor,
                      no_dsp_grad: bool, noise_std: float):
        actual_std = self.noise_std if noise_std < 0 else noise_std
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, actual_std, z_shape).to(x.device) if actual_std > 0 \
            else torch.zeros(z_shape, device=x.device)

        if self.use_f0_embed:
            f0_embed = self.f0_embedder(cf0)          # (B, T, embed_dim)
            x = torch.cat([x, f0_embed], dim=-1)      # (B, T, in_channels)

        ccep_harm, ccep_noise = self.convs_onnx(x)
        if no_dsp_grad:
            ccep_harm  = ccep_harm.detach()
            ccep_noise = ccep_noise.detach()

        cf0_resampled   = upsample_f0(cf0, self.hop_size, self.f0_upsample)
        harmonic_source = self.impulse_generator(cf0_resampled, self.n_harmonic, float(self.fs))
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


# ── hop512 → hop256 整列補間(V3X 用。even=元フレーム, odd=線形中点。mel/f0=線形, uv=hold)──
def _up_lin_t(x):    # [B, T, C] -> [B, 2T-1, C]
    mid = 0.5 * (x[:, :-1] + x[:, 1:])
    out = torch.stack([x[:, :-1], mid], dim=2).reshape(x.shape[0], -1, x.shape[2])
    return torch.cat([out, x[:, -1:]], dim=1)


def _up_lin_c(x):    # [B, 1, T] -> [B, 1, 2T-1]
    mid = 0.5 * (x[..., :-1] + x[..., 1:])
    out = torch.stack([x[..., :-1], mid], dim=-1).reshape(x.shape[0], x.shape[1], -1)
    return torch.cat([out, x[..., -1:]], dim=-1)


def _up_hold_c(x):   # [B, 1, T] -> [B, 1, 2T]
    return torch.repeat_interleave(x, 2, dim=-1)


class NHVSingV3(nn.Module):
    """NHVSing V3 — 公開最終モデル(hop256 native)。

    元論文 NHV(Liu2020)構成: **mel のみ入力**(F0 embedding なし)/ 白色 200 倍音励起 /
    quef_norm α=1.0 / harmonic・noise とも同型 ccep(NHVConvsShared 2ヘッド)。加えて
    uv hard gate(推論時の無声保護)・f0_upsample linear・weight_norm を維持。
    高速化: dsp.ltv_fir は fft_corr(時間相関 grouped-conv1d とビット等価で CPU 約8倍速)。
    state_dict は convs_onnx.*(NHVConvsShared)のみ。

    forward(x, cf0, uv):
        x:   (B, T, mel_dim)  ln-mel(hop256)
        cf0: (B, 1, T)        連続(補間済み)F0[Hz]、0=無声
        uv:  (B, 1, T)        1=無声 / 0=有声(norm_interp_f0 準拠)
    """

    def __init__(self, vocoder_cfg: dict, ltv_filter_cfg: dict):
        super().__init__()
        self.fs             = vocoder_cfg['sample_rate']
        self.hop_size       = vocoder_cfg['hop_size']
        self.fft_size_harm  = ltv_filter_cfg['fft_size']
        self.fft_size_noise = ltv_filter_cfg.get('fft_size_noise', self.fft_size_harm)
        self.noise_std      = vocoder_cfg['noise_std']
        self.f0_upsample    = vocoder_cfg.get('f0_upsample', 'linear')
        self.n_harmonic     = vocoder_cfg.get('n_harmonic', 200)

        ltv_params = {
            **ltv_filter_cfg,
            "in_channels":   ltv_filter_cfg['in_channels'],           # = mel_dim
            "conv_channels": ltv_filter_cfg['conv_channels'],
            "kernel_size":   ltv_filter_cfg.get('kernel_size',   vocoder_cfg['kernel_size']),
            "dilation_size": ltv_filter_cfg.get('dilation_size', vocoder_cfg['dilation_size']),
            "group_size":    ltv_filter_cfg.get('group_size',    vocoder_cfg['group_size']),
            "n_ltv_layers":  ltv_filter_cfg.get('n_ltv_layers',  2),
            "use_causal":    ltv_filter_cfg.get('use_causal',    vocoder_cfg['use_causal']),
            "hop_size":      vocoder_cfg['hop_size'],
        }
        assert ltv_params['in_channels'] == vocoder_cfg['in_channels'], (
            f"V3 は mel のみ入力。ltv in_channels({ltv_params['in_channels']}) を "
            f"mel_dim(vocoder.in_channels={vocoder_cfg['in_channels']}) に合わせること")

        self.convs_onnx = NHVConvsShared(ltv_params)      # quef_norm は config(α=1.0)
        self.impulse_generator = generate_impulse_train    # 白色 200 倍音

        if vocoder_cfg.get('use_weight_norm', True):
            self._apply_weight_norm()

    def _forward_impl(self, x, cf0, uv, no_dsp_grad: bool = False, noise_std: float = -1.0,
                      harmonic_gain: float = 1.0, noise_gain: float = 1.0):
        actual_std = self.noise_std if noise_std < 0 else noise_std
        z_shape = (x.size(0), 1, x.size(1) * self.hop_size)
        z = torch.normal(0.0, actual_std, z_shape).to(x.device) if actual_std > 0 \
            else torch.zeros(z_shape, device=x.device)

        ccep_harm, ccep_noise = self.convs_onnx(x)         # mel のみ(concat なし)
        if no_dsp_grad:
            ccep_harm  = ccep_harm.detach()
            ccep_noise = ccep_noise.detach()

        cf0_resampled   = upsample_f0(cf0, self.hop_size, self.f0_upsample)
        harmonic_source = self.impulse_generator(cf0_resampled, self.n_harmonic, float(self.fs))
        voiced_resampled = repeat_interpolate((1.0 - uv), self.hop_size)   # hard v/uv gate(ZOH)
        harmonic_source  = harmonic_source * voiced_resampled

        sig_harm  = ltv_fir(harmonic_source, complex_cepstrum_to_imp(ccep_harm,  self.fft_size_harm),  self.hop_size)
        sig_noise = ltv_fir(z,               complex_cepstrum_to_imp(ccep_noise, self.fft_size_noise), self.hop_size)

        y = torch.clamp(harmonic_gain * sig_harm + noise_gain * sig_noise, -1, 1)
        return y.reshape(x.size(0), -1), sig_harm, sig_noise

    def forward(self, x, cf0, uv, no_dsp_grad: bool = False, noise_std: float = -1.0,
                harmonic_gain: float = 1.0, noise_gain: float = 1.0):
        y, _, _ = self._forward_impl(x, cf0, uv, no_dsp_grad, noise_std, harmonic_gain, noise_gain)
        return y

    def forward_train(self, x, cf0, uv, no_dsp_grad: bool = False, noise_std: float = -1.0):
        return self._forward_impl(x, cf0, uv, no_dsp_grad, noise_std)

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


class NHVSingV3X(NHVSingV3):
    """NHVSing V3X — V3 を hop512 入力で使えるようにした派生。

    OpenUtau 等が出す hop512 の mel/f0/uv を受け、内部で整列中点補間して hop256 グリッドへ
    戻してから V3 を通す(重みは V3 と共有。state_dict も V3 と同一 = convs_onnx.*)。
    forward(mel512, cf0_512, uv_512): それぞれ hop512(11.6ms/フレーム @44.1k)。
    """

    def forward(self, mel512, cf0_512, uv_512, no_dsp_grad: bool = False, noise_std: float = -1.0,
                harmonic_gain: float = 1.0, noise_gain: float = 1.0):
        mel = _up_lin_t(mel512)
        cf0 = _up_lin_c(cf0_512)
        uv  = _up_hold_c(uv_512)[..., :cf0.shape[-1]]
        return super().forward(mel, cf0, uv, no_dsp_grad, noise_std, harmonic_gain, noise_gain)

    def forward_train(self, mel512, cf0_512, uv_512, no_dsp_grad: bool = False, noise_std: float = -1.0):
        mel = _up_lin_t(mel512)
        cf0 = _up_lin_c(cf0_512)
        uv  = _up_hold_c(uv_512)[..., :cf0.shape[-1]]
        return super().forward_train(mel, cf0, uv, no_dsp_grad, noise_std)
