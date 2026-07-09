'''
The MIT License (MIT)

Copyright (c) 2020 Zhengxi Liu <xcmyz@outlook.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_2d_padding(kernel_size, dilation=(1, 1)):
    """'same' padding (same spatial size before striding) for Conv2d."""
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def _norm_conv2d(in_ch, out_ch, **kwargs):
    """Conv2d + weight normalization."""
    return nn.utils.weight_norm(nn.Conv2d(in_ch, out_ch, **kwargs))


# ---------------------------------------------------------------------------
# MelGAN Multi-Scale Discriminator
# ---------------------------------------------------------------------------

class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 ):
        """Initilize MelGAN discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs, out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs, out_chs, kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs, out_channels, kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 scales=3,
                 downsample_pooling="AvgPool1d",
                 # follow the official implementation setting
                 downsample_pooling_params={
                     "kernel_size": 4,
                     "stride": 2,
                     "padding": 1,
                     "count_include_pad": False,
                 },
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_weight_norm=True,
                 ):
        """Initilize MelGAN multi-scale discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


# ---------------------------------------------------------------------------
# Complex STFT Discriminator
# Reference: EnCodec (Défossez et al., 2022) Multi-Scale STFT Discriminator
#   https://github.com/facebookresearch/encodec
# ---------------------------------------------------------------------------

class ComplexSTFTSubDiscriminator(nn.Module):
    """Single-scale complex STFT sub-discriminator.

    Takes waveform [B, 1, T], computes complex STFT, stacks real and imaginary
    parts as 2 input channels, and processes through 2D convolution layers.

    Returns List[Tensor] (intermediate feature maps + final score).
    The last element [-1] is the discriminator score; earlier elements are
    used for feature matching loss.

    Architecture:
      - spec_transform: torchaudio.Spectrogram (power=None → complex output)
      - conv_in: (2 → filters) kernel=(3,9), same-pad
      - 3 × dilation conv: (filters → filters) kernel=(3,9), stride=(1,2) in freq,
                            dilation=(1/2/4 in time, 1 in freq)
      - conv_square: (filters → filters) kernel=(3,3), same-pad
      - conv_post: (filters → 1) kernel=(3,3), same-pad  ← final score

    Time axis: no stride (preserves temporal resolution)
    Freq axis: halved at each dilation layer → progressively integrates freq info
    """

    def __init__(
        self,
        filters: int = 32,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size=(3, 9),
        dilations=(1, 2, 4),
        stride=(1, 2),
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # power=None → complex spectrogram output
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            normalized=True,
            center=False,
            pad_mode=None,
            power=None,
        )

        self.convs = nn.ModuleList()

        # input: real + imag = 2 channels
        # EnCodec(DiscriminatorSTFT)忠実: 入力 conv は norm='none'(生 Conv2d, weight_norm無し)。
        # msstftd.py:67-68 で入力層は NormConv2d に norm 引数を渡さない=デフォルト'none'(conv.py:129)。
        # 以降の dilated/square/post のみ weight_norm(_norm_conv2d)。
        self.convs.append(
            nn.Conv2d(
                2, filters,
                kernel_size=kernel_size,
                padding=_get_2d_padding(kernel_size),
            )
        )

        # EnCodec(DiscriminatorSTFT)忠実: filters_scale で層ごとに channel を増やす(max_filters で頭打ち)。
        # filters_scale=1 は channel 固定(従来), >1 で out_chs=filters_scale^(i+1)*filters と増え表現力アップ。
        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                _norm_conv2d(
                    in_chs, out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=_get_2d_padding(kernel_size, (dilation, 1)),
                )
            )
            in_chs = out_chs

        # square kernel to integrate local info
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        sq_kernel = (kernel_size[0], kernel_size[0])
        self.convs.append(
            _norm_conv2d(
                in_chs, out_chs,
                kernel_size=sq_kernel,
                padding=_get_2d_padding(sq_kernel),
            )
        )

        # final output: 1 channel (discriminator score)
        self.conv_post = _norm_conv2d(
            out_chs, 1,
            kernel_size=sq_kernel,
            padding=_get_2d_padding(sq_kernel),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T]
        z = self.spec_transform(x)                   # [B, 1, F, T_frames], complex
        z = torch.cat([z.real, z.imag], dim=1)       # [B, 2, F, T_frames]
        z = z.permute(0, 1, 3, 2)                    # [B, 2, T_frames, F]

        outs = []
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            outs.append(z)
        z = self.conv_post(z)
        outs.append(z)  # last element is the discriminator score
        return outs


class MultiScaleComplexSTFTDiscriminator(nn.Module):
    """Multi-scale complex STFT discriminator.

    Default settings for 44.1kHz:
      - (n_fft=2048, hop=441, win=2048): ~46ms window, coarse time / fine freq
      - (n_fft=1024, hop=220, win=1024): ~23ms window, medium
      - (n_fft=512,  hop=110, win=512):  ~12ms window, fine time / coarse freq
    """

    def __init__(
        self,
        filters: int = 32,
        n_ffts=(1024, 2048, 512),    # EnCodec(msstftd.py)忠実: 各scaleに役割(2048=周波数分解能で低周波の質感・倍音に寄与)
        hop_lengths=(256, 512, 128), # EnCodec デフォルト
        win_lengths=(1024, 2048, 512),
        **kwargs,
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            ComplexSTFTSubDiscriminator(
                filters=filters,
                n_fft=n,
                hop_length=h,
                win_length=w,
                **kwargs,
            )
            for n, h, w in zip(n_ffts, hop_lengths, win_lengths)
        ])

    def forward(self, x: torch.Tensor):
        return [disc(x) for disc in self.discriminators]


# ---------------------------------------------------------------------------
# Mel-conditioned WaveNet Discriminator
# Reference: Liu et al. (2020) "Neural Homomorphic Vocoder" §2.4.2
#   non-causal WaveNet conditioned on log-Mel.
#   - waveform main input + log-Mel as per-layer condition
#   - 14 dilated convolutions, dilation doubling 1..64 then repeat
#   - 64 skip/residual channels, kernel size 3, gated activation
#   Loss: kept as LSGAN (MSELoss) to match the existing NHVSing pipeline
#         (the paper uses hinge, but we prefer compatibility here).
# ---------------------------------------------------------------------------

def _norm_conv1d(in_ch, out_ch, **kwargs):
    """Conv1d + weight normalization (matches MelGAN weight-norm pattern)."""
    return nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, **kwargs))


class WaveNetDiscriminatorBlock(nn.Module):
    """Single non-causal dilated residual block with log-Mel conditioning.

    - dilated Conv1d(ch -> 2*ch, kernel, dilation, two-sided 'same' padding)
    - mel condition injected via cond_proj(mel_ch -> 2*ch, kernel1), added
    - gated activation: tanh(a) * sigmoid(b)
    - residual: (x + res_conv(h)) * 2**-0.5  ;  skip: skip_conv(h)
    """

    def __init__(self, ch: int, mel_ch: int, kernel_size: int = 3, dilation: int = 1,
                 negative_slope: float = 0.2):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # non-causal two-sided 'same' padding
        # gated activation(tanh*sigmoid)は飽和・発散しやすく、skip 合計(14層)も
        # 出力スケールを膨張させ fp16 overflow(inf)の主因だった。
        # → LeakyReLU + 残差のみのシンプルな dilated conv に変更(skip 廃止)。
        self.dconv = _norm_conv1d(
            ch, ch, kernel_size=kernel_size, dilation=dilation, padding=pad
        )
        self.cond_proj = _norm_conv1d(mel_ch, ch, kernel_size=1)
        self.res_conv = _norm_conv1d(ch, ch, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x, cond):
        # x: [B, ch, T], cond: [B, mel_ch, T]  (already upsampled to T)
        h = self.act(self.dconv(x) + self.cond_proj(cond))   # condition injection + 非飽和活性
        res = (x + self.res_conv(h)) * (2.0 ** -0.5)         # 残差のみ(skip 合計を断つ)
        return res


class WaveNetDiscriminator(nn.Module):
    """Mel-conditioned non-causal WaveNet discriminator (NHV §2.4.2).

    forward(wav, mel):
        wav: [B, 1, T_samples]
        mel: [B, mel_ch, T_frames]  (log-Mel condition)
    Returns List[Tensor] = [fmap1, fmap2, ..., logit] where the last element
    is the discriminator score and the earlier ones are feature maps for
    feature-matching loss (interface compatible with the other sub-discs).
    """

    # in_conv で stride=downsample(2) するため 1 sample が実時間で downsample 倍。
    # dilation [1..64]*2(14層)で RF = 1 + 2·254 = 509 sample(downsample後)。実時間では
    # 509·downsample/48000 = 21.2ms @48kHz ≈ 論文の 23ms。F0最低~47Hz の1周期をカバー。
    # dilation doubling 1..64, repeated -> 14 layers
    DILATIONS = [1, 2, 4, 8, 16, 32, 64] * 2

    def __init__(
        self,
        ch: int = 32,
        mel_ch: int = 80,
        kernel_size: int = 3,
        hop_size: int = 256,
        downsample: int = 2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.downsample = downsample
        # strided input conv = anti-aliased downsample（kernel>1, stride=downsample）。
        # sample 数を 1/downsample に減らし OOM/速度を緩和（Nyquist=24000/downsample, 2なら12kHz）。
        # kernel=2·downsample, padding=downsample//2 で出力長 ≈ T_samples/downsample。
        self.in_conv = _norm_conv1d(1, ch, kernel_size=2 * downsample,
                                    stride=downsample, padding=downsample // 2)
        self.blocks = nn.ModuleList([
            WaveNetDiscriminatorBlock(ch, mel_ch, kernel_size=kernel_size, dilation=d)
            for d in self.DILATIONS
        ])
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.out_conv1 = _norm_conv1d(ch, ch, kernel_size=1)
        self.out_conv2 = _norm_conv1d(ch, 1, kernel_size=1)

    def _upsample_mel(self, mel, T_target):
        # in_conv で sample が 1/downsample になるので、mel は hop_size/downsample 倍に上げて
        # downsample 後の時間長に合わせる（hop_size は downsample で割り切れる前提: 256/2=128）。
        mel = mel.repeat_interleave(self.hop_size // self.downsample, dim=-1)
        T = min(mel.size(-1), T_target)
        return mel, T

    def forward(self, wav, mel):
        # wav: [B, 1, T_samples], mel: [B, mel_ch, T_frames]
        x = self.in_conv(wav)         # strided downsample -> [B, ch, ~T_samples/downsample]
        mel, T = self._upsample_mel(mel, x.size(-1))
        x = x[..., :T]
        cond = mel[..., :T]
        outs = []
        for blk in self.blocks:
            x = blk(x, cond)
            outs.append(x)            # collect per-layer outputs as feature maps
        # skip 合計を廃止し、最終層出力を head に通す(スケール膨張・発散の回避)。
        h = self.activation(self.out_conv1(x))
        outs.append(h)                # feature map
        logit = self.out_conv2(h)     # final score
        outs.append(logit)
        return outs


# ---------------------------------------------------------------------------
# Combined discriminator (main entry point)
# ---------------------------------------------------------------------------

class DiscriminatorWithComplexSTFT(nn.Module):
    """Combined discriminator: MelGAN MSD (waveform) + Multi-Scale Complex STFT (phase-aware).

    Interface is fully compatible with the old Discriminator class:
      forward(x) → List[List[Tensor]]
      Each sublist's [-1] is the discriminator score; [:-1] are feature maps
      for feature matching loss.

    Args:
        use_msd (bool): Whether to include MelGAN MSD. Set False to save memory.
        stft_filters (int): Number of filters in the complex STFT discriminator.
        use_wavenet (bool): Whether to include the mel-conditioned WaveNet disc
            (NHV §2.4.2). Requires `mel` to be passed to forward().
        wavenet_ch (int): Residual/skip channels of the WaveNet disc.
    """

    def __init__(self, use_msd: bool = True, stft_filters: int = 32,
                 use_wavenet: bool = False, wavenet_ch: int = 32,
                 stft_filters_scale: int = 1):
        super().__init__()
        self.use_msd = use_msd
        if use_msd:
            self.msd = MelGANMultiScaleDiscriminator()
        self.ms_stft = MultiScaleComplexSTFTDiscriminator(
            filters=stft_filters, filters_scale=stft_filters_scale)
        self.use_wavenet = use_wavenet
        if use_wavenet:
            self.wavenet = WaveNetDiscriminator(ch=wavenet_ch)

    def forward(self, x: torch.Tensor, mel: torch.Tensor = None):
        # mel=None and use_wavenet=False -> identical to the legacy behaviour.
        outs = []
        if self.use_msd:
            outs += self.msd(x)
        outs += self.ms_stft(x)
        if self.use_wavenet and mel is not None:
            outs += [self.wavenet(x, mel)]
        return outs


# ===========================================================================
# NSF-HiFiGAN 標準 discriminator (MSD + MPD) — SingingVocoders(OpenVPI)由来。
# 自前 disc(MSD melgan + 複素STFT + WaveNet)が 0.5 陥落=品質差を捉えられなかった
# ため、実績ある HiFi-GAN disc に置換。weight_norm/spectral_norm で安定(AMP でも
# inf しにくい)。波形入力のみで mel 仕様(48k/80)非依存。
# ===========================================================================

_HIFIGAN_LRELU = 0.1


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorS(nn.Module):
    """Multi-Scale 用の単一スケール判別器(波形 [B,1,T] 入力)。"""
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), _HIFIGAN_LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y):
        logits, fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
            logit, fmap = d(y)
            logits.append(logit); fmaps.append(fmap)
        return logits, fmaps


class DiscriminatorP(nn.Module):
    """Multi-Period 用の単一周期判別器(波形を period で 2D reshape)。"""
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        pad = (_get_padding(5, 1), 0)
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), _HIFIGAN_LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=None):
        super().__init__()
        self.periods = periods if periods is not None else [3, 5, 7, 11, 17, 23, 37]
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in self.periods])

    def forward(self, y):
        logits, fmaps = [], []
        for d in self.discriminators:
            logit, fmap = d(y)
            logits.append(logit); fmaps.append(fmap)
        return logits, fmaps


# ---------------------------------------------------------------------------
# UnivNet Multi-Resolution Spectrogram Discriminator (magnitude のみ)
# Reference: UnivNet (Jang et al., 2021) / RefineGAN(ACE)も採用。univnet/model/mrd.py 由来。
#   torch.stft → magnitude(abs)。複素STFT disc(real+imag, 0.5陥落)と違い magnitude のみで、
#   位相判定は MPD に任せ高周波 magnitude に専念(SOTA 2種=UnivNet/RefineGAN 共通の分業)。
#   HiFiGANDiscriminator(MSD+MPD)に "単に追加" して MSD+MPD+MRD にする(MSD/MPD は継承)。
# ---------------------------------------------------------------------------

class DiscriminatorR(nn.Module):
    """単一解像度 magnitude spectrogram 判別器。resolution=(n_fft, hop, win)。"""
    def __init__(self, resolution, use_spectral_norm=False, lrelu_slope=_HIFIGAN_LRELU):
        super().__init__()
        self.resolution = resolution
        self.lrelu_slope = lrelu_slope
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def _spectrogram(self, x):
        n_fft, hop, win = self.resolution
        pad = int((n_fft - hop) / 2)
        x = F.pad(x, (pad, pad), mode='reflect').squeeze(1)
        # UnivNet 原典(mrd.py)は window 未指定=矩形窓。原典と数値一致させるため ones を
        # 明示(spectral-leakage 警告の抑制も兼ねる)。win_length の矩形窓 + n_fft へ zero-pad。
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=torch.ones(win, device=x.device),
                          center=False, return_complex=True)  # [B, F, T]
        return torch.abs(spec)  # magnitude のみ(複素STFTと違い位相成分は持たない)

    def forward(self, x):
        fmap = []
        x = self._spectrogram(x).unsqueeze(1)  # [B, 1, F, T]
        for l in self.convs:
            x = F.leaky_relu(l(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    """UnivNet MRD。resolutions = (n_fft, hop, win) のリスト。default は UnivNet 公式値
    (22kHz 用だが、RefineGAN は 44.1kHz でもこれをスケールせず使用=SOTA の実践)。"""
    def __init__(self, resolutions=None):
        super().__init__()
        self.resolutions = resolutions if resolutions is not None else [
            (1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
        self.discriminators = nn.ModuleList([DiscriminatorR(tuple(r)) for r in self.resolutions])

    def forward(self, y):
        logits, fmaps = [], []
        for d in self.discriminators:
            logit, fmap = d(y)
            logits.append(logit); fmaps.append(fmap)
        return logits, fmaps


class HiFiGANDiscriminator(nn.Module):
    """MSD / MPD / MRD を任意に組み合わせる disc。NHVSing IF 互換:
    forward(x, mel=None) → List[List[Tensor]]。各 sublist の [-1]=logit, [:-1]=feature maps
    なので既存 train.py の adv/disc/feature-matching loss がそのまま動く。mel は未使用(波形のみ)。

    - use_msd: MelGAN/HiFi-GAN 流 Multi-Scale(波形 magnitude)
    - use_mpd: Multi-Period(周期/位相)。period×N の 2D conv で重い。効果薄なら False
    - use_mrd: UnivNet Multi-Resolution(spectrogram magnitude)。RefineGAN/UnivNet 採用

    forward 出力順は常に [MSD..., MPD..., MRD...](存在する群のみ)。"""
    def __init__(self, periods=None, use_msd=True, use_mpd=True,
                 use_mrd=False, mrd_resolutions=None):
        super().__init__()
        assert use_msd or use_mpd or use_mrd, "at least one sub-discriminator must be enabled"
        self.use_msd = use_msd
        self.use_mpd = use_mpd
        self.use_mrd = use_mrd
        if use_msd:
            self.msd = MultiScaleDiscriminator()
        if use_mpd:
            self.mpd = MultiPeriodDiscriminator(periods=periods)
        if use_mrd:
            self.mrd = MultiResolutionDiscriminator(resolutions=mrd_resolutions)

    def forward(self, x, mel=None):
        outs = []
        if self.use_msd:
            for logit, fmap in zip(*self.msd(x)):
                outs.append(fmap + [logit])
        if self.use_mpd:
            for logit, fmap in zip(*self.mpd(x)):
                outs.append(fmap + [logit])
        if self.use_mrd:
            for logit, fmap in zip(*self.mrd(x)):
                outs.append(fmap + [logit])
        return outs


# ===========================================================================
# Wave-U-Net Discriminator — Kaneko et al. (2023), arXiv:2303.13909
# 単一 disc(U-Net: encoder-decoder + skip)で HiFi-GAN disc アンサンブル(MSD+MPD)と
# 同等品質、2.31倍速・14.5倍軽量(70.7M→4.9M)。NHVSing では disc が学習律速だった
# (disc OFF 22秒 vs ON 360秒=disc 94%)ため導入。出典: Wave-U-Net-Discriminator-Pytorch
# /discriminators.py(公式実装)を移植。安定化: Global Norm + residual×0.4(論文 §3.2)。
# ===========================================================================

class GlobalNorm(nn.Module):
    """feature を RMS で正規化(trainable param 無)。論文 §3.2 default は dim=(1,2)(全
    channel×時間)。DSP generator では fake/real の微細差(振幅/位相)を消す疑いがあり、
    dim=(1)(各時刻の channel のみ=時間方向の振幅変動を保持)で差を残せるか実験するため可変に。"""
    def __init__(self, eps: float = 1e-8, dim=(1, 2)):
        super().__init__()
        self.eps = eps
        self.dim = tuple(dim)

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x / norm


class ResBlockDown(nn.Module):
    """downsampling residual block(stride 3)。skip(AvgPool + 1x1 conv で channel 拡張
    + concat)+ residual(conv k6 s3)×0.4 → GlobalNorm。"""
    def __init__(self, in_ch: int, out_ch: int, gn_dim=(1, 2)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_pool = nn.AvgPool1d(kernel_size=3, stride=3)
        self.skip_conv = nn.Conv1d(in_ch, out_ch - in_ch, kernel_size=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=6, stride=3, padding=2)
        self.res_pool = nn.AvgPool1d(kernel_size=3, stride=3)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.norm = GlobalNorm(dim=gn_dim)

    def _dup_channels(self, x):
        B, C, T = x.shape
        if C == self.out_ch:
            return x
        reps = -(-self.out_ch // C)
        return x.repeat(1, reps, 1)[:, :self.out_ch, :]

    def forward(self, x):
        pool = self.skip_pool(x)
        skip = self.skip_conv(pool)
        skip = torch.cat([pool, skip], dim=1)
        res = self.lrelu1(x)
        r = self.conv1(res)
        sc = self.res_pool(res)
        sc = self._dup_channels(sc)
        r = r + sc
        r = self.lrelu2(r)
        r = self.conv2(r)
        out = skip + (r * 0.4)
        return self.norm(out)


class ResBlockUp(nn.Module):
    """upsampling residual block(stride 3, ConvTranspose)。skip(1x1 conv + upsample)
    + residual(convT k6 s3)×0.4 → GlobalNorm。"""
    def __init__(self, in_ch: int, out_ch: int, gn_dim=(1, 2)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.convt = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=6, stride=3,
                                        padding=2, output_padding=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.norm = GlobalNorm(dim=gn_dim)

    @staticmethod
    def _upsample(x, factor: int = 3):
        return x.repeat_interleave(factor, dim=2)

    def _drop_channels(self, x):
        return x[:, :self.out_ch, :]

    def forward(self, x):
        skip = self.skip_conv(x)
        skip = self._upsample(skip)
        res = self.lrelu1(x)
        r = self.convt(res)
        sc = self._upsample(res)
        sc = self._drop_channels(sc)
        r = r + sc
        r = self.lrelu2(r)
        r = self.conv2(r)
        out = skip + (r * 0.4)
        return self.norm(out)


class WaveUNetDiscriminator(nn.Module):
    """単一 U-Net disc。sample-wise(入力と同解像度)で real/fake を判定し、
    enc-dec + skip で multilevel feature を抽出。forward(x) → (logits[B,1,T], features[10層])。"""
    def __init__(self, gn_dim=(1, 2)):
        super().__init__()
        self.enc1 = ResBlockDown(1, 32, gn_dim)
        self.enc2 = ResBlockDown(32, 64, gn_dim)
        self.enc3 = ResBlockDown(64, 128, gn_dim)
        self.enc4 = ResBlockDown(128, 256, gn_dim)
        self.enc5 = ResBlockDown(256, 512, gn_dim)
        self.dec1 = ResBlockUp(512, 256, gn_dim)
        self.dec2 = ResBlockUp(512, 128, gn_dim)
        self.dec3 = ResBlockUp(256, 64, gn_dim)
        self.dec4 = ResBlockUp(128, 32, gn_dim)
        self.dec5 = ResBlockUp(64, 32, gn_dim)
        self.out_conv = nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2)
        self.reset_parameters()

    def reset_parameters(self):
        """HiFi-GAN 流 weight 初期化(全 conv を normal(0, 0.01))。論文は official HiFi-GAN
        実装ベースで、HiFi-GAN は init_weights で conv を normal(0, 0.01)で初期化する。
        移植元(unofficial)は PyTorch default(kaiming_uniform=大きめ)だったため明示的に合わせる。
        小さい初期 weight で初期 logit が穏やかになり、disc が安定して学習しやすくなる。"""
        def _init(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.apply(_init)

    def pad_to_multiple(self, x, multiple: int = 243):
        T = x.shape[-1]
        pad = (multiple - T % multiple) % multiple
        return F.pad(x, (0, pad)), pad

    def forward(self, x):
        x, pad = self.pad_to_multiple(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        d5 = self.dec5(torch.cat([d4, e1], dim=1))
        logits = self.out_conv(d5)
        if pad > 0:
            logits = logits[..., :-pad]
        features = [e1, e2, e3, e4, e5, d1, d2, d3, d4, d5]
        return logits, features


class WaveUNetDiscriminatorNHV(nn.Module):
    """NHVSing IF ラッパー: forward(x, mel=None) → List[List[Tensor]]。単一 disc なので
    sublist は1個([features..., logit])。train_gan.py の群動的(groups=[('wun', wun, 1)])が
    そのまま動く(adv/disc loss は logit[-1] に LSGAN、feature matching は features[:-1])。"""
    def __init__(self, gn_dim=(1, 2)):
        super().__init__()
        self.wun = WaveUNetDiscriminator(gn_dim=gn_dim)

    def forward(self, x, mel=None):
        logits, features = self.wun(x)
        return [features + [logits]]
