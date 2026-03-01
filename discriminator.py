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
        self.convs.append(
            _norm_conv2d(
                2, filters,
                kernel_size=kernel_size,
                padding=_get_2d_padding(kernel_size),
            )
        )

        # dilated convs (progressively downsample freq axis)
        for dilation in dilations:
            self.convs.append(
                _norm_conv2d(
                    filters, filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=_get_2d_padding(kernel_size, (dilation, 1)),
                )
            )

        # square kernel to integrate local info
        sq_kernel = (kernel_size[0], kernel_size[0])
        self.convs.append(
            _norm_conv2d(
                filters, filters,
                kernel_size=sq_kernel,
                padding=_get_2d_padding(sq_kernel),
            )
        )

        # final output: 1 channel (discriminator score)
        self.conv_post = _norm_conv2d(
            filters, 1,
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
        n_ffts=(2048, 1024, 512),
        hop_lengths=(441, 220, 110),
        win_lengths=(2048, 1024, 512),
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
    """

    def __init__(self, use_msd: bool = True, stft_filters: int = 32):
        super().__init__()
        self.use_msd = use_msd
        if use_msd:
            self.msd = MelGANMultiScaleDiscriminator()
        self.ms_stft = MultiScaleComplexSTFTDiscriminator(filters=stft_filters)

    def forward(self, x: torch.Tensor):
        outs = []
        if self.use_msd:
            outs += self.msd(x)
        outs += self.ms_stft(x)
        return outs
