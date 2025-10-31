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

import torch
import torch.nn as nn
import logging
import numpy as np

from distutils.version import LooseVersion
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")

class Discriminator(nn.Module):
    """A discriminator module that combines MelGANMultiScaleDiscriminator and MultiResolutionSTFTDiscriminator.

    This module processes the input audio through both a multi-scale discriminator (MelGAN style)
    and a multi-resolution STFT-based discriminator. The final output is the sum of the outputs
    from both discriminators, providing a comprehensive analysis of the audio's authenticity.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.msd = MelGANMultiScaleDiscriminator()
        self.mfd = MultiResolutionSTFTDiscriminator()

    def forward(self, x):
        # outs_1 = self.mpd(x)
        outs_1 = self.msd(x)
        outs_2 = self.mfd(x)
        return outs_1 + outs_2
    
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

class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=512,
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
    

class MultiResolutionSTFTDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[441, 220, 110],
        win_lengths=[2048, 1024, 512],
        window="hann_window",
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        }
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTDiscriminator, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_discriminator = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_discriminator += [STFTDiscriminator(fft_size=fs, shift_size=ss, win_length=wl, window=window)]

    def forward(self, x):
        x = x.squeeze(1)

        outs = []
        for f in self.stft_discriminator:
            outs += [f(x)]
        return outs


class STFTDiscriminator(torch.nn.Module):
    """STFT-based discriminator that analyzes the spectral representation of audio.

    This discriminator first transforms the input audio waveform into a magnitude spectrogram
    using a Short-Time Fourier Transform (STFT). It then passes this spectrogram through a
    series of 1D convolutional layers to produce a discriminative output. The architecture
    is designed to identify artifacts and inconsistencies in the frequency domain.

    Args:
        fft_size (int): The size of the FFT window.
        shift_size (int): The hop size or step size between consecutive STFT frames.
        win_length (int): The length of the window function.
        window (str): The name of the window function to use (e.g., "hann_window").
        out_channels (int): The number of output channels for the final convolutional layer.
        kernel_sizes (list of int): A list of two kernel sizes for the convolutional layers.
        channels (int): The initial number of channels in the convolutional layers.
        max_downsample_channels (int): The maximum number of channels in the downsampling layers.
        bias (bool): Whether to include a bias term in the convolutional layers.
        downsample_scales (list of int): A list of scales for the downsampling operations.
        nonlinear_activation (str): The name of the non-linear activation function to use.
        nonlinear_activation_params (dict): Parameters for the non-linear activation function.
        pad (str): The type of padding to use in the convolutional layers.
        pad_params (dict): Parameters for the padding function.
    """

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=64,
        max_downsample_channels=512,
        bias=True,
        downsample_scales=[4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
    ):
        super(STFTDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(fft_size // 2 + 1, channels, np.prod(kernel_sizes), bias=bias),
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
                        kernel_size=downsample_scale * 6 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 3,
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

        # apply weight norm
        self.apply_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x):
        x = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]
        return outs


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    spectrum = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
    return spectrum
