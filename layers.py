import math

import torch
import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation_size=1,
        group_size=1,
        use_causal=False,
    ):
        super().__init__()
        self.use_causal = use_causal
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation_size

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation_size,
            groups=group_size,
        )
        nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        x = x.transpose(1, 2)
        y = self.conv1d(x)
        if self.kernel_size > 1 and self.conv1d.padding != (0,):
            if self.use_causal:
                y = y[..., : -self.padding]
            else:
                y = y[..., self.padding // 2 : -self.padding // 2]
        return y.transpose(1, 2)


class ConvLayers(nn.Module):
    """Stacked Conv1d layers: input projection → N hidden layers → output projection."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_size: int,
        group_size: int,
        n_conv_layers: int,
        use_causal: bool = False,
    ):
        super().__init__()
        layers = [
            Conv1d(in_channels, conv_channels, kernel_size, dilation_size,
                   group_size=1, use_causal=use_causal),
            nn.ReLU(),
        ]
        for _ in range(n_conv_layers):
            layers += [
                Conv1d(conv_channels, conv_channels, kernel_size, dilation_size,
                       group_size=group_size, use_causal=use_causal),
                nn.ReLU(),
            ]
        layers.append(
            Conv1d(conv_channels, out_channels, kernel_size, dilation_size,
                   group_size=1, use_causal=use_causal)
        )
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class F0Embedder(nn.Module):
    """Quantizes continuous F0 (Hz) to log2-scale bins and embeds to embed_dim.

    Index 0 is reserved for F0 below f0_min (unvoiced / out-of-range).
    Bins 1..n_bins cover [f0_min, f0_max] on a log2 scale.
    """

    def __init__(self, n_bins: int = 256, embed_dim: int = 128,
                 f0_min: float = 40.0, f0_max: float = 1200.0):
        super().__init__()
        self.embed = nn.Embedding(n_bins + 1, embed_dim)
        self.n_bins = n_bins
        self.log_fmin = math.log2(f0_min)
        self.log_frange = math.log2(f0_max / f0_min)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (B, 1, T) Hz → (B, T, embed_dim)"""
        log_f0 = torch.log2(f0.clamp(min=2 ** self.log_fmin))
        idx = ((log_f0 - self.log_fmin) / self.log_frange * self.n_bins
               ).long().clamp(0, self.n_bins)
        return self.embed(idx.squeeze(1))
