#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.


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
        """
        x: (B, T, D)
        y: (B, T, D)
        """
        x = x.transpose(1, 2)
        y = self.conv1d(x)
        if self.kernel_size > 1 and self.conv1d.padding != (0,):
            if self.use_causal:
                y = y[..., : -self.padding]
            else:
                y = y[..., self.padding // 2 : -self.padding // 2]
        return y.transpose(1, 2)


class ConvLayers(nn.Module):
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
        conv_type: str = "original",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.group_size = group_size
        self.n_conv_layers = n_conv_layers
        self.use_causal = use_causal
        self.conv_layers = self._build_original()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_channels) -> (B, T, out_channels)
        """
        return self.conv_layers(x)

    def _build_original(self) -> nn.Sequential:
        layers = []
        # First layer
        layers += [
            Conv1d(
                self.in_channels,
                self.conv_channels,
                self.kernel_size,
                self.dilation_size,
                group_size=1,
                use_causal=self.use_causal,
            ),
            nn.ReLU(),
        ]
        # Intermediate stack
        for _ in range(self.n_conv_layers):
            layers += [
                Conv1d(
                    self.conv_channels,
                    self.conv_channels,
                    self.kernel_size,
                    self.dilation_size,
                    group_size=self.group_size,
                    use_causal=self.use_causal,
                ),
                nn.ReLU(),
            ]
        # Last layer
        layers += [
            Conv1d(
                self.conv_channels,
                self.out_channels,
                self.kernel_size,
                self.dilation_size,
                group_size=1,
                use_causal=self.use_causal,
            )
        ]
        return nn.Sequential(*layers)