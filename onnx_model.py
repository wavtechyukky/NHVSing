import torch
import torch.nn as nn
from layers import ConvLayers


def _make_quef_norm(ccep_size: int, alpha: float) -> torch.Tensor:
    pos = torch.arange(1, ccep_size // 2).float()
    neg = torch.arange(ccep_size // 2, 0, -1).float()
    return torch.cat([torch.ones(1), pos, neg]) ** alpha


class NHVConvsONNX(nn.Module):
    """Dual-branch CNN backbone for NHVSing (legacy).

    Two independent ConvLayers branches predict harmonic and noise complex
    cepstra respectively.  Used by the original NHVSing class for loading
    pre-trained checkpoints that do not use a shared trunk.
    """

    def __init__(self, ltv_params: dict):
        super().__init__()
        ccep_size_harm  = ltv_params["ccep_size"]
        ccep_size_noise = ltv_params.get("ccep_size_noise", ccep_size_harm)
        n_ltv_layers    = ltv_params.get("n_ltv_layers", 10)

        use_quef_norm   = ltv_params.get("use_quef_norm", True)
        alpha           = float(ltv_params.get("quef_norm_alpha", 1.0))
        if use_quef_norm:
            self.register_buffer("quef_norm_harm",  _make_quef_norm(ccep_size_harm,  alpha))
            self.register_buffer("quef_norm_noise", _make_quef_norm(ccep_size_noise, alpha))
        else:
            self.register_buffer("quef_norm_harm",  torch.ones(ccep_size_harm))
            self.register_buffer("quef_norm_noise", torch.ones(ccep_size_noise))

        shared = dict(
            kernel_size=ltv_params["kernel_size"],
            dilation_size=ltv_params["dilation_size"],
            group_size=ltv_params["group_size"],
            n_conv_layers=n_ltv_layers,
            use_causal=ltv_params["use_causal"],
        )
        self.conv_harmonic = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=ltv_params["conv_channels"],
            out_channels=ccep_size_harm,
            **shared,
        )
        self.conv_noise = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=ltv_params["conv_channels"],
            out_channels=ccep_size_noise,
            **shared,
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, D) → ccep_harm, ccep_noise each (B, T, ccep_size)"""
        return (self.conv_harmonic(x) / self.quef_norm_harm,
                self.conv_noise(x)    / self.quef_norm_noise)


class NHVConvsShared(nn.Module):
    """Shared-trunk CNN backbone for NHVSingV2.

    A single ConvLayers trunk builds a common representation; two lightweight
    linear heads then specialise for harmonic and noise cepstra.  Both heads
    are zero-initialised so the network starts from a flat spectrum (no exp()
    overflow).
    """

    def __init__(self, ltv_params: dict):
        super().__init__()
        ccep_size_harm  = ltv_params["ccep_size"]
        ccep_size_noise = ltv_params.get("ccep_size_noise", ccep_size_harm)
        conv_channels   = ltv_params["conv_channels"]
        n_ltv_layers    = ltv_params.get("n_ltv_layers", 10)

        use_quef_norm   = ltv_params.get("use_quef_norm", True)
        alpha           = float(ltv_params.get("quef_norm_alpha", 1.0))
        if use_quef_norm:
            self.register_buffer("quef_norm_harm",  _make_quef_norm(ccep_size_harm,  alpha))
            self.register_buffer("quef_norm_noise", _make_quef_norm(ccep_size_noise, alpha))
        else:
            self.register_buffer("quef_norm_harm",  torch.ones(ccep_size_harm))
            self.register_buffer("quef_norm_noise", torch.ones(ccep_size_noise))

        self.shared_conv = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=ltv_params["kernel_size"],
            dilation_size=ltv_params["dilation_size"],
            group_size=ltv_params["group_size"],
            n_conv_layers=n_ltv_layers,
            use_causal=ltv_params["use_causal"],
        )

        self.head_harm  = nn.Linear(conv_channels, ccep_size_harm)
        self.head_noise = nn.Linear(conv_channels, ccep_size_noise)
        nn.init.zeros_(self.head_harm.weight);  nn.init.zeros_(self.head_harm.bias)
        nn.init.zeros_(self.head_noise.weight); nn.init.zeros_(self.head_noise.bias)

    def forward(self, x: torch.Tensor):
        """x: (B, T, D) → ccep_harm, ccep_noise each (B, T, ccep_size)"""
        h = self.shared_conv(x)
        return (self.head_harm(h)  / self.quef_norm_harm,
                self.head_noise(h) / self.quef_norm_noise)
