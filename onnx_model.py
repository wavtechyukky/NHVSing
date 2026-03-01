import torch
import torch.nn as nn
from layers import ConvLayers

class NHVConvsONNX(nn.Module):
    """
    A model that contains only the convolution parts of NHVSing
    for ONNX export.
    """
    def __init__(self, ltv_params: dict):
        super().__init__()
        # Harmonic and noise branches can have different ccep sizes.
        # Noise only needs a smooth spectral envelope, so a smaller ccep_size suffices.
        ccep_size_harm = ltv_params["ccep_size"]
        ccep_size_noise = ltv_params.get("ccep_size_noise", ltv_params["ccep_size"])

        # Scale by 1/|n| as specified in Liu et al. (Interspeech 2020):
        # "outputs need to be scaled by 1/|n|, as natural complex cepstrums decay at least as fast as 1/|n|"
        # quef_norm[k] = |n(k)|, shape: [1, 1, 2, ..., N//2-1, N//2, N//2-1, ..., 1]
        # Set use_quef_norm: false in ltv_filter config to disable (all ones = no scaling).
        use_quef_norm = ltv_params.get("use_quef_norm", True)
        if use_quef_norm:
            pos_harm = torch.arange(1, ccep_size_harm // 2).float()
            neg_harm = torch.arange(ccep_size_harm // 2, 0, -1).float()
            quef_norm_harm = torch.cat([torch.ones(1), pos_harm, neg_harm])
            pos_noise = torch.arange(1, ccep_size_noise // 2).float()
            neg_noise = torch.arange(ccep_size_noise // 2, 0, -1).float()
            quef_norm_noise = torch.cat([torch.ones(1), pos_noise, neg_noise])
        else:
            quef_norm_harm = torch.ones(ccep_size_harm)
            quef_norm_noise = torch.ones(ccep_size_noise)
        self.register_buffer("quef_norm_harm", quef_norm_harm)
        self.register_buffer("quef_norm_noise", quef_norm_noise)

        n_ltv_layers = ltv_params.get("n_ltv_layers", 10)

        self.conv_harmonic = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=ltv_params["conv_channels"],
            out_channels=ccep_size_harm,
            kernel_size=ltv_params["kernel_size"],
            dilation_size=ltv_params["dilation_size"],
            group_size=ltv_params["group_size"],
            n_conv_layers=n_ltv_layers,
            use_causal=ltv_params["use_causal"],
            conv_type=ltv_params["conv_type"],
        )
        self.conv_noise = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=ltv_params["conv_channels"],
            out_channels=ccep_size_noise,
            kernel_size=ltv_params["kernel_size"],
            dilation_size=ltv_params["dilation_size"],
            group_size=ltv_params["group_size"],
            n_conv_layers=n_ltv_layers,
            use_causal=ltv_params["use_causal"],
            conv_type=ltv_params["conv_type"],
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input mel-cepstrum tensor (B, T, D)

        Returns:
            ccep_harm (Tensor): Harmonic complex cepstrum (B, T, ccep_size)
            ccep_noise (Tensor): Noise complex cepstrum (B, T, ccep_size_noise)
        """
        ccep_harm = self.conv_harmonic(x) / self.quef_norm_harm
        ccep_noise = self.conv_noise(x) / self.quef_norm_noise
        return ccep_harm, ccep_noise

    def load_weights(self, original_model):
        """
        Copies weights from the original NHVSing model.

        Args:
            original_model (NHVSing): The trained NHVSing model.
        """
        self.conv_harmonic.load_state_dict(original_model.ltv_harmonic.conv.state_dict())
        self.conv_noise.load_state_dict(original_model.ltv_noise.conv.state_dict())
