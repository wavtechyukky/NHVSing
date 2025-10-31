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
        # quef_norm is needed for scaling the output of conv
        ccep_size = ltv_params["ccep_size"]
        idx = torch.arange(1, ccep_size // 2 + 1).float()
        quef_norm = torch.cat([torch.flip(idx, dims=[-1]), idx], dim=-1)
        self.register_buffer("quef_norm", quef_norm)

        # Create two convolution modules, one for harmonic and one for noise
        # These parameters should be passed from the main config
        n_ltv_layers = ltv_params.get("n_ltv_layers", 10) # Default from a config if not present

        self.conv_harmonic = ConvLayers(
            in_channels=ltv_params["in_channels"],
            conv_channels=ltv_params["conv_channels"],
            out_channels=ltv_params["ccep_size"],
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
            out_channels=ltv_params["ccep_size"],
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
            ccep_noise (Tensor): Noise complex cepstrum (B, T, ccep_size)
        """
        ccep_harm = self.conv_harmonic(x) / self.quef_norm
        ccep_noise = self.conv_noise(x) / self.quef_norm
        return ccep_harm, ccep_noise

    def load_weights(self, original_model):
        """
        Copies weights from the original NHVSing model.
        
        Args:
            original_model (NHVSing): The trained NHVSing model.
        """
        self.conv_harmonic.load_state_dict(original_model.ltv_harmonic.conv.state_dict())
        self.conv_noise.load_state_dict(original_model.ltv_noise.conv.state_dict())
