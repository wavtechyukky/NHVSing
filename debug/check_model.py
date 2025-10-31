# debug/check_model.py

import torch
import numpy as np
import yaml
import soundfile as sf
import argparse
from pathlib import Path

# Assuming refactored_model.py is in the same root directory
from model import NHVSing

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(npz_path, output_wav_path, onnx_path=None):
    """
    Instantiates NHVSing, runs inference on an NPZ file,
    saves the output audio, and optionally exports the conv part to ONNX.
    """
    # 1. Load config
    print("Loading configuration from config.yaml...")
    cfg = load_config('config.yaml')
    model_cfg = cfg['model']
    preprocess_cfg = cfg['preprocess']

    # 2. Instantiate model
    print("Instantiating NHVSing model...")
    model = NHVSing(
        vocoder_cfg=model_cfg['vocoder'],
        ltv_filter_cfg=model_cfg['ltv_filter']
    )
    model.eval()  # Set to evaluation mode
    print("Model instantiated successfully.")

    # 3. Load data from NPZ file
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    f0 = data['f0']
    log_melspc = data['log_melspc']
    print(f"  - F0 shape: {f0.shape}")
    print(f"  - Log Mel-Spectrogram shape: {log_melspc.shape}")

    # 4. Convert to tensor and adjust dimensions for batch processing
    # f0: (T,) -> (B, 1, T)
    f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
    # log_melspc: (T, D) -> (B, T, D)
    log_melspc_tensor = torch.from_numpy(log_melspc).float().unsqueeze(0)
    
    print(f"  - F0 tensor shape: {f0_tensor.shape}")
    print(f"  - Log Mel-Spectrogram tensor shape: {log_melspc_tensor.shape}")

    # 5. (Optional) Export the convolution part to ONNX
    if onnx_path:
        print(f"Exporting the convolution module to {onnx_path}...")
        # The input for the convs_onnx part is just the mel spectrogram tensor.
        # This tensor will act as a dummy input for tracing the model graph.
        dummy_input = log_melspc_tensor
        
        torch.onnx.export(
            model.convs_onnx,               # The model to export
            dummy_input,                    # Model dummy input
            onnx_path,                      # Where to save the model
            export_params=True,             # Store the trained parameter weights inside the model file
            opset_version=11,               # The ONNX version to export the model to
            do_constant_folding=True,       # Whether to execute constant folding for optimization
            input_names=['log_melspc'],     # The model's input names
            output_names=['ccep_harm', 'ccep_noise'], # The model's output names
            dynamic_axes={
                'log_melspc': {1: 'n_frames'}, # Variable length axis
                'ccep_harm': {1: 'n_frames'},
                'ccep_noise': {1: 'n_frames'}
            }
        )
        print(f"Successfully exported ONNX model to {onnx_path}")

    # 6. Run inference for audio generation
    print("Running model inference to generate audio...")
    with torch.no_grad():
        # The forward pass expects (mel, f0)
        output_wav = model(log_melspc_tensor, f0_tensor)
    print("Inference complete.")

    # 7. Save output audio
    output_wav_np = output_wav.squeeze().cpu().numpy()
    sample_rate = preprocess_cfg['sample_rate']
    
    print(f"Saving output waveform to {output_wav_path} (Sample Rate: {sample_rate} Hz)...")
    sf.write(output_wav_path, output_wav_np, sample_rate)

    print(f"Successfully saved model output to: {output_wav_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check the initial state of the NHVSing model and optionally export to ONNX."
    )
    parser.add_argument(
        "npz_path", 
        type=str, 
        help="Path to the input NPZ file (e.g., 'dataset/npz/savename_01_000.npz')."
    )
    parser.add_argument(
        "output_wav_path", 
        type=str, 
        help="Path to save the output WAV file (e.g., 'debug/output.wav')."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Optional: Path to save the exported ONNX model for the convolution part (e.g., 'debug/nhvsing_convs.onnx')."
    )
    args = parser.parse_args()

    # Create parent directory for the output file if it doesn't exist
    output_dir = Path(args.output_wav_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    main(args.npz_path, args.output_wav_path, args.onnx_path)