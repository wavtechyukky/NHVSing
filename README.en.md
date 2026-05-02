[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf) model **tuned for singing voice synthesis**. Implemented in PyTorch with support for JIT compilation and single-file ONNX export.

This repository contains two model classes: **NHVSing** (single-speaker model) and **NHVSingV2** (multi-speaker general-purpose model).

***

## Audio Samples

→ **[NHVSingV2 Demo Page](https://wavtechyukky.github.io/NHVSing/v2.html)**

→ [NHVSing (V1) Demo Page](https://wavtechyukky.github.io/NHVSing/)

Compare synthesized audio from Kiritan & Natsume Yuri (NHVSing) and M4Singer / GTSinger evaluation data (NHVSingV2).

***

## NHVSing vs NHVSingV2

| | NHVSing | NHVSingV2 |
|---|---|---|
| CNN backbone | Dual-branch (Harmonic/Noise independent) | Shared trunk CNN |
| F0 input | Mel spectrogram only | F0 embedder (256 bins, log₂ scale, 128-dim) concatenated with mel |
| quef_norm | Off (was found to inhibit high-freq learning in V1) | Soft scaling with alpha=0.3 (stabilizes training without sacrificing high frequencies) |
| Amplitude augmentation | None | 0.5–2.0× (log-uniform) random scale |
| Speaker generalization | **Single-speaker specialized.** Artifacts occur easily for speakers outside training data | **Multi-speaker capable.** Can be trained on multi-speaker corpora such as M4Singer and ACE-Opencpop |
| Config file | `config.yaml` | `config_v2.yaml` |

NHVSing excels at faithfully reproducing the voice quality of a single speaker trained on a speaker-specific dataset. NHVSingV2 can vocoder acoustic features of various speakers at high quality by training on multi-speaker corpora.

***

## Performance

Measured under the following environment and conditions.

- **Measurement Environment:** Apple M-series CPU (MacBook)
- **Measurement Conditions:** 44.1kHz input, approx. 26 seconds, batch size 1

### NHVSing (V1)

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 2.048 sec           | 0.0788   |
| JIT Script     | CPU    | 2.145 sec           | 0.0825   |
| Unified ONNX   | CPU    | 4.275 sec           | 0.1645   |

### NHVSingV2

| Model Type     | Device | Avg. Inference Time | RTF      |
|----------------|--------|---------------------|----------|
| Native Python  | CPU    | 1.920 sec           | 0.0739   |
| JIT Script     | CPU    | 1.991 sec           | 0.0766   |
| Unified ONNX   | CPU    | 4.043 sec           | 0.1556   |

***

## Architecture

The following changes have been made from the original paper's implementation.

* **Sampling Rate**: Supports **44.1kHz**.
* **Complex Cepstrum**: Dimensions expanded to **512**.
* **Removal of FIR (postfilter)**: Although it reduces STFT loss, it was judged not to contribute to waveform learning.
* **Discriminator**: Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminator. An adversarial loss warmup period allows fine-tuning from mid-training (`adversarial_warmup_epochs`).
* **Additional Loss Functions**:
    * **Envelope loss**: Extracts upper and lower envelopes via 1D max-pooling and computes MAE (RefineGAN §2.5.1). Suppresses amplitude envelope instability (`envelope_scale`).
    * **Harmonic penalty loss**: L1 penalty when voiced components (`sig_harm`) are output in unvoiced regions (frames where F0=0). Suppresses buzzing in unvoiced regions (`harmonic_penalty_scale`).
* **F0 Input**: Takes **linearly interpolated F0** in unvoiced regions as input, eliminating the need for an Unvoiced/Voiced flag. In singing voice synthesis, drawing the F0 curve through unvoiced regions is important, and UV flag-based switching cannot reproduce smooth unvoiced→voiced transitions.
* **Log Mel Spectrogram**: Takes the full band from **40Hz to 22050Hz** as input. High-frequency reproduction was judged to contribute to intuitive quality improvement.

### NHVSingV2-Specific Changes

* **Shared trunk CNN** (`use_shared_trunk: true`): Both harmonic and noise branches share a common trunk CNN before splitting into their respective heads. This eliminates the need for each branch to independently learn how to decompose the input acoustic features into harmonic and noise components.
* **F0 Embedder** (`use_f0_embed: true`): Continuous F0 is discretized into 256 bins on a log₂ scale and embedded into 128 dimensions, then concatenated with the mel spectrogram. Providing explicit pitch information gives the network additional cues about the waveform shape it needs to generate per period.
* **quef_norm** (`use_quef_norm: true`, `quef_norm_alpha: 0.3`): Applies gentle 1/|n|^α scaling to quefrency components, stabilizing training without over-suppressing high harmonics. In V1, enabling this inhibited high-frequency learning, but in V2 a small alpha of 0.3 allows stabilization without sacrificing high frequencies.

### Export Formats

*  **PyTorch Native** (`model.pth`): Same as the model used during training.
*  **TorchScript** (`model_jit.pt`): Executable from other languages via JIT compilation.
*  **Unified ONNX** (`full_vocoder.onnx`): Exports the entire vocoder (NN + DSP) as a single ONNX file. Inference possible with ONNXRuntime only.

***

## Environment

* Verified on Python 3.10

```bash
pip install -r requirements.txt
```

***

## Usage

### NHVSing (V1)

#### 1. Preprocessing

```bash
# Run WAV trimming → F0/mel extraction → train/test split all at once
python preprocess.py --config config.yaml --step all
```

#### 2. Training

```bash
python train.py --config config.yaml
```

#### 3. Fine-tuning (Transfer Learning to a New Speaker)

Inherits only the model weights from a trained model and re-trains on a different speaker's dataset. The Discriminator and Optimizers are freshly initialized.

```bash
python prepare_finetune.py \
  --weights exported_models/natsume/model.pth \
  --config config.yaml \
  --output snapshots_kiritan/000000epoch.pth

python train.py --resume_path snapshots_kiritan/000000epoch.pth --config config_fine_tuning.yaml
```

#### 4. Exporting

```bash
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models/kiritan \
  --all
```

#### 5. Inference

```bash
python inference.py input.wav \
  --snapshot exported_models/kiritan/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/kiritan/full_vocoder.onnx
```

---

### NHVSingV2

Use `config_v2.yaml` as the starting point. Edit paths, speaker prefixes, and training parameters to match your environment.

#### 1. Preprocessing

F0 extraction uses RMVPE (model is auto-downloaded on first run).

```bash
python preprocess.py --config config_v2.yaml --step all
```

#### 2. Training

```bash
python train.py --config config_v2.yaml
```

#### 3. Fine-tuning (Transfer Learning to a New Speaker / Corpus)

```bash
python prepare_finetune.py \
  --weights exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output snapshots_finetune/000000epoch.pth

python train.py --resume_path snapshots_finetune/000000epoch.pth --config config_v2.yaml
```

#### 4. Exporting

```bash
python export.py \
  --checkpoint snapshots_v2/000900epoch.pth \
  --config config_v2.yaml \
  --output_dir exported_models/v2 \
  --all
```

#### 5. Inference

```bash
python inference.py input.wav \
  --snapshot exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output_dir output \
  --onnx exported_models/v2/full_vocoder.onnx
```

For WAV input, if `target_rms: 0.083` is set in `config_v2.yaml`, the RMS of voiced regions is automatically normalized to the median of M4Singer training data.

***

## Training Best Practices

### Amplitude Augmentation (`amp_augment`)

NHVSingV2 introduces amplitude augmentation during training, randomly scaling volume by 0.5–2.0× (log-uniform) (`amp_augment: true`, `amp_aug_range: [0.5, 2.0]`). This makes the model robust to input volume variation, so quality does not degrade even if the volume is slightly off during inference.

### Quefrency Norm Scale (`quef_norm_alpha`)

Setting `use_quef_norm: true` normalizes quefrency components and improves training stability. However, too large an alpha degrades reproducibility of high-frequency bands (consonants, fricatives). `quef_norm_alpha: 0.3` is a reasonable value that stabilizes training without sacrificing high frequencies.

### Harmonic Penalty Loss Strength (`harmonic_penalty_scale`)

This penalty suppresses harmonic components from leaking in unvoiced regions (frames where F0=0). This value has a large impact on quality.

* Too small (0–10): Buzzing sounds tend to occur in unvoiced regions.
* Too large (1000+): When given acoustic features that are ambiguous between voiced and unvoiced, the model may attempt to reproduce the mel spectrogram using unvoiced components even in regions that should be voiced.
* **Recommended: `harmonic_penalty_scale: 100`**

***

## Known Issues

*   **Training Process:** Multi-GPU training is not supported.
*   **Other Frame Sizes:** Quality degrades significantly with hop_size=512.
*   **NHVSing (V1) Speaker Dependency:** The generated waveform strongly reflects the characteristics of the trained speaker. Use NHVSingV2 if multi-speaker support is required.

***

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This repository is based on the following papers and repositories published by Liu, et al.

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)

## Singing Voice Databases Used

*   Tohoku Kiritan — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   Natsume Yuri — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   M4Singer (CC BY-NC-SA 4.0) — [M4Singer GitHub](https://github.com/M4Singer/M4Singer)
    *   Zhang et al., "M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus," *NeurIPS 2022*.
*   ACE-Opencpop — [HuggingFace](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
