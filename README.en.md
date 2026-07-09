[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf) model **tuned for singing voice synthesis**. Implemented in PyTorch with support for JIT compilation and single-file ONNX export.

This repository contains the latest **NHVSing V3 / V3X** (recommended), plus the legacy **NHVSing** (V1, single-speaker) and **NHVSingV2** (multi-speaker).

🎧 **Listen → [NHVSing V3 Demo Page](https://wavtechyukky.github.io/NHVSing/v3.html)** (V3 / V3X vs NSF-HiFiGAN comparison, RTF & model-size figures, singing synthesis from DiffSinger acoustic models)

***

## NHVSing V3 / V3X (latest, recommended)

**V3** is the latest model, refined for singing (44.1kHz / hop256 / 128-mel [40–16000Hz, **ln**]). Three essential improvements over V2:

- **Multi-Resolution Discriminator (MRD)** (+ MPD): the single biggest factor in quality. UnivNet-style multi-resolution spectrogram-magnitude discrimination.
- **Faster impulse-response synthesis (`fft_corr`)**: the LTV-FIR time-correlation is computed via FFT — bit-equivalent to the time-domain version, ~8× faster on CPU.
- **Training-data curation & augmentation**: gathered high-quality 44.1 kHz audio and, during training, fed it while randomly varying volume and pitch. Although this feeds the model waveforms that deviate from ordinary ones, it significantly improved the vocoder's generalization.
- **Cleaner excitation/conditioning**: many measures were tried to raise quality, but the two above had the largest effect, so the configuration was reverted to follow the original paper for now (a major restructuring would likely be V4 or later). White 200-harmonic source / quef_norm α=1.0 / **mel-only input** (F0 embedding removed) / linear F0 interpolation (a return to the original NHV).

**V3X** lets V3 run on **hop512 input**: it takes the hop512 mel/F0 emitted by e.g. OpenUtau, interpolates internally to the hop256 grid, then runs V3 (weights/state_dict shared with V3). This **resolves the "hop512 severely degrades quality" limitation** of V1/V2.

| Class | Purpose | config |
|---|---|---|
| `NHVSing` | V1 legacy (single-speaker) | `config.yaml` |
| `NHVSingV2` | V2 legacy (multi-speaker) | `config_v2.yaml` |
| **`NHVSingV3`** | **final model (hop256 native)** | **`config_v3.yaml`** |
| **`NHVSingV3X`** | **hop512-input variant of V3** | `config_v3.yaml` + `ltv_filter.use_v3x: true` |

### Performance (V3)

**Model size**: `nhv_v3.onnx` is about **2.2 MB** (no quantization) — roughly **1/26** the size of NSF-HiFiGAN (pc-nsf-hifigan, 56.7 MB), the reference we compare against.

**RTF** (Real-Time Factor = seconds of compute per second of audio; lower is faster, and < 1 means faster than real time). Measured on an Apple 10-core CPU / ~5 s input / batch 1 / median of 9 runs.

Under ONNX Runtime (CPU), side by side with NSF-HiFiGAN:

| CPU threads | NHVSing V3 | NSF-HiFiGAN | NHVSing speed-up |
|---|---|---|---|
| 1 | 0.076 (13×) | 0.604 (2×) | **8.0×** |
| 2 | 0.065 (15×) | 0.316 (3×) | 4.9× |
| 4 | 0.062 (16×) | 0.201 (5×) | 3.3× |
| 8 | 0.063 (16×) | 0.198 (5×) | 3.1× |

**The two scale very differently with core count.** NHVSing V3's per-frame impulse-response generation is fully independent (*embarrassingly parallel*), but **ONNX Runtime barely exploits this**, so V3 is essentially single-core-bound (13×→16× and then flat). NSF-HiFiGAN's large transposed convolutions parallelize well, so it keeps speeding up with more cores (2×→5×). As a result, **NHVSing's speed lead is largest on low-core devices (~8×) and narrows to ~3× on many cores**, but it stays ahead throughout.

**Native PyTorch does realize the parallelism.** Running the same V3 in torch scales with core count as the per-frame independence allows:

| CPU threads | ONNX Runtime | PyTorch |
|---|---|---|
| 1 | 0.076 (13×) | 0.082 (12×) |
| 2 | 0.065 (15×) | 0.050 (20×) |
| 4 | 0.062 (16×) | 0.043 (23×) |
| 8 | 0.063 (16×) | **0.031 (32×)** |

On multiple cores, **torch actually beats our own ONNX export** (~2× at 8 threads): for such a tiny, FFT-dominated model, torch's batched-FFT parallelism helps more than ORT's graph optimizations. So "NHVSing is fast/slow" cannot be captured by a single number — it depends on the **runtime × core-count** combination.

> RTF depends only on the amount of compute, not on the weight values (it is the same for any checkpoint).

The ONNX graph pads the `LTVFirONNX` FFT length to a **power of two** for speed (ONNX Runtime's DFT is fast only for power-of-two sizes); V3X is comparable. The LTV-FIR time-correlation itself is also FFT-based via `fft_corr` (bit-equivalent to the time-domain version, ~8× faster on CPU — see "Key changes" above).

### Usage (V3)

**Preprocess** (F0 = RMVPE only + jump cleaning; `rmvpe.pt` auto-downloads on first run):
```bash
python preprocess.py --indir <dir_of_singing_wavs> --out <npz_dir> --config config_v3.yaml
```

> **Splitting train / test**: `preprocess.py` just writes all shards to `--out`; it does **not** auto-split into train/test. **Run it twice on separate wav sets** (train vs. eval) and point each to its own directory (a few held-out songs are enough for test):
> ```bash
> python preprocess.py --indir wavs/train --out dataset/train --config config_v3.yaml
> python preprocess.py --indir wavs/eval  --out dataset/test  --config config_v3.yaml
> ```
> Set `training.train_dir` / `test_dir` in `config_v3.yaml` accordingly. `VocoderDataset` recursively reads both shard npz (`<sid>|f0` / `|log_melspc` / `|wav`) and single-segment npz, so either layout works.

**Train** (MRD + MPD GAN; set `training.train_dir` / `test_dir` / `snapshot_dir` in `config_v3.yaml`):
```bash
python train_v3.py --config config_v3.yaml
```

**ONNX export** (both V3 and V3X; 3 outputs: `waveform` / `harmonic` / `noise`):
```bash
python export.py --config config_v3.yaml --ckpt <weights.ckpt> --out exported_models
```
By default this writes to `exported_models/v3/`:

- **`nhv_v3.pth`** — shared weights for V3/V3X (load with `NHVSingV3(vc, lc).load_state_dict(torch.load('nhv_v3.pth'))`). **V3X shares these weights, so it has no `.pth`** (ONNX only).
- **`nhv_v3.onnx` / `nhv_v3x.onnx`** — each a **single self-contained ONNX** (NN + DSP, weights embedded — no external `.onnx.data`; runnable with ONNX Runtime alone, same format as V1/V2's `full_vocoder.onnx`). Inputs `mel` / `f0` / `uv` → 3 outputs `waveform` / `harmonic` / `noise`, with `clamp(harmonic + noise) == waveform`. The time length T is dynamic (any length).

**Inference (Python)**:
```python
from nhv_vocoder import NHVVocoder
voc = NHVVocoder('weights.ckpt', 'config_v3.yaml')      # V3/V3X auto-selected via config use_v3x
cf0, uv = NHVVocoder.prep_f0(f0_hz)                       # raw F0 (0=unvoiced) → continuous F0 + uv
wav = voc.infer(mel, cf0, uv)                            # mel: [T, 128] ln-mel
```

### F0 extraction

Preprocessing F0 uses **RMVPE only + jump-cleaning post-processing** (`tools/f0`). The RMVPE weights `rmvpe.pt` (~173MB) are **not** bundled — they **auto-download from HuggingFace on first run**.

### Weight licensing

Distributed trained weights are **non-commercial** (due to the training data). Check each dataset's license directly at its original source (listed under *Singing Voice Databases Used* below).

***

# Everything below is about the legacy versions (NHVSing V1 / NHVSingV2)

> ⚠️ **The sections below (Audio Samples, Architecture, Performance, Usage, etc.) all describe the older V1 / V2 models.** For new work, use **V3 / V3X (recommended)** above. V1 / V2 are kept for compatibility.

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

## Performance (V1 / V2)

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

## Architecture (V1 / V2)

The following changes have been made from the original paper's implementation. (V3's discriminator is MRD + MPD, which differs from the MSD + complex-STFT setup described below.)

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

## Usage (V1 / V2)

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
*   **Frame size:** V1/V2 degrade significantly at hop_size=512. **V3X handles hop512 input** (interpolating internally to hop256), resolving this.
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

### V3 (distributed weights, non-commercial)

The distributed V3 weights are trained on the following non-commercial data. **Please check each dataset's license and terms directly at its original source** (Kiritan / No.7 require SNS account authentication; Opencpop may require contacting the rights holder):

*   Tohoku Kiritan — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   Natsume Yuri — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   Namine Ritsu (Ritsu Singing DB Ver2.0-2.2 / Soft) — [Canon Voice](https://www.canon-voice.com/voicebanks/)
*   Children's Song Dataset (CSD) — [Zenodo](https://zenodo.org/records/4916302)
*   NUS-48E — [Zenodo](https://zenodo.org/records/19595152)
*   ONIKU_KURUMI Utagoe DB — [Onikuru](https://onikuru.info/db-download/)
*   Opencpop — [WeNet Opencpop](https://wenet-e2e.github.io/opencpop/download/)
*   No.7 — [VOICE SEVEN](https://voiceseven.com/7dev/login.php)
*   VocalSet — [Zenodo](https://zenodo.org/records/1193957)
*   ccmusic-database / acapella — [HuggingFace](https://huggingface.co/datasets/ccmusic-database/acapella)

### V1 / V2 (legacy)

*   Tohoku Kiritan — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   Natsume Yuri — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   M4Singer (CC BY-NC-SA 4.0) — [M4Singer GitHub](https://github.com/M4Singer/M4Singer)
    *   Zhang et al., "M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus," *NeurIPS 2022*.
*   ACE-Opencpop — [HuggingFace](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
