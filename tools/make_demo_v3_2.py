"""Build the NHVSing V3.2 demo page (docs/v3_2.html + docs/audio_v3_2/).

Copy-synthesis of every ``docs/audio_v3/<stem>_gt.wav`` with the released V3.2
weights, following the exact V3/V3.1-demo methodology:

  * features = preprocess-identical diffsinger ln-mel (hop 256) + RMVPE F0
    (+ ``clean_jumps``), RMS-normalized to ``preprocess.target_rms``
  * V3.2  : hop-256 features as-is (Hann WOLA in the LTV filter)
  * V3.2X : the same features sub-sampled to the hop-512 grid (``[::2]``, the
    grid OpenUtau/DiffSinger produce) and upsampled back internally — same as
    the V3X column of the V3 demo
  * mel-L1 = mean |ln-mel(resynth) − ln-mel(GT)|, recomputed here for V3.2,
    V3.2X **and** the existing NSF-HiFiGAN wavs so all numbers on the page
    share one measurement

Writes ``docs/audio_v3_2/<stem>_v32{,x}.wav`` + ``_mel.png``, then renders the
whole ``docs/v3_2.html``. GT / NSF audio and GT mel images are reused from
``docs/audio_v3/`` (unchanged assets).

Usage (from the repo root; RMVPE weights auto-download on first run):
    python tools/make_demo_v3_2.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import make_mel_fn, active_rms, clean_jumps
from nhv_vocoder import NHVVocoder

AUDIO_V3 = _REPO / "docs" / "audio_v3"
OUT_DIR = _REPO / "docs" / "audio_v3_2"
PAGE = _REPO / "docs" / "v3_2.html"
CFG = _REPO / "config_v3_2.yaml"       # ola_mode: hann -> Hann WOLA in the eager model
PTH = _REPO / "exported_models/v3_2/nhv_v3_2.pth"
SEED = 0


def save_mel_png(mel_TD: np.ndarray, path: Path, vmin=None, vmax=None):
    """600x280 mel image, same look as the V3 demo assets.

    vmin/vmax should come from the clip's GT mel: with per-image autoscaling the
    vocoder outputs render darker overall, because their silence noise floor is
    lower than the GT's room noise and stretches the color range."""
    fig = plt.figure(figsize=(6.0, 2.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mel_TD.T, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = yaml.safe_load(open(CFG))
    p_cfg = cfg["preprocess"]
    sr = p_cfg["sample_rate"]
    mel_fn = make_mel_fn(p_cfg)

    # F0: RMVPE at the training hop (single instance, single process)
    from tools.f0.algorithms.rmvpe import RMVPEPitchAlgorithm
    algo = RMVPEPitchAlgorithm(sample_rate=sr, hop_size=p_cfg["hop_size"],
                               fmin=p_cfg["f0_min"], fmax=p_cfg["f0_max"], device="cpu")

    voc = NHVVocoder(str(PTH), str(CFG))
    cfg_x = yaml.safe_load(open(CFG))
    cfg_x["model"]["ltv_filter"]["use_v3x"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg_x, f)
        cfg_x_path = f.name
    voc_x = NHVVocoder(str(PTH), cfg_x_path)
    os.unlink(cfg_x_path)

    OUT_DIR.mkdir(exist_ok=True)

    def feats(y):
        """wav -> (mel[T,128], cf0[T], uv[T]) with preprocess-identical settings."""
        rms = active_rms(y.astype(np.float32))
        if rms > 1e-6:
            y = np.clip(y * (p_cfg["target_rms"] / rms), -1.0, 1.0)
        mel = mel_fn(y).T                                       # [T, mel_dim]
        y_n = y.astype(np.float32)
        m = np.abs(y_n).max()
        if m > 1e-6:
            y_n = np.clip(y_n / m, -1.0, 1.0)
        pitch, voiced, *_ = algo.extract_pitch(y_n)
        f0 = np.where(np.asarray(voiced) > 0, np.asarray(pitch), 0.0).astype(np.float32)
        f0 = clean_jumps(f0)
        T = min(len(mel), len(f0))
        cf0, uv = NHVVocoder.prep_f0(f0[:T])
        return mel[:T], cf0, uv

    def mel_of_wav(y):
        """Loudness-aligned ln-mel for the metric: every wav (GT / resynth / NSF)
        is first normalized to the same active RMS, so mel-L1 measures spectral
        distance, not gain offsets (stored wavs differ in loudness)."""
        rms = active_rms(y.astype(np.float32))
        if rms > 1e-6:
            y = np.clip(y * (p_cfg["target_rms"] / rms), -1.0, 1.0)
        return mel_fn(y).T

    def mel_l1(a, b):
        T = min(len(a), len(b))
        return float(np.mean(np.abs(a[:T] - b[:T])))

    rows = []
    for gt_path in sorted(AUDIO_V3.glob("*_gt.wav")):
        stem = gt_path.name[:-len("_gt.wav")]
        y, rd_sr = sf.read(gt_path)
        if y.ndim == 2:
            y = y.mean(axis=1)
        assert rd_sr == sr, f"{stem}: {rd_sr} != {sr}"
        mel, cf0, uv = feats(y)

        wav32 = voc.infer(mel, cf0, uv, seed=SEED)
        wav32x = voc_x.infer(mel[::2], cf0[::2], uv[::2], seed=SEED)

        # loudness-match the saved wavs to the GT clip (the model outputs at its
        # training RMS ~0.111, much quieter than the stored GT/NSF wavs)
        def match_loudness(w, ref_rms):
            r = active_rms(w.astype(np.float32))
            if r > 1e-6:
                w = w * (ref_rms / r)
            peak = float(np.abs(w).max())
            if peak > 0.98:
                w = w * (0.98 / peak)
            return w.astype(np.float32)

        rms_gt = active_rms(y.astype(np.float32))
        wav32 = match_loudness(wav32, rms_gt)
        wav32x = match_loudness(wav32x, rms_gt)
        sf.write(OUT_DIR / f"{stem}_v32.wav", wav32, sr)
        sf.write(OUT_DIR / f"{stem}_v32x.wav", wav32x, sr)
        y_nsf, _ = sf.read(AUDIO_V3 / f"{stem}_nsf.wav")
        if y_nsf.ndim == 2:
            y_nsf = y_nsf.mean(axis=1)
        mel32, mel32x = mel_of_wav(wav32), mel_of_wav(wav32x)
        mel_gt, mel_nsf = mel_of_wav(y), mel_of_wav(y_nsf)
        # one renderer + one per-clip color range for ALL four columns, so the
        # images are visually comparable (mixing renderers/autoscale makes the
        # vocoder columns look darker than GT)
        vmin, vmax = float(mel_gt.min()), float(mel_gt.max())
        save_mel_png(mel_gt, OUT_DIR / f"{stem}_gt_mel.png", vmin, vmax)
        save_mel_png(mel32, OUT_DIR / f"{stem}_v32_mel.png", vmin, vmax)
        save_mel_png(mel32x, OUT_DIR / f"{stem}_v32x_mel.png", vmin, vmax)
        save_mel_png(mel_nsf, OUT_DIR / f"{stem}_nsf_mel.png", vmin, vmax)
        m32, m32x = mel_l1(mel32, mel_gt), mel_l1(mel32x, mel_gt)
        mnsf = mel_l1(mel_nsf, mel_gt)
        rows.append((stem, m32, m32x, mnsf))
        print(f"{stem}: mel-L1 v3.2={m32:.3f} v3.2x={m32x:.3f} nsf={mnsf:.3f}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({s: {"v32": a, "v32x": b, "nsf": c} for s, a, b, c in rows}, f, indent=1)

    render_page(rows)
    print(f"wrote {PAGE}")


def render_page(rows):
    def cell(audio, img, l1=None):
        m = f'<br><span class="m">mel-L1 {l1:.3f}</span>' if l1 is not None else ""
        return (f'<td><audio controls preload="none" src="{audio}"></audio>'
                f'<br><img class="mel" src="{img}">{m}</td>')

    trs = []
    for stem, m32, m32x, mnsf in rows:
        trs.append(
            f"<tr><td>{stem}</td>"
            + cell(f"audio_v3/{stem}_gt.wav", f"audio_v3_2/{stem}_gt_mel.png")
            + cell(f"audio_v3_2/{stem}_v32.wav", f"audio_v3_2/{stem}_v32_mel.png", m32)
            + cell(f"audio_v3_2/{stem}_v32x.wav", f"audio_v3_2/{stem}_v32x_mel.png", m32x)
            + cell(f"audio_v3/{stem}_nsf.wav", f"audio_v3_2/{stem}_nsf_mel.png", mnsf)
            + "</tr>")
    table_rows = "\n".join(trs)

    style = (_REPO / "docs" / "v3.html").read_text(encoding="utf-8")
    style = style[style.index("<style>"):style.index("</style>") + len("</style>")]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHVSing V3.2 Demo</title>
  {style}
</head>
<body>

<header>
  <h1>NHVSing V3.2 Demo</h1>
  <p>V3.2 = V3.1 with the high-pitch <b>1-period dropout</b> fixed. The square overlap-add in the LTV filter let a frame's filtered spillover cancel a whole glottal period in anti-phase at high F0; V3.2 replaces it with a <b>Hann WOLA</b> (50% overlap) that tapers the spillover, and the ONNX export matches. Training also randomises the excitation start phase per batch and uses a higher-resolution multi-resolution STFT loss.</p>
  <p>
    <a href="https://github.com/wavtechyukky/NHVSing" target="_blank">GitHub Repository</a>
    &nbsp;·&nbsp;
    <a href="v3.html">← NHVSing V3 Demo (RTF tables &amp; DiffSinger synthesis)</a>
  </p>
</header>

<main>

  <div class="intro">Copy-synthesis of the same held-out validation clips as the <a href="v3.html">V3 demo</a>, re-synthesized with the released <b>V3.2</b> weights (<code>exported_models/v3_2/</code>). <b>V3.2X receives the mel sub-sampled to the hop-512 grid</b> (the grid OpenUtau / DiffSinger produce) and upsamples it back internally, exactly like the V3X column of the V3 demo. mel-L1 (mean |Δln-mel| against the ground-truth mel; lower = closer) is recomputed here for all three synthesized columns with one measurement, so the numbers are directly comparable within this page — but not to the V3 page. Note that the dropout fix is a phase/periodicity change that mel-L1 (a magnitude metric) barely sees; judge it by listening to the high notes. Architecture, model size and RTF are unchanged from V3 (see the <a href="v3.html">V3 page</a> for the efficiency tables).</div>

  <section>
    <h2>Copy-synthesis (NHVSing V3.2 / V3.2X vs pc-nsf-hifigan)</h2>
    <p class="section-note">One validation clip per dataset, none included in the training data. Judge by listening — mel-L1 does not match perception.</p>
    <table><tr><th>segment</th><th>Ground Truth</th><th>NHVSing V3.2</th><th>NHVSing V3.2X</th><th>NSF-HiFiGAN</th></tr>
{table_rows}
</table>
  </section>

</main>

<footer>
  NHVSing V3.2 &nbsp;·&nbsp; <a href="https://github.com/wavtechyukky/NHVSing" target="_blank">GitHub</a>
</footer>

</body>
</html>
"""
    PAGE.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
