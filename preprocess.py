"""NHVSing V3 preprocess — 歌唱 wav ディレクトリ → 学習用 shard npz。

各 segment に対し:
  - mel   : 44.1kHz / hop256 / fft2048 / 128mel(40-16000)/ **ln**(nvSTFT・pc-nsf 互換, center=False + reflect pad)
  - F0    : **RMVPE 単体** + 跳躍除外(下記③④②new)。0=無声で保存(無声補間は学習時 dataset.norm_interp_f0)
  - wav   : target_rms 正規化した segment 波形
出力 npz キー: '<sid>|f0'(float32[T]), '<sid>|log_melspc'(float32[T,128] ln), '<sid>|wav'(float32[T*hop])。

★ F0 は RMVPE のみ(DIO/Praat の合議は使わない)。RMVPE weights(rmvpe.pt ~173MB)は tools/f0 が初回自動DL。

跳躍除外ルール(RMVPE 出力に適用。RMVPE は元々クリーンで稀にしか効かないが安全網):
  ④ 両隣と半オク以上離れた単フレーム → 両隣の線形補間で埋める。
  ③ 有声区間の端点(オンセット/オフセット)3フレーム以内の半オク跳躍 → 端点側を無声化。
  ②new ≤3フレーム無声を挟んだ孤立短run(≤3)が両隣と半オク → run 全体を無声化。

Usage:
    python preprocess.py --indir <wav_dir> --out <npz_dir> --config config_v3.yaml
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

from tools.cut_by_phrases import detect_regions, build_segments
from tools.f0.algorithms.rmvpe import RMVPEPitchAlgorithm

HALF_OCT = 0.5           # 半オクターブ = log2 で 0.5
SILENT_MEL_MAX = -9.2    # ln-mel の無音しきい値(全体がこれ未満なら無音 segment)


def make_mel_fn(cfg):
    SR, NFFT, HOP = cfg['sample_rate'], cfg['fft_size'], cfg['hop_size']
    basis = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=cfg['mel_dim'],
                                fmin=cfg['mel_min'], fmax=cfg['mel_max'])

    def wav_to_mel(y):
        pad = (NFFT - HOP) // 2
        yp = np.pad(y.astype(np.float64), (pad, pad), mode='reflect')
        stft = librosa.stft(yp, n_fft=NFFT, hop_length=HOP, win_length=cfg['win_size'],
                            window='hann', center=False)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            mel = basis @ np.abs(stft)
        return np.log(np.maximum(1e-5, mel)).astype(np.float32)   # [n_mels, T]
    return wav_to_mel


def active_rms(y, thresh_db=-40.0):
    thresh = 10.0 ** (thresh_db / 20.0)
    a = y[np.abs(y) > thresh]
    if len(a) < max(1, int(len(y) * 0.01)):
        return float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.mean(a ** 2)))


# ── 跳躍除外 ────────────────────────────────────────────────────────────────
def _jump(a, b):
    return abs(np.log2(a / b)) >= HALF_OCT


def _segments(mask):
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segs.append((i, j)); i = j + 1
        else:
            i += 1
    return segs


def clean_jumps(f0):
    f0 = f0.copy()
    # ④ 単フレームスパイク
    for i in range(1, len(f0) - 1):
        if f0[i] > 0 and f0[i - 1] > 0 and f0[i + 1] > 0 and _jump(f0[i], f0[i - 1]) and _jump(f0[i], f0[i + 1]):
            f0[i] = 0.5 * (f0[i - 1] + f0[i + 1])
    # ③ 端点の隣接跳躍 → 端点側を無声化
    orig = f0.copy()
    for a, b in _segments(orig > 0):
        for j in range(a + 1, min(a + 4, b + 1)):
            if _jump(orig[j], orig[j - 1]):
                f0[a:j] = 0.0; break
        for j in range(b, max(b - 3, a), -1):
            if _jump(orig[j], orig[j - 1]):
                f0[j:b + 1] = 0.0; break
    # ②new ≤3フレーム無声を挟んだ孤立短run(≤3)が両隣と半オク → run 全体を無声化
    orig = f0.copy()
    runs = _segments(orig > 0)
    for idx, (a, b) in enumerate(runs):
        if b - a + 1 > 3:
            continue
        left = orig[runs[idx - 1][1]] if idx > 0 and a - runs[idx - 1][1] - 1 <= 3 else None
        right = orig[runs[idx + 1][0]] if idx < len(runs) - 1 and runs[idx + 1][0] - b - 1 <= 3 else None
        med = float(np.median(orig[a:b + 1]))
        off_l = left is not None and _jump(med, left)
        off_r = right is not None and _jump(med, right)
        if (off_l and off_r) or (off_l and right is None) or (off_r and left is None):
            f0[a:b + 1] = 0.0
    return f0


def rmvpe_f0(algo, thr, wav, length):
    pitch, voiced, _ = algo.extract_pitch(wav.astype(np.float32), thresholds=thr)
    f0 = np.where(np.asarray(voiced) > 0, np.asarray(pitch), 0.0).astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    return f0[:length]


def main():
    ap = argparse.ArgumentParser(description='NHVSing V3 preprocess (RMVPE F0)')
    ap.add_argument('--indir', required=True, help='歌唱 wav の入ったディレクトリ(再帰探索)')
    ap.add_argument('--out', required=True, help='shard npz の出力先')
    ap.add_argument('--config', default='config_v3.yaml')
    ap.add_argument('--segs_per_shard', type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))['preprocess']
    SR, HOP = cfg['sample_rate'], cfg['hop_size']
    TARGET_RMS = cfg['target_rms']
    MIN_FRAMES = cfg['data_filtering']['min_frames']
    cw = cfg['cut_wavs']
    wav_to_mel = make_mel_fn(cfg)
    algo = RMVPEPitchAlgorithm(sample_rate=SR, hop_size=HOP, fmin=cfg['f0_min'], fmax=cfg['f0_max'])
    thr = algo._get_default_threshold()
    os.makedirs(args.out, exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(args.indir, '**', '*.wav'), recursive=True))
    print(f'{len(wavs)} wav files, RMVPE threshold={thr}, target_rms={TARGET_RMS}')
    shard, shard_i, nseg, nskip = {}, 0, 0, 0

    def flush():
        nonlocal shard, shard_i
        if shard:
            np.savez_compressed(os.path.join(args.out, f'shard-{shard_i:04d}.npz'), **shard)
            shard = {}; shard_i += 1

    for wp in tqdm(wavs):
        y, sr0 = sf.read(wp)
        if y.ndim > 1:
            y = y.mean(1)
        y = y.astype(np.float32)
        if sr0 != SR:
            y = librosa.resample(y.astype(np.float64), orig_sr=sr0, target_sr=SR).astype(np.float32)
        regions = detect_regions(y, SR, silence_thresh_db=cw['silence_thresh'],
                                 min_silence_dur=cw['min_silence_dur'])
        segs = build_segments(regions, SR, max_dur=cw['max_dur'], long_silence=cw['long_silence'],
                              pad=cw['pad'], total_samples=len(y))
        stem = os.path.splitext(os.path.basename(wp))[0]
        for idx, (s, e) in enumerate(segs):
            yseg = y[s:e].copy()
            rms = active_rms(yseg.astype(np.float64))
            if rms > 1e-4:
                yseg = yseg * (TARGET_RMS / rms)
            yseg = np.clip(yseg, -1.0, 1.0).astype(np.float32)
            mel = wav_to_mel(yseg)                                   # [n_mels, T]
            if mel.shape[1] < MIN_FRAMES or mel.max() < SILENT_MEL_MAX:
                nskip += 1; continue
            f0 = clean_jumps(rmvpe_f0(algo, thr, yseg, mel.shape[1]))
            T = min(mel.shape[1], len(f0))
            if T < MIN_FRAMES:
                nskip += 1; continue
            ywav = yseg[:T * HOP]
            if len(ywav) < T * HOP:
                ywav = np.pad(ywav, (0, T * HOP - len(ywav)))
            sid = f'{stem}_{idx:04d}'
            shard[f'{sid}|f0'] = f0[:T].astype(np.float32)
            shard[f'{sid}|log_melspc'] = mel[:, :T].T.astype(np.float32)   # [T, n_mels] ln
            shard[f'{sid}|wav'] = ywav.astype(np.float32)
            nseg += 1
            if nseg % args.segs_per_shard == 0:
                flush()
    flush()
    print(f'done: {nseg} segments, {nskip} skipped -> {args.out}')


if __name__ == '__main__':
    main()
