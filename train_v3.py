import os
import argparse
import time
import random
from tqdm import tqdm
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import soundfile as sf
import librosa
import torchaudio.functional as taF

from dataset import VocoderDataset, collate_fn_padd, norm_interp_f0
from model import NHVSing, NHVSingV2, NHVSingV3, NHVSingV3X, select_model_class
from discriminator import DiscriminatorWithComplexSTFT
from dsp import stft_loss as stft_loss_fn, envelope_loss as envelope_loss_fn, mel_loss as mel_loss_fn, wav_to_mel_torch
import glob


def pitch_augment_batch(f0, log_melspc, wav, uv, mask, cfg,
                        mel_sr, mel_nfft, mel_hop, mel_win, mel_dim, mel_fmin, mel_fmax):
    """RefineGAN流 pitch augmentation(per-batch, GPU on-the-fly)。確率 pitch_aug_prob で
    バッチ全体を単一 r=2^(ζ/12) でピッチシフト。リサンプルで 新F0=旧F0×r(RMVPE不要)、mel は
    リサンプル波形から wav_to_mel_torch で再計算(preprocess 一致)。既定OFF or 非発動で現行と完全一致。
    入力/返り: f0[B,1,Tf], log_melspc[B,Tf,M], wav[B,Ts], uv[B,1,Tf], mask[B,Tf](True=pad)。
    ※ ピッチダウン(ζ<0)=クロップが伸びるので多res STFT窓に安全。アップは短縮→レンジ上限を控えめに。"""
    tcfg = cfg['training']
    if (not tcfg.get('pitch_aug', False)) or random.random() >= tcfg.get('pitch_aug_prob', 0.0):
        return f0, log_melspc, wav, uv, mask
    zmin, zmax = tcfg.get('pitch_aug_semitones', [-7, 2])
    z = random.uniform(float(zmin), float(zmax))
    r = 2.0 ** (z / 12.0)
    if abs(z) < 1e-3:
        return f0, log_melspc, wav, uv, mask
    # 1) wav リサンプル(ratio = new/orig = 1/r → 出力長 ≈ Ts/r、バッチ一様)。
    #    base=100 = cent級・小整数比でカーネル小=高速。ζ∈[-7,2]で ratio≈1近傍。
    base = 100
    new_wav = taF.resample(wav, orig_freq=int(round(base * r)), new_freq=base)  # [B, ~Ts/r]
    Tf_new = new_wav.shape[-1] // mel_hop
    if Tf_new < 8:                                      # 短すぎガード(通常発生しない)
        return f0, log_melspc, wav, uv, mask
    new_wav = new_wav[..., :Tf_new * mel_hop]
    # 2) mel 再計算(GPU, preprocess 一致)。[B,M,Tf_new] -> [B,Tf_new,M]
    new_mel = wav_to_mel_torch(new_wav, mel_sr, mel_nfft, mel_hop, mel_win,
                               mel_dim, mel_fmin, mel_fmax).transpose(1, 2)

    # f0/uv は frame 解像度([B,1,Tf])→ Tf_new フレームへ。mask は sample 解像度
    # ([B,Ts], get_mask_from_lengths=wav長)→ Ts_new サンプルへ(別解像度)。
    def _warp(x3, size, mode):
        kw = {'align_corners': False} if mode == 'linear' else {}
        return torch.nn.functional.interpolate(x3, size=size, mode=mode, **kw)
    new_f0 = _warp(f0.float(), Tf_new, 'linear') * r    # 3) 時間ワープ + ×r(周波数 r倍)
    new_uv = (_warp(uv.float(), Tf_new, 'nearest') > 0.5).float()
    Ts_new = new_wav.shape[-1]                           # = Tf_new * mel_hop
    new_mask = (_warp(mask.float().unsqueeze(1), Ts_new, 'nearest') > 0.5).squeeze(1) \
        if mask is not None else None
    return new_f0, new_mel, new_wav, new_uv, new_mask


def make_random_crop_collate(crop_frames, hop_size: int):
    """NSF-HiFiGAN 流の固定長ランダムクロップ。各 item を crop_frames フレーム
    (= crop_frames*hop_size sample)に frame0 整列でランダム切り出し。短い item は
    末尾ゼロパディング。全 item 同一長になるので collate_fn_padd は無パディング=mask 全 False
    (下流の RMS/masked_fill は no-op)。head-crop だった make_capped_collate を置換。
    __getitem__ 返り値: f0[1,T], mel[T,n_mels], wav[T*hop], uv[1,T]。"""
    if crop_frames is None:
        return collate_fn_padd

    crop_samples = crop_frames * hop_size

    def random_crop_collate(batch):
        cropped = []
        for (f0, melspc, wav, uv) in batch:
            T = melspc.shape[0]
            if T < crop_frames:
                pad_f = crop_frames - T
                f0 = np.pad(f0, ((0, 0), (0, pad_f)))
                uv = np.pad(uv, ((0, 0), (0, pad_f)))
                melspc = np.pad(melspc, ((0, pad_f), (0, 0)))
                wav = np.pad(wav, (0, crop_samples - wav.shape[0]))
                s = 0
            else:
                s = np.random.randint(0, T - crop_frames + 1)
            e = s + crop_frames
            ss = s * hop_size
            se = ss + crop_samples
            cropped.append((f0[:, s:e], melspc[s:e, :], wav[ss:se], uv[:, s:e]))
        return collate_fn_padd(cropped)

    return random_crop_collate


class NaNDetected(Exception):
    def __init__(self, epoch):
        self.epoch = epoch
        super().__init__(f"NaN detected at epoch {epoch}")


class NaNStop(Exception):
    """nan_restart_until_epoch 以降に NaN を検出したときに送出。再スタートせず即停止。"""
    def __init__(self, epoch):
        self.epoch = epoch
        super().__init__(f"NaN detected at epoch {epoch} (above restart threshold)")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "epoch": epoch,
    }, save_path)


def evaluate(model, test_loader, loss_fn, cfg, device):
    model.eval()
    stft_loss_eval = 0
    env_loss_eval = 0

    window_lengths = cfg['training']['window_lengths']
    fft_lengths = [int(2 * i) for i in window_lengths]
    hop_lengths = [int(i / 4) for i in window_lengths]
    stft_loss_type = cfg['training'].get('stft_loss_type', 'log_linear')
    envelope_scale = cfg['training'].get('envelope_scale', 0.0)
    hop_size = cfg['preprocess']['hop_size']
    envelope_kernel = cfg['training'].get('envelope_kernel_size', hop_size * 2)
    envelope_stride = cfg['training'].get('envelope_stride', hop_size)
    envelope_directional = cfg['training'].get('envelope_directional', False)

    with torch.no_grad():
        for f0, log_melspc, wav, uv, mask in test_loader:
            f0 = torch.from_numpy(f0).float().to(device)
            log_melspc = torch.from_numpy(log_melspc).float().to(device)
            wav = torch.from_numpy(wav).float().to(device)
            uv = torch.from_numpy(uv).float().to(device)
            mask = mask.to(device)

            if isinstance(model, NHVSingV3):
                output = model(log_melspc, f0, uv)
            else:
                output = model(log_melspc, f0)
            del f0, log_melspc

            output = output.unsqueeze(1).masked_fill(mask.unsqueeze(1), 0)
            wav_for_loss = wav.unsqueeze(1).masked_fill(mask.unsqueeze(1), 0)
            valid = (~mask).unsqueeze(1).float()
            n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rms = ((wav_for_loss.pow(2) * valid).sum(dim=-1, keepdim=True) / n_valid).sqrt().clamp(min=1e-8)
            output = output / rms
            wav_for_loss = wav_for_loss / rms
            del wav, mask, valid, n_valid, rms

            stft_loss = loss_fn(output, wav_for_loss, fft_lengths,
                                window_lengths, hop_lengths, stft_loss_type)
            stft_loss_eval += stft_loss.item()
            del stft_loss

            if envelope_scale > 0:
                env_loss = envelope_loss_fn(output, wav_for_loss, envelope_kernel, envelope_stride,
                                            directional=envelope_directional)
                env_loss_eval += env_loss.item()
                del env_loss

            del wav_for_loss, output

    model.train()
    return stft_loss_eval / len(test_loader), env_loss_eval / len(test_loader)


def plot_spectrogram(spectrogram, title="", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none', vmin=vmin, vmax=vmax, cmap='magma')
    fig.colorbar(im, ax=ax, label='dB')
    if title:
        ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel bin')
    fig.tight_layout()
    fig.canvas.draw()
    plt.close()
    return fig


def plot_mel_diff(diff, title="", vmax=20):
    """diff: (mel_dim, T), fake_mel - real_mel [dB]"""
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(diff, aspect="auto", origin="lower",
                   interpolation='none', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    fig.colorbar(im, ax=ax, label='dB (fake − real)')
    if title:
        ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel bin')
    fig.tight_layout()
    fig.canvas.draw()
    plt.close()
    return fig


def compute_disc_badness(discriminator, wav_np, n_msd_sub, periods, device, T_frames):
    """disc を波形に通し、各 sub-disc の位置別 logit を frame 軸(T_frames)へ揃えた
    badness = 1 - logit ([n_subdisc, T_frames]) と行ラベルを返す。
    MSD は時間 conv の 1D logit を補間、MPD は period reshape を位相平均してから補間。"""
    wav_t = torch.from_numpy(np.ascontiguousarray(wav_np)).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        outs = discriminator(wav_t)
    x_dst = np.linspace(0.0, 1.0, num=T_frames)
    rows, labels = [], []
    for i, sub in enumerate(outs):
        flat = sub[-1].reshape(-1).float().cpu().numpy()           # 位置別 logit を 1D 化
        if i < n_msd_sub:
            v = flat                                               # MSD: [T_out]
            labels.append(f'MSD-s{i}')
        else:
            p = periods[i - n_msd_sub]
            T_conv = max(1, flat.shape[0] // p)
            v = flat[:T_conv * p].reshape(T_conv, p).mean(axis=1)  # period 位相平均 → [T_conv]
            labels.append(f'MPD-p{p}')
        if len(v) < 2:
            v = np.repeat(v, 2)
        x_src = np.linspace(0.0, 1.0, num=len(v))
        rows.append(1.0 - np.interp(x_dst, x_src, v))              # badness = 1 - logit(高=fake判定)
    return np.stack(rows, axis=0), labels


def plot_disc_badness(fake_bad, real_bad, row_labels, title=""):
    """fake_bad/real_bad: [n_subdisc, T_frames] の badness。上=fake / 下=real を同一スケールで対比
    (real が暗く fake のみ明るい箇所 = fake 特有の欠陥)。明=disc が fake と判定した時刻。"""
    n = len(row_labels)
    # 外れ値ロバストな vmax(稀な発散 logit が heatmap を潰さないよう 99 パーセンタイル)
    vmax = float(max(np.percentile(np.concatenate([fake_bad.ravel(), real_bad.ravel()]), 99.0), 1e-3))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, dat, tag in ((axes[0], fake_bad, 'fake'), (axes[1], real_bad, 'real')):
        im = ax.imshow(dat, aspect='auto', origin='lower', interpolation='none',
                       vmin=0.0, vmax=vmax, cmap='hot')
        fig.colorbar(im, ax=ax, label=f'badness [{tag}]')
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=7)
    axes[1].set_xlabel('Frame (mel time-axis)')
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.canvas.draw()
    plt.close()
    return fig


def _log_disc_badness(discriminator, npz_path, wav, real_mel, device, writer, epoch, cfg, basename):
    """tb_disc_badness:true かつ HiFiGAN disc(.msd/.mpd)あり のとき、fake(gate-on)/real 波形を
    disc に通し、各 sub-disc が「波形のどこを fake と判定したか」を heatmap で TensorBoard に記録。"""
    if not cfg['training'].get('tb_disc_badness', False) or discriminator is None:
        return
    if not (hasattr(discriminator, 'msd') and hasattr(discriminator, 'mpd')):
        return
    try:
        n_msd_sub = len(discriminator.msd.discriminators)
        periods = [d.period for d in discriminator.mpd.discriminators]
        T_frames = int(real_mel.shape[0])
        real_wav = np.load(npz_path)['wav']
        fake_bad, labels = compute_disc_badness(discriminator, wav, n_msd_sub, periods, device, T_frames)
        real_bad, _ = compute_disc_badness(discriminator, real_wav, n_msd_sub, periods, device, T_frames)
        writer.add_figure('disc_badness/' + basename,
                          plot_disc_badness(fake_bad, real_bad, labels,
                                            title=f"disc badness: {basename}  ep{epoch}"), epoch)
    except Exception as e:
        print(f"  [disc_badness] error {basename}: {e}")


def plot_linear_spec_pair(fake_wav, real_wav, sr, n_fft, hop, title="", mel_max=None):
    """fake/real の線形周波数スペクトログラム(0-Nyquist)を上下2段で対比。
    mel では潰れる高域の harmonic(横線)vs noise(広帯域)が見える。
    plot_stft_v4_ep1340.py のパターン(amplitude_to_db + extent[0,t,0,sr/2])踏襲。"""
    specs = []
    for w in (fake_wav, real_wav):
        S = np.abs(librosa.stft(np.asarray(w, dtype=np.float64), n_fft=n_fft, hop_length=hop))
        specs.append(librosa.amplitude_to_db(S + 1e-9))
    vmax = float(max(specs[0].max(), specs[1].max()))
    vmin = vmax - 80.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
    for ax, S, tag in ((axes[0], specs[0], 'fake'), (axes[1], specs[1], 'real')):
        im = ax.imshow(S, aspect='auto', origin='lower', cmap='magma', interpolation='none',
                       vmin=vmin, vmax=vmax, extent=[0, S.shape[1] * hop / sr, 0, sr / 2])
        fig.colorbar(im, ax=ax, label=f'dB [{tag}]')
        ax.set_ylabel(f'{tag}   Hz')
        if mel_max:
            ax.axhline(mel_max, color='cyan', lw=0.6, ls='--')   # mel 上限(これ以上 mel は盲目)
    axes[1].set_xlabel('Time [s]')
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.canvas.draw()
    plt.close()
    return fig


def _log_linear_spec(npz_path, wav, writer, epoch, cfg, basename):
    """tb_linear_spec:true のとき fake(gate-on)/real の線形スペクトログラムを TensorBoard に記録。
    mel では潰れる高域 harmonic-vs-noise を監視(本来 noise の高域が harmonic で再現されていないか)。"""
    if not cfg['training'].get('tb_linear_spec', False):
        return
    try:
        p = cfg['preprocess']
        sr = p['sample_rate']
        n_fft = p.get('fft_size', 2048)
        hop = p['hop_size']
        real_wav = np.load(npz_path)['wav']
        writer.add_figure('linear_spec/' + basename,
                          plot_linear_spec_pair(wav, real_wav, sr, n_fft, hop,
                                                title=f"linear spec: {basename}  ep{epoch}",
                                                mel_max=p.get('mel_max')), epoch)
    except Exception as e:
        print(f"  [linear_spec] error {basename}: {e}")


def inference(model, npz_path, device, cfg, disable_uv=False):
    """disable_uv=True で v/uv hard gate を無効化(uv=0=全 voiced 扱い)して生成する。
    無声区間で harmonic 源が gate されなくなるので、ccep フィルタが自力で harmonic を
    抑制できているか(=vuv_dropout 学習の達成度)を可視化・試聴できる。"""
    model.eval()

    with torch.no_grad():
        data = np.load(npz_path)
        f0_np, uv_np = norm_interp_f0(data['f0'])
        f0_np = f0_np[np.newaxis][np.newaxis]
        uv_np = uv_np[np.newaxis][np.newaxis]
        log_melspc_np = data['log_melspc'][np.newaxis]
        real_mel = data['log_melspc']
        del data

        f0 = torch.Tensor(f0_np).to(device)
        uv = torch.Tensor(uv_np).to(device)
        log_melspc = torch.Tensor(log_melspc_np).to(device)
        del f0_np, log_melspc_np, uv_np

        if isinstance(model, NHVSingV3):
            if disable_uv:
                uv = torch.zeros_like(uv)   # hard gate 無効化: 無声でも harmonic 源を gate しない
            synthesized_tensor = model(log_melspc, f0, uv)
        else:
            synthesized_tensor = model(log_melspc, f0)
        del f0, log_melspc

        synthesized = torch.squeeze(synthesized_tensor).to('cpu').detach().numpy().copy()
        del synthesized_tensor

    p_cfg = cfg['preprocess']
    if p_cfg.get('mel_format') == 'diffsinger':
        # preprocess(wav_to_mel)と同一計算: ln(自然対数, clip 1e-5) + center=False + reflect pad
        # (nvSTFT/pc_nsf 互換)。real(log_melspc=ln)と同スケールになり plot/diff が一致する。
        # ※ 旧版は log10 だったが real が ln に変わったため、log10 のままだと fake が約2.3倍小さく
        #   表示され「やたら赤い」ズレになる(音声自体は ln 一貫で正常)。
        win = p_cfg.get('win_size', p_cfg['fft_size'])
        pad = (p_cfg['fft_size'] - p_cfg['hop_size']) // 2
        yp = np.pad(synthesized.astype(np.float64), (pad, pad), mode='reflect')
        x_stft = librosa.stft(yp, n_fft=p_cfg['fft_size'], hop_length=p_cfg['hop_size'],
                              win_length=win, window='hann', center=False)
        mel_basis = librosa.filters.mel(sr=p_cfg['sample_rate'], n_fft=p_cfg['fft_size'],
                                        n_mels=p_cfg['mel_dim'], fmin=p_cfg['mel_min'], fmax=p_cfg['mel_max'])
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            _mel = mel_basis.astype(np.float64) @ np.abs(x_stft)
        fake_mel = np.log(np.maximum(1e-5, _mel)).T.astype(np.float32)
    else:  # V2 互換（power_to_db）
        S_fake = librosa.feature.melspectrogram(
            y=synthesized,
            sr=p_cfg['sample_rate'],
            n_fft=p_cfg['fft_size'],
            hop_length=p_cfg['hop_size'],
            win_length=p_cfg['hop_size'] * 4,
            n_mels=p_cfg['mel_dim'],
            fmin=p_cfg['mel_min'],
            fmax=p_cfg['mel_max'],
            center=True,
        )
        fake_mel = librosa.power_to_db(S_fake, ref=1.0).T

    model.train()
    return synthesized, real_mel, fake_mel


def _select_tb_files(npz_paths: list, n_per_singer: int) -> list:
    """歌手ごとに最大 n_per_singer 件を選んで返す。
    ファイル名の '#' 前の部分（例: Alto-1）を歌手IDとして使う。
    歌手IDが取れない場合（'#' なし）はそのまま通す。
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for p in npz_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        # '#' があれば "#" 前 (例: Alto-1#newboy_0000 → Alto-1)
        # なければ最初の '_' 前 (例: Alto-1_newboy_0000 → Alto-1)
        singer = stem.split('#')[0] if '#' in stem else stem.split('_')[0]
        groups[singer].append(p)
    result = []
    for singer in sorted(groups):
        result.extend(groups[singer][:n_per_singer])
    return result


def _log_uvoff_and_get_mel(model, npz_path, device, writer, epoch, cfg, basename, sample_rate):
    """v/uv ゲート無効化(uv=0)版を生成し、その音声を audio/.../fake_uvoff に記録、
    uv-off の mel を返す(plot/.../fake と mel_diff のデフォルトに使う)。
    NHVSingV3 の hard gate を外すと無声区間で harmonic 源が gate されないので、
    ccep フィルタが自力で harmonic を抑制できているか(=学習目標の達成度)が plot で見える:
      - 抑制できている → 無声区間が real と同じく noise のみ(横線なし)。
      - できていない   → 無声区間に harmonic(横線 / 試聴でビリつき)が出る。
    tb_log_uvoff:false / 非 NHVSingV3 / エラー時は None を返し、呼び出し側は gate-on mel に
    フォールバックする(=従来どおりの fake plot)。"""
    if not cfg['training'].get('tb_log_uvoff', False) or not isinstance(model, NHVSingV3):
        return None
    try:
        wav_u, _, fake_mel_u = inference(model, npz_path, device, cfg, disable_uv=True)
    except Exception as e:
        print(f"  [uvoff] inference error {npz_path}: {e}")
        return None
    writer.add_audio('audio/' + basename + '/fake_uvoff',
                     torch.from_numpy(wav_u).unsqueeze(0), epoch, sample_rate)
    return fake_mel_u


def inference_pinpoint_files(model, device, writer, epoch, cfg, logged_real_mels: set, discriminator=None):
    """tb_pinpoint_files に列挙した npz を必ず全件ログに記録する。
    別の SummaryWriter（{log_dir}_pinpoint）に書くので TensorBoard で独立した run として表示される。
    """
    pinpoint_paths = cfg['training'].get('tb_pinpoint_files', [])
    if not pinpoint_paths:
        return

    sample_rate = cfg['preprocess']['sample_rate']
    for npz_path in pinpoint_paths:
        if not os.path.exists(npz_path):
            print(f"  [pinpoint] not found, skip: {npz_path}")
            continue
        try:
            wav, real_mel, fake_mel = inference(model, npz_path, device, cfg)
        except Exception as e:
            print(f"  [pinpoint] inference error {npz_path}: {e}")
            continue

        basename = os.path.splitext(os.path.basename(npz_path))[0]

        if basename not in logged_real_mels:
            writer.add_figure('plot/' + basename + '/real',
                              plot_spectrogram(real_mel.T, title=f"real: {basename}",
                                               vmin=cfg['preprocess'].get('mel_vmin', -80), vmax=cfg['preprocess'].get('mel_vmax', 0)), 0)
            real_wav = np.load(npz_path)['wav']
            writer.add_audio('audio/' + basename + '/real',
                             torch.from_numpy(real_wav).unsqueeze(0), 0, sample_rate)
            logged_real_mels.add(basename)

        # plot/diff は uv-off 版をデフォルトに(無声 harmonic の自力抑制を可視化)。
        # uv-off 音声は audio/.../fake_uvoff に記録され、uv-off mel が返る。無効時は gate-on にフォールバック。
        fake_mel_u = _log_uvoff_and_get_mel(model, npz_path, device, writer, epoch, cfg, basename, sample_rate)
        fake_mel_plot = fake_mel_u if fake_mel_u is not None else fake_mel
        _fake_tag = 'fake uv-off' if fake_mel_u is not None else 'fake'
        writer.add_figure('plot/' + basename + '/fake',
                          plot_spectrogram(fake_mel_plot.T, title=f"{_fake_tag}: {basename}  ep{epoch}",
                                           vmin=cfg['preprocess'].get('mel_vmin', -80), vmax=cfg['preprocess'].get('mel_vmax', 0)), epoch)
        writer.add_audio('audio/' + basename + '/fake',
                         torch.from_numpy(wav).unsqueeze(0), epoch, sample_rate)

        min_T = min(real_mel.shape[0], fake_mel_plot.shape[0])
        diff = fake_mel_plot[:min_T] - real_mel[:min_T]  # (T, mel_dim)
        writer.add_figure('mel_diff/' + basename,
                          plot_mel_diff(diff.T, title=f"diff: {basename}  ep{epoch}",
                                        vmax=(5.0 if cfg['preprocess'].get('mel_format') == 'diffsinger' else 20.0)), epoch)

        _log_disc_badness(discriminator, npz_path, wav, real_mel, device, writer, epoch, cfg, basename)
        _log_linear_spec(npz_path, wav, writer, epoch, cfg, basename)


def inference_test_data(model, device, writer, epoch, cfg, logged_real_mels: set, discriminator=None):
    test_data_folder = cfg['training']['test_dir']
    all_npz_paths = sorted(glob.glob(os.path.join(test_data_folder, '**/*.npz'), recursive=True))

    n_per_singer = cfg['training'].get('tb_files_per_singer', 0)
    if n_per_singer > 0:
        all_npz_paths = _select_tb_files(all_npz_paths, n_per_singer)

    # inference wav 廃止に伴い save_dir 作成も不要（試聴は tensorboard の add_audio で行う）。
    # inference_output_base_dir = Path(cfg['training'].get('inference_output_dir', 'dataset/inference'))
    # save_dir = inference_output_base_dir / f"{epoch}"
    # save_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = cfg['preprocess']['sample_rate']
    for i, npz_path in enumerate(all_npz_paths):
        wav, real_mel, fake_mel = inference(model, npz_path, device, cfg)

        basename = os.path.splitext(os.path.basename(npz_path))[0]

        # real_mel / ground truth audio は固定なので初回のみ記録
        if basename not in logged_real_mels:
            writer.add_figure('plot/' + basename + '/real',
                              plot_spectrogram(real_mel.T, title=f"real: {basename}",
                                               vmin=cfg['preprocess'].get('mel_vmin', -80), vmax=cfg['preprocess'].get('mel_vmax', 0)), 0)
            real_wav = np.load(npz_path)['wav']
            writer.add_audio('audio/' + basename + '/real',
                             torch.from_numpy(real_wav).unsqueeze(0), 0, sample_rate)
            logged_real_mels.add(basename)

        # plot/diff は uv-off 版をデフォルトに(無声 harmonic の自力抑制を可視化)。
        # uv-off 音声は audio/.../fake_uvoff に記録され、uv-off mel が返る。無効時は gate-on にフォールバック。
        fake_mel_u = _log_uvoff_and_get_mel(model, npz_path, device, writer, epoch, cfg, basename, sample_rate)
        fake_mel_plot = fake_mel_u if fake_mel_u is not None else fake_mel
        _fake_tag = 'fake uv-off' if fake_mel_u is not None else 'fake'
        writer.add_figure('plot/' + basename + '/fake',
                          plot_spectrogram(fake_mel_plot.T, title=f"{_fake_tag}: {basename}  ep{epoch}",
                                           vmin=cfg['preprocess'].get('mel_vmin', -80), vmax=cfg['preprocess'].get('mel_vmax', 0)), epoch)
        writer.add_audio('audio/' + basename + '/fake',
                         torch.from_numpy(wav).unsqueeze(0), epoch, sample_rate)

        min_T = min(real_mel.shape[0], fake_mel_plot.shape[0])
        diff = fake_mel_plot[:min_T] - real_mel[:min_T]  # (T, mel_dim)
        writer.add_figure('mel_diff/' + basename,
                          plot_mel_diff(diff.T, title=f"diff: {basename}  ep{epoch}",
                                        vmax=(5.0 if cfg['preprocess'].get('mel_format') == 'diffsinger' else 20.0)), epoch)

        _log_disc_badness(discriminator, npz_path, wav, real_mel, device, writer, epoch, cfg, basename)
        _log_linear_spec(npz_path, wav, writer, epoch, cfg, basename)

        # inference wav の書き出しは廃止（試聴は tensorboard の add_audio で可能）。
        # Google Drive(FUSE) では libsndfile 書き込みが System error になる問題も回避できる。
        # save_path = save_dir / f"{i:03d}.wav"
        # sf.write(save_path, wav, sample_rate)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(args, force_restart: bool = False):
    cfg = load_config(args.config)
    log_dir = Path(cfg['training']['log_dir'])
    snapshot_dir = Path(cfg['training']['snapshot_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    pinpoint_log_dir = Path(str(log_dir) + '_pinpoint')
    pinpoint_writer = SummaryWriter(log_dir=pinpoint_log_dir) \
        if cfg['training'].get('tb_pinpoint_files') else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ltv_filter_cfg = cfg['model']['ltv_filter']
    ModelClass = select_model_class(cfg['model']['vocoder'], ltv_filter_cfg)
    # uv を forward に渡すか = 選択された生成器が V3 系(V3/V4/V5/V6)か。以前は
    # use_hard_vuv フラグ単独で判定していたが、クラス選択(use_polar_phase/use_v6 等)と
    # ズレると uv 送出とクラスが食い違い初回バッチで TypeError になる(例: use_v6:true だけ
    # 書き use_hard_vuv 省略 → V6 は forward(x,cf0,uv) 必須なのに uv 未送出)。選択クラス
    # から判定して dispatch を常にクラスと一致させる(eval/inference の isinstance 判定とも整合)。
    use_v3 = issubclass(ModelClass, NHVSingV3)
    print(f"Model: {ModelClass.__name__}")
    _tcfg = cfg['training']                                   # pitch augmentation の設定を起動時に明示
    if _tcfg.get('pitch_aug', False):
        _pa_lo, _pa_hi = _tcfg.get('pitch_aug_semitones', [-7, 2])
        _pa_p = _tcfg.get('pitch_aug_prob', 0.0)
        _pa_when = '毎バッチ' if _pa_p >= 1.0 else f'約{_pa_p * 100:.0f}% のバッチ'
        print(f"pitch_aug: ON  prob={_pa_p} ({_pa_when})  range=[{_pa_lo}, {_pa_hi}] 半音"
              f"  (RefineGAN流: wav リサンプルで F0×r・フォルマントも移動)")
    else:
        print("pitch_aug: OFF")
    model = ModelClass(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=ltv_filter_cfg,
    ).to(device)

    disc_cfg = cfg.get('discriminator', {})
    use_hifigan = disc_cfg.get('use_hifigan_disc', False)
    use_wavenet = disc_cfg.get('use_wavenet', False)
    if use_hifigan:
        # train_mrd: MSD/MPD/MRD を config で選択(既定 MRD+MPD = UnivNet/RefineGAN/BigVGAN 構成)。
        # forward 出力順は常に [MSD, MPD, MRD]。有効群を順に grp1/grp2 に割当(最大2群)。
        from discriminator import HiFiGANDiscriminator
        periods = disc_cfg.get('discriminator_periods', [3, 5, 7, 11, 17, 23, 37])
        use_msd = disc_cfg.get('use_msd', False)     # 既定 MSD OFF(凍結群を外す)
        use_mpd = disc_cfg.get('use_mpd', True)
        use_mrd = disc_cfg.get('use_mrd', True)
        mrd_resolutions = disc_cfg.get('mrd_resolutions', None)
        discriminator = HiFiGANDiscriminator(
            periods=periods, use_msd=use_msd, use_mpd=use_mpd,
            use_mrd=use_mrd, mrd_resolutions=mrd_resolutions).to(device)
        _groups = []
        if use_msd: _groups.append(('msd', len(discriminator.msd.discriminators)))
        if use_mpd: _groups.append(('mpd', len(discriminator.mpd.discriminators)))
        if use_mrd: _groups.append(('mrd', len(discriminator.mrd.discriminators)))
        assert 1 <= len(_groups) <= 2, \
            "train_mrd は1-2群(MSD/MPD/MRD のうち最大2つ)。3群は train.py の grp3 拡張が必要"
        n_msd_sub  = _groups[0][1]                                  # grp1(forward 先頭群)
        n_grp2_sub = _groups[1][1] if len(_groups) > 1 else 0       # grp2(2番目の群)
        n_wn_sub = 0
        use_wavenet = False
        # ログラベル: (grp1, grp2, None)。MRD+MPD なら ('mpd','mrd',None) → d_mpd/d_mrd で表示。
        sub_labels = (_groups[0][0], _groups[1][0] if len(_groups) > 1 else 'grp2', None)
        _grp2_module_name = _groups[1][0] if len(_groups) > 1 else None
        print(f"train_mrd Discriminator: grp1={sub_labels[0]}:{n_msd_sub} "
              f"grp2={sub_labels[1]}:{n_grp2_sub}  (use_msd={use_msd}, use_mpd={use_mpd}, use_mrd={use_mrd})")
    else:
        use_msd = disc_cfg.get('use_msd', True)
        stft_filters = disc_cfg.get('stft_filters', 32)
        stft_filters_scale = disc_cfg.get('stft_filters_scale', 1)
        discriminator = DiscriminatorWithComplexSTFT(
            use_msd=use_msd, stft_filters=stft_filters, use_wavenet=use_wavenet,
            stft_filters_scale=stft_filters_scale
        ).to(device)
        print(f"DiscriminatorWithComplexSTFT: use_msd={use_msd}, "
              f"stft_filters={stft_filters}, use_wavenet={use_wavenet}")
        n_msd_sub = len(discriminator.msd.discriminators) if use_msd else 0
        n_grp2_sub = len(discriminator.ms_stft.discriminators)
        n_wn_sub = 1 if use_wavenet else 0
        # 自前 disc は (MSD 群, 複素STFT 群, WaveNet 群)。後方互換ラベル。
        sub_labels = ('msd', 'stft', 'wn')
        _grp2_module_name = 'ms_stft'
        print(f"  sub-disc counts: msd={n_msd_sub}, stft={n_grp2_sub}, wavenet={n_wn_sub}")

    # RAdam(NHVSing 既存)。variance rectification(warmup 内蔵)で序盤が安定し lr 0.001 でも
    # 発散しない。weight_decay=0 では AdamW の利点(正しい weight decay)が無く、RAdam が有利。
    optimizer_g = torch.optim.RAdam(model.parameters(), lr=cfg['training']['lr_g'], eps=1e-4)
    # disc の sub-disc 別 lr/clip/gnorm: MSD と「第2群」(複素STFT=ms_stft or MPD=mpd)は構造/grad
    # スケールが違うので別管理できる。第2群は自動判定。lr_d_grp2 で第2群だけ別 lr を与える
    # (未指定なら lr_d と同じ=従来動作)。param_groups[0]=MSD等, [1]=第2群。
    _grp2_module = getattr(discriminator, _grp2_module_name, None) if _grp2_module_name else None
    _grp2_ids_set = {id(p) for p in _grp2_module.parameters()} if _grp2_module is not None else set()
    _grp2_label = sub_labels[1] if sub_labels[1] else 'grp2'   # 'stft'(複素STFT) or 'mpd'(HiFi-GAN)
    _lr_d = cfg['training']['lr_d']
    _lr_d_grp2 = cfg['training'].get('lr_d_grp2', _lr_d)
    # disc betas: GAN は β1=0.5(DCGAN以来の標準。非定常な敵対信号に追従するには低momentum)。
    # RAdam 既定 β1=0.9 は高momentumで MRD がサドル(d=0.5定数)から抜けにくい → config で 0.5 推奨。
    _disc_betas = (float(cfg['training'].get('disc_beta1', 0.9)),
                   float(cfg['training'].get('disc_beta2', 0.999)))
    if _grp2_module is not None and _lr_d_grp2 != _lr_d:
        _grp2_params  = [p for p in discriminator.parameters() if id(p) in _grp2_ids_set]
        _other_params = [p for p in discriminator.parameters() if id(p) not in _grp2_ids_set]
        optimizer_d = torch.optim.RAdam([
            {'params': _other_params, 'lr': _lr_d},          # group 0: grp1
            {'params': _grp2_params,  'lr': _lr_d_grp2},     # group 1: grp2
        ], betas=_disc_betas)
        print(f"optimizer: RAdam lr_g={cfg['training']['lr_g']} lr_d={_lr_d}(grp1) lr_d_grp2={_lr_d_grp2}({_grp2_label}) disc_betas={_disc_betas}")
    else:
        optimizer_d = torch.optim.RAdam(discriminator.parameters(), lr=_lr_d, betas=_disc_betas)
        print(f"optimizer: RAdam lr_g={cfg['training']['lr_g']} lr_d={_lr_d} disc_betas={_disc_betas}")

    use_amp = cfg['training'].get('use_amp', False) and device.type == 'cuda'
    # amp_dtype: 'bfloat16'(推奨。fp32 と同じ指数部で overflow しない) or 'float16'(要 GradScaler)。
    _amp_dtype_str = cfg['training'].get('amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if _amp_dtype_str == 'bfloat16' else torch.float16
    # bf16 は overflow しないため GradScaler 不要(enabled=False で no-op)。fp16 のみ loss scaling 必要。
    _use_scaler = use_amp and (amp_dtype == torch.float16)
    if use_amp:
        print(f"AMP dtype={_amp_dtype_str}, GradScaler={'on' if _use_scaler else 'off(bf16はscaling不要)'}")
    _init_scale = 2 ** 10  # fp16 用 init_scale (mel×100 の overflow 緩和)。bf16 では未使用。
    scaler_g = torch.amp.GradScaler('cuda', enabled=_use_scaler, init_scale=_init_scale)
    scaler_d = torch.amp.GradScaler('cuda', enabled=_use_scaler, init_scale=_init_scale)
    # NaN debug: 環境変数 DETECT_ANOMALY=1 で backward の NaN 発生演算を trace する
    # (遅くなるが、どの loss/演算が NaN を出すか正確に特定できる)。
    if os.environ.get('DETECT_ANOMALY'):
        torch.autograd.set_detect_anomaly(True)
        print(">>> torch.autograd.set_detect_anomaly(True): NaN 発生演算を trace します(低速)")
    if use_amp:
        print("AMP (mixed precision) enabled.")

    # --- Gradient Accumulation ---
    accum_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    print(f"gradient_accumulation_steps={accum_steps} "
          f"(実効バッチサイズ = {cfg['training']['batch_size']} × {accum_steps} = "
          f"{cfg['training']['batch_size'] * accum_steps})")

    hop_size = cfg['preprocess']['hop_size']
    amp_augment = cfg['training'].get('amp_augment', False)
    amp_aug_range = tuple(cfg['training'].get('amp_aug_range', [0.5, 2.0]))
    vuv_dropout_prob = cfg['training'].get('vuv_dropout_prob', 0.0)
    if vuv_dropout_prob > 0:
        print(f"v/uv dropout: prob={vuv_dropout_prob} (確率で uv gate 無効化=無声harmonic抑制学習+uv誤りロバスト)")
    if amp_augment:
        print(f"amp_augment: enabled, range={amp_aug_range}")
    train_dataset = VocoderDataset(dataset_dir=cfg['training']['train_dir'], hop_size=hop_size,
                                   augment=amp_augment, amp_aug_range=amp_aug_range,
                                   diffsinger_mel=use_v3)
    crop_frames = cfg['training'].get('crop_frames', cfg['training'].get('max_train_frames', None))
    collate_train = make_random_crop_collate(crop_frames, hop_size=hop_size)
    if crop_frames:
        print(f"crop_frames={crop_frames}: NSF流ランダム固定長クロップ ({crop_frames * hop_size} samp)")

    max_samples = cfg['training'].get('max_samples_per_epoch', None)
    if max_samples and max_samples < len(train_dataset):
        sampler = RandomSampler(train_dataset, replacement=False, num_samples=max_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg['training']['batch_size'], sampler=sampler,
            num_workers=cfg['training']['num_workers'], collate_fn=collate_train,
            drop_last=True, pin_memory=True
        )
        print(f"max_samples_per_epoch={max_samples}: {max_samples // cfg['training']['batch_size']} steps/epoch")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
            num_workers=cfg['training']['num_workers'], collate_fn=collate_train,
            drop_last=True, pin_memory=True
        )
    test_dataset = VocoderDataset(dataset_dir=cfg['training']['test_dir'], hop_size=hop_size,
                                  diffsinger_mel=use_v3)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_padd
    )
    test_list = list(test_loader)

    start_epoch = 0

    if force_restart:
        print("Force restart: starting from epoch 0.")
    else:
        resume_path = args.resume_path
        if resume_path is None:
            snapshots = sorted(snapshot_dir.glob("*.pth"))
            if snapshots:
                resume_path = str(snapshots[-1])
                print(f"Auto-resuming from latest snapshot: {resume_path}")

        if resume_path:
            print(f"Resuming from: {resume_path}")
            snapshot = torch.load(resume_path, map_location=device)
            model.load_state_dict(snapshot['model'])
            _reinit_disc = cfg['training'].get('reinit_disc', False)
            if _reinit_disc:
                print("  reinit_disc=True: Discriminator を新規初期化(勝ち切った鋭い D をリセット、G は継承)")
            else:
                # WaveNet disc を簡素化(構造変更)したため strict=False。不一致(WaveNet)は再初期化。
                _dmiss, _dunexp = discriminator.load_state_dict(snapshot['discriminator'], strict=False)
                if _dmiss or _dunexp:
                    print(f"  disc strict=False: missing={len(_dmiss)} unexpected={len(_dunexp)} (WaveNet 簡素化で再初期化)")
            try:
                optimizer_g.load_state_dict(snapshot['optimizer_g'])
            except (ValueError, KeyError) as e:
                print(f"  optimizer_g は optimizer 変更(AdamW)のため fresh: {type(e).__name__}")
            if _reinit_disc:
                print("  optimizer_d も fresh(reinit_disc=True)")
            else:
                try:
                    optimizer_d.load_state_dict(snapshot['optimizer_d'])
                except (ValueError, KeyError) as e:
                    print(f"  optimizer_d は disc/optimizer 変更のため fresh: {type(e).__name__}")
            # load_state_dict は ckpt 内の古い lr(param_groups)も復元するため、
            # config で lr を変えても resume では効かない。config 値で上書きする。
            for pg in optimizer_g.param_groups:
                pg['lr'] = cfg['training']['lr_g']
            # param_groups が複数(MSD/第2群別lr)なら group0=lr_d, group1=lr_d_grp2 で上書き。
            _lr_d_ov = cfg['training']['lr_d']
            _lr_d_grp2_ov = cfg['training'].get('lr_d_grp2', _lr_d_ov)
            if len(optimizer_d.param_groups) >= 2:
                optimizer_d.param_groups[0]['lr'] = _lr_d_ov
                optimizer_d.param_groups[1]['lr'] = _lr_d_grp2_ov
                print(f"  optimizer lr 上書き: lr_g={cfg['training']['lr_g']} lr_d={_lr_d_ov}(MSD) lr_d_grp2={_lr_d_grp2_ov}({_grp2_label})")
            else:
                for pg in optimizer_d.param_groups:
                    pg['lr'] = _lr_d_ov
                print(f"  optimizer lr を config で上書き: lr_g={cfg['training']['lr_g']} lr_d={_lr_d_ov}")
            start_epoch = snapshot['epoch'] + 1
            print(f"Starting from epoch {start_epoch}")

        elif args.finetune_from:
            print(f"Fine-tuning from: {args.finetune_from}")
            snapshot = torch.load(args.finetune_from, map_location=device)
            missing, unexpected = model.load_state_dict(snapshot['model'], strict=False)
            print(f"  Generator weights loaded (strict=False). 新規(load外)param: {list(missing)}")
            # postfilter 等で param 構成が変わると optimizer state が不一致になるため、
            # 構成一致(missing 無し)のときだけ optimizer を引き継ぐ。V4 は postfilter が
            # missing になるので optimizer は fresh（Adam momentum 再構築）。
            if not missing and 'optimizer_g' in snapshot:
                optimizer_g.load_state_dict(snapshot['optimizer_g'])
                print("  optimizer_g state loaded (Adam momentum preserved).")
            else:
                print("  optimizer_g: fresh (param 構成が異なるため momentum 再構築).")
            start_epoch = 0
            print("  Discriminator: initialized fresh.")
            print(f"Starting from epoch {start_epoch}")

        else:
            print("Starting new training (no checkpoint found).")

    # --- Noise branch curriculum ---
    freeze_until = cfg['training'].get('noise_branch_freeze_epochs', 0)
    if freeze_until > 0 and start_epoch < freeze_until:
        model.convs_onnx.conv_noise.requires_grad_(False)
        print(f"[Curriculum] Noise branch frozen until epoch {freeze_until}")

    # real_mel は固定なので basename ごとに初回のみ TensorBoard に記録する
    logged_real_mels: set = set()
    pinpoint_logged_reals: set = set()

    # --- 学習パラメータ ---
    harmonic_penalty_scale = cfg['training'].get('harmonic_penalty_scale', 0.0)
    harmonic_penalty_start = cfg['training'].get('harmonic_penalty_start', 0)
    if harmonic_penalty_scale > 0:
        print(f"harmonic_penalty_scale={harmonic_penalty_scale}, start={harmonic_penalty_start}: unvoiced harmonic penalty enabled")
    adversarial_start = cfg['training']['adversarial_start']
    disc_grad_clip = cfg['training'].get('disc_grad_clip', 1.0)  # disc(MSD)勾配 clip 上限
    disc_grad_clip_grp2 = cfg['training'].get('disc_grad_clip_grp2', disc_grad_clip)  # 第2群(複素STFT/MPD)用
    _disc_has_grp2 = _grp2_module is not None  # ms_stft(複素STFT) or mpd(HiFi-GAN) があれば分離clip可能
    gen_grad_clip = cfg['training'].get('gen_grad_clip', 1.0)     # generator 勾配 clip 上限（gnorm_g 実測で調整）
    print(f"disc_grad_clip={disc_grad_clip}(MSD) disc_grad_clip_grp2={disc_grad_clip_grp2}({_grp2_label},分離={_disc_has_grp2}) gen_grad_clip={gen_grad_clip}")
    adversarial_scale = cfg['training']['adversarial_scale']
    # WaveNet disc 用の独立スケール（未指定なら adversarial_scale と同じ＝寄与を 1/N→対等に）
    adversarial_scale_wavenet = cfg['training'].get('adversarial_scale_wavenet', adversarial_scale)
    print("adversarial_scale:", adversarial_scale, "/ wavenet:", adversarial_scale_wavenet)
    feature_matching_scale = cfg['training']['feature_matching_scale']
    # 群2(=MPD)の adv/FM 重み。NHV は周期を DSP 源で作り MPD は均衡=満足→MPD7本 vs MSD3本で G 勾配の~70%が満たされた MPD に偏る。
    # <1 で MPD を下げ、波形テクスチャ本丸の MSD へ配分(0=MPD無効。ゼロにはせず周期/倍音の番人は残す)。1.0=従来どおり。ログの adv/d は生値(重み前)で監視可。
    grp2_loss_weight = cfg['training'].get('grp2_loss_weight', 1.0)
    print("grp2(MPD) loss weight:", grp2_loss_weight)
    adversarial_warmup_epochs = cfg['training'].get('adversarial_warmup_epochs', 0)
    envelope_scale = cfg['training'].get('envelope_scale', 0.0)
    envelope_start = cfg['training'].get('envelope_start', 0)
    envelope_kernel = cfg['training'].get('envelope_kernel_size', hop_size * 2)
    envelope_stride = cfg['training'].get('envelope_stride', hop_size)
    envelope_directional = cfg['training'].get('envelope_directional', False)
    if envelope_scale > 0:
        print(f"envelope_loss: scale={envelope_scale}, start={envelope_start}, kernel={envelope_kernel}, stride={envelope_stride}, directional={envelope_directional}")
    window_lengths = cfg['training']['window_lengths']
    fft_lengths = [int(2 * i) for i in window_lengths]
    hop_lengths = [int(i / 4) for i in window_lengths]
    stft_loss_scale = cfg['training'].get('stft_loss_scale', 1.0)  # 多res STFT 主再構成 loss 重み(NSF流: 2.5)
    stft_loss_type = cfg['training'].get('stft_loss_type', 'log_linear')  # 'log_linear'(従来) or 'nsf'(SC+log-mag, 平均)
    nan_restart_until = cfg['training'].get('nan_restart_until_epoch', 30)
    r1_gamma = cfg['training'].get('r1_gamma', 0.0)        # R1 勾配ペナルティ(0=無効)。D 過強の抑制
    r1_interval = cfg['training'].get('r1_interval', 16)   # lazy R1: 何バッチ毎に計算するか
    if r1_gamma > 0:
        print(f"R1 gradient penalty: gamma={r1_gamma}, interval={r1_interval}(lazy)")
    # 適応ゲート: EMA d_loss が下限割れ(D勝ち過ぎ)で D 更新を skip し自動 balance
    adaptive_d_gate = cfg['training'].get('adaptive_d_gate', False)
    d_gate_floor = cfg['training'].get('d_gate_floor', 0.2)
    d_gate_ema = cfg['training'].get('d_gate_ema', 0.98)
    d_msd_ema = 0.5    # MSD群 d_loss EMA(群別ゲート。区別不能=0.5 から開始)
    d_mpd_ema = 0.5    # MPD群 d_loss EMA
    if adaptive_d_gate:
        print(f"adaptive_d_gate: ON 群別(MSD/MPD)(floor={d_gate_floor}, ema={d_gate_ema}) — 各群 EMA d_loss<floor でその群の更新skip")

    # mel reconstruction loss (NSF-HiFiGAN/SingingVocoders 流の主再構成 loss)。
    # 生成 wav を diffsinger mel に戻して入力 mel と L1。preprocess と完全一致(検証済 0.00002)。
    mel_loss_scale = cfg['training'].get('mel_loss_scale', 0.0)
    _pp = cfg['preprocess']
    mel_sr, mel_nfft, mel_hop = _pp['sample_rate'], _pp['fft_size'], _pp['hop_size']
    mel_win = _pp.get('win_size', _pp['fft_size'])
    mel_dim, mel_fmin, mel_fmax = _pp['mel_dim'], _pp['mel_min'], _pp['mel_max']
    if mel_loss_scale > 0:
        print(f"mel_loss: scale={mel_loss_scale} ({mel_sr}/{mel_nfft}/hop{mel_hop}/{mel_dim}bin)")

    # スクリプト開始時に real(GT) のみ再プロット（vmin/vmax のスケール修正を反映）。
    # fake は未学習(zero-init で flat)なので出さない。fake は学習中の save_interval で記録される。
    _vmin = cfg['preprocess'].get('mel_vmin', -80)
    _vmax = cfg['preprocess'].get('mel_vmax', 0)
    _sr = cfg['preprocess']['sample_rate']
    for _p in sorted(glob.glob(os.path.join(cfg['training']['test_dir'], '**/*.npz'), recursive=True)):
        _z = np.load(_p)
        _bn = os.path.splitext(os.path.basename(_p))[0]
        if _bn in logged_real_mels:
            continue
        writer.add_figure('plot/' + _bn + '/real',
                          plot_spectrogram(_z['log_melspc'].T, title=f"real: {_bn}",
                                           vmin=_vmin, vmax=_vmax), 0)
        writer.add_audio('audio/' + _bn + '/real',
                         torch.from_numpy(_z['wav']).unsqueeze(0), 0, _sr)
        logged_real_mels.add(_bn)

    for epoch in range(start_epoch, cfg['training']['n_epoch']):
        # Noise branch unfreeze
        if freeze_until > 0 and epoch == freeze_until:
            model.convs_onnx.conv_noise.requires_grad_(True)
            print(f"[Curriculum] Noise branch unfrozen at epoch {epoch}")

        # noise_std=0 でノイズ枝を無効化（freeze 期間中は white noise が harmonic を妨害しないように）
        effective_noise_std = 0.0 if (freeze_until > 0 and epoch < freeze_until) else -1.0

        tic = time.time()
        stft_loss_epoch, loss_g_epoch = 0, 0
        loss_real_epoch, loss_fake_epoch, loss_d_epoch, loss_f_epoch, loss_env_epoch = 0, 0, 0, 0, 0
        loss_harm_pen_epoch = 0
        loss_mel_epoch = 0.0
        # sub-disc 群別の寄与（adv=generator を騙す側, d=disc real+fake）。
        # grp1=MSD, grp2=複素STFT or MPD, wn=WaveNet(自前 disc のみ)。ラベルは sub_labels。
        adv_msd_epoch, adv_grp2_epoch, adv_wn_epoch = 0.0, 0.0, 0.0
        d_msd_epoch, d_grp2_epoch, d_wn_epoch = 0.0, 0.0, 0.0
        grad_norm_d_epoch, n_disc_updates = 0.0, 0   # disc 勾配ノルム(clip前): MSD(or全体)分
        n_msd_skipped_epoch = 0                        # 群別ゲートで MSD 更新スキップ回数
        n_mpd_skipped_epoch = 0                        # 〃 MPD
        grad_norm_d_grp2_epoch = 0.0                 # 第2群(複素STFT/MPD)の勾配ノルム(分離時のみ集計)
        grad_norm_g_epoch, n_gen_updates = 0.0, 0    # generator 勾配ノルム(clip前): 停滞=DSP限界か勾配消失かの切り分け

        # accumulation 用: ループ前に zero_grad
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        n_batches = len(train_loader)

        for batch_idx, (f0, log_melspc, wav, uv, mask) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")):

            # accumulation boundary (always step on the last batch too)
            is_update_step = ((batch_idx + 1) % accum_steps == 0) or \
                             ((batch_idx + 1) == n_batches)

            f0 = torch.from_numpy(f0).float().to(device)
            log_melspc = torch.from_numpy(log_melspc).float().to(device)
            wav = torch.from_numpy(wav).float().to(device)
            uv = torch.from_numpy(uv).float().to(device)  # (B, 1, T_frames)
            mask = mask.to(device)

            # RefineGAN流 pitch augmentation(per-batch, GPU)。既定OFF/非発動でビット不変。
            # 発動時: wav をリサンプル→mel再計算→f0×r→uv/mask 時間ワープ(バッチ長が変動)。
            f0, log_melspc, wav, uv, mask = pitch_augment_batch(
                f0, log_melspc, wav, uv, mask, cfg,
                mel_sr, mel_nfft, mel_hop, mel_win, mel_dim, mel_fmin, mel_fmax)

            # v/uv dropout: 確率 vuv_dropout_prob で uv gate を無効化(全 voiced 扱い)。
            # 大半(1-p)は gate 有効で「無声区間で harmonic を鳴らさない」を学習(無声で
            # noise+harmonic 同時=過大音量→強い損失でharmonic抑制を獲得)。たまに(p)無効化で
            # F0抽出器の uv 誤判定(無声に有声混入)にロバスト化。batch 単位、train 時のみ。
            uv_model = uv
            if vuv_dropout_prob > 0 and random.random() < vuv_dropout_prob:
                uv_model = torch.zeros_like(uv)

            # mel condition for the (optional) mel-conditioned WaveNet disc.
            # log_melspc: [B, T_frames, 80] -> [B, 80, T_frames].
            # Built once here so it survives `del log_melspc` in the disc update.
            mel_cond = log_melspc.transpose(1, 2)

            if use_v3:
                # V3: harmonic を uv で hard gate。harmonic_penalty は不要(scale=0)。
                est_source = model(log_melspc, f0, uv_model, noise_std=effective_noise_std)
            elif harmonic_penalty_scale > 0:
                est_source, sig_harm, sig_noise = model.forward_train(
                    log_melspc, f0, noise_std=effective_noise_std)
            else:
                est_source = model(log_melspc, f0, noise_std=effective_noise_std)
            est_source = est_source.unsqueeze(1)
            wav = wav.unsqueeze(1)
            # mel_loss は masked_fill 前の生 est_source で計算する。padding を 0 埋めすると
            # mel(0)→log10(clamp,1e-5)→微分が巨大になり勾配を汚すため。mask は mel_loss 内で
            # 分母(valid フレーム数)に使う。stft/envelope 用には従来どおり 0 埋め版を使う。
            est_source_raw = est_source
            est_source = est_source.masked_fill(mask.unsqueeze(1), 0)
            wav = wav.masked_fill(mask.unsqueeze(1), 0)

            # Per-utterance RMS normalization for reconstruction losses
            # (prevents loud utterances from dominating STFT/envelope loss)
            valid = (~mask).unsqueeze(1).float()
            n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rms = ((wav.pow(2) * valid).sum(dim=-1, keepdim=True) / n_valid).sqrt().clamp(min=1e-8)
            wav_n = wav / rms
            est_n = est_source / rms
            del valid, n_valid, rms

            # ================================================================
            # SingingVocoders 流: (1) Discriminator を先に更新 → (2) Generator
            # ================================================================

            # ----------------------------------------------------------------
            # (1) Discriminator update (先)
            # ----------------------------------------------------------------
            if epoch > adversarial_start:
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    p_real = discriminator(wav, mel_cond)
                    p_fake = discriminator(est_source.detach(), mel_cond)
                    real_loss = 0.0
                    fake_loss = 0.0
                    msd_loss_term = 0.0   # MSD群の (real+fake) 合計テンソル(群別ゲート用)
                    mpd_loss_term = 0.0   # grp2(MPD)群の合計テンソル
                    _d_msd_sum = 0.0
                    _d_grp2_sum = 0.0
                    for ii in range(len(p_real)):
                        _rl = nn.MSELoss()(
                            p_real[ii][-1], p_real[ii][-1].new_ones(p_real[ii][-1].size())
                        )
                        _fl = nn.MSELoss()(
                            p_fake[ii][-1], p_fake[ii][-1].new_zeros(p_fake[ii][-1].size())
                        )
                        real_loss += _rl
                        fake_loss += _fl
                        if ii < n_msd_sub:                                 # 1群目(MSD)
                            msd_loss_term = msd_loss_term + _rl + _fl
                            _d_msd_sum += (_rl.item() + _fl.item())
                        elif n_msd_sub <= ii < n_msd_sub + n_grp2_sub:     # 2群目(MPD)
                            mpd_loss_term = mpd_loss_term + _rl + _fl
                            _d_grp2_sum += (_rl.item() + _fl.item())
                        elif use_wavenet and ii == len(p_real) - 1:        # 末尾(WaveNet, 自前のみ)
                            d_wn_epoch += (_rl.item() + _fl.item())
                    real_loss /= float(len(p_real))
                    fake_loss /= float(len(p_real))
                    if n_msd_sub > 0:
                        d_msd_epoch += _d_msd_sum / n_msd_sub
                    if n_grp2_sub > 0:
                        d_grp2_epoch += _d_grp2_sum / n_grp2_sub
                # --- 群別適応ゲート: MSD/MPD を別々に。各群の EMA d_loss が floor 割れ(=その群が勝ち過ぎ)なら
                #     その群だけ更新をスキップ(他群は更新)。スキップ群は backward に入らず grad=None→optimizer が更新しない。
                d_msd_cur = (_d_msd_sum / n_msd_sub) if n_msd_sub > 0 else 1.0
                d_mpd_cur = (_d_grp2_sum / n_grp2_sub) if n_grp2_sub > 0 else 1.0
                d_msd_ema = d_gate_ema * d_msd_ema + (1.0 - d_gate_ema) * d_msd_cur
                d_mpd_ema = d_gate_ema * d_mpd_ema + (1.0 - d_gate_ema) * d_mpd_cur
                do_msd = ((not adaptive_d_gate) or (d_msd_ema > d_gate_floor)) and n_msd_sub > 0
                do_mpd = ((not adaptive_d_gate) or (d_mpd_ema > d_gate_floor)) and n_grp2_sub > 0
                _dl = None
                if do_msd:
                    _dl = msd_loss_term
                if do_mpd:
                    _dl = mpd_loss_term if _dl is None else _dl + mpd_loss_term
                loss_real_epoch += real_loss.item()
                loss_fake_epoch += fake_loss.item()
                if not do_msd:
                    n_msd_skipped_epoch += 1
                if not do_mpd:
                    n_mpd_skipped_epoch += 1
                if _dl is not None:
                    discriminator_loss = _dl / float(len(p_real)) / accum_steps
                    loss_d_epoch += discriminator_loss.item() * accum_steps
                    # --- R1 勾配ペナルティ(r1_gamma>0時のみ。autocast外=fp32 で二重逆伝播, lazy, 失敗時 skip)---
                    if r1_gamma > 0 and (batch_idx % r1_interval == 0):
                        try:
                            wav_r1 = wav.detach().requires_grad_(True)
                            p_r1 = discriminator(wav_r1, mel_cond)
                            r1_logits = sum(o[-1].float().sum() for o in p_r1)
                            grad_r1 = torch.autograd.grad(r1_logits, wav_r1, create_graph=True)[0]
                            r1 = grad_r1.float().pow(2).flatten(1).sum(1).mean()
                            discriminator_loss = discriminator_loss + (0.5 * r1_gamma * r1_interval / accum_steps) * r1
                        except RuntimeError as _e:
                            if batch_idx == 0:
                                print(f"  [warn] R1 skipped: {type(_e).__name__}: {_e}")
                    scaler_d.scale(discriminator_loss).backward()
                    if is_update_step:
                        scaler_d.unscale_(optimizer_d)
                        if _disc_has_grp2:
                            # MSD と第2群(MPD)は別々に clip + gnorm 記録(スキップ群は grad=None→clip 0/更新なし)
                            _gn_d_grp2 = nn.utils.clip_grad_norm_(_grp2_module.parameters(), disc_grad_clip_grp2)
                            _other_d_params = [p for p in discriminator.parameters() if id(p) not in _grp2_ids_set]
                            _gn_d = nn.utils.clip_grad_norm_(_other_d_params, disc_grad_clip)
                            grad_norm_d_grp2_epoch += float(_gn_d_grp2)
                        else:
                            _gn_d = nn.utils.clip_grad_norm_(discriminator.parameters(), disc_grad_clip)
                        grad_norm_d_epoch += float(_gn_d); n_disc_updates += 1
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                        optimizer_d.zero_grad()
                    del discriminator_loss
                del p_real, p_fake, real_loss, fake_loss, msd_loss_term, mpd_loss_term

            # ----------------------------------------------------------------
            # (2) Generator update (後)
            # ----------------------------------------------------------------
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                # STFT loss(stft_loss_scale=0 で完全スキップ。NSF/HiFi-GAN 標準は multi-res STFT 不使用)
                if stft_loss_scale > 0:
                    stft_loss = stft_loss_fn(
                        est_n, wav_n, fft_lengths,
                        window_lengths, hop_lengths, stft_loss_type
                    )
                    if torch.isnan(stft_loss):
                        if epoch < nan_restart_until:
                            raise NaNDetected(epoch)
                        else:
                            raise NaNStop(epoch)
                    total_loss = stft_loss * stft_loss_scale / accum_steps
                    stft_loss_epoch += stft_loss.item()
                    del stft_loss
                else:
                    total_loss = torch.zeros((), device=est_source.device)

                # mel reconstruction loss (NSF-HiFiGAN流の主再構成 loss)。
                # masked_fill 前の est_source_raw を使う(padding 0埋めの log10 爆発を回避)。
                # use_amp=False(FP32)なら autocast は no-op。fp32 を明示して安全側に倒す。
                if mel_loss_scale > 0:
                    with torch.amp.autocast('cuda', enabled=False):
                        m_loss = mel_loss_fn(est_source_raw.float(), log_melspc.float(),
                                             mel_sr, mel_nfft, mel_hop, mel_win,
                                             mel_dim, mel_fmin, mel_fmax, mask=mask)
                    loss_mel_epoch += m_loss.item()
                    total_loss = total_loss + m_loss * mel_loss_scale / accum_steps
                    del m_loss

                if envelope_scale > 0 and epoch > envelope_start:
                    env_loss = envelope_loss_fn(
                        est_n, wav_n, envelope_kernel, envelope_stride,
                        directional=envelope_directional
                    )
                    loss_env_epoch += env_loss.item()
                    total_loss = total_loss + env_loss * envelope_scale / accum_steps
                    del env_loss

                if harmonic_penalty_scale > 0 and epoch > harmonic_penalty_start:
                    uv_resampled = uv.repeat_interleave(hop_size, dim=-1)
                    T = min(sig_harm.size(-1), uv_resampled.size(-1))
                    valid_mask = (~mask.unsqueeze(1)).float()[..., :T]
                    harm_pen = (sig_harm[..., :T] * uv_resampled[..., :T] * valid_mask).abs().mean()
                    loss_harm_pen_epoch += harm_pen.item()
                    total_loss = total_loss + harm_pen * harmonic_penalty_scale / accum_steps
                    del harm_pen, uv_resampled, valid_mask

                if epoch > adversarial_start:
                    discriminator.requires_grad_(False)
                    est_p = discriminator(est_source, mel_cond)
                    with torch.no_grad():
                        p = discriminator(wav, mel_cond)

                    if adversarial_warmup_epochs > 0:
                        adv_ramp = min(1.0, (epoch - adversarial_start) / adversarial_warmup_epochs)
                    else:
                        adv_ramp = 1.0

                    # adversarial loss (LSGAN, 全 sub-disc 合計 = HiFi-GAN/NSF流 論文式7 Σ_k。旧:平均は誤り)
                    adversarial_loss = 0.0
                    _adv_msd_sum = 0.0
                    _adv_grp2_sum = 0.0
                    for ii in range(len(est_p)):
                        _a = nn.MSELoss()(
                            est_p[ii][-1], est_p[ii][-1].new_ones(est_p[ii][-1].size())
                        )
                        _w = grp2_loss_weight if (n_msd_sub <= ii < n_msd_sub + n_grp2_sub) else 1.0
                        adversarial_loss += _a * _w                        # 勾配は重み付き、ログ(_adv_*_sum)は生値
                        if ii < n_msd_sub:                                 # 1群目(MSD)
                            _adv_msd_sum += _a.item()
                        elif n_msd_sub <= ii < n_msd_sub + n_grp2_sub:     # 2群目(複素STFT or MPD)
                            _adv_grp2_sum += _a.item()
                        elif use_wavenet and ii == len(est_p) - 1:         # 末尾(WaveNet, 自前のみ)
                            adv_wn_epoch += _a.item()
                    # NSF/HiFi-GAN は sub-disc を「合計」(論文 式7 Σ_k)。以前の平均(÷len)で adv が ~1/10 に弱体化していた→合計に修正
                    if n_msd_sub > 0:
                        adv_msd_epoch += _adv_msd_sum / n_msd_sub
                    if n_grp2_sub > 0:
                        adv_grp2_epoch += _adv_grp2_sum / n_grp2_sub
                    total_loss = total_loss + adversarial_loss * adversarial_scale * adv_ramp / accum_steps

                    # feature matching (生 L1 × feature_matching_scale = SingingVocoders流)
                    feature_map_loss = 0.0
                    n_pairs = 0
                    for jj, (real_fmaps, fake_fmaps) in enumerate(zip(p, est_p)):
                        _wfm = grp2_loss_weight if (n_msd_sub <= jj < n_msd_sub + n_grp2_sub) else 1.0
                        for real, fake in zip(real_fmaps[:-1], fake_fmaps[:-1]):
                            feature_map_loss += (fake - real.detach()).abs().mean() * _wfm
                            n_pairs += 1
                    # NSF/HiFi-GAN は全(sub-disc×層)を「合計」(HiFiloss.py feature_loss)。以前の平均(÷n_pairs~60)で FM が ~1/60 に弱体化→合計に修正
                    total_loss = total_loss + feature_map_loss * feature_matching_scale * adv_ramp / accum_steps
                    loss_f_epoch += feature_map_loss.item()

                    del adversarial_loss, feature_map_loss, est_p, p
                    discriminator.requires_grad_(True)

            loss_g_epoch += total_loss.item() * accum_steps
            scaler_g.scale(total_loss).backward()
            del total_loss

            if is_update_step:
                scaler_g.unscale_(optimizer_g)
                _gn_g = nn.utils.clip_grad_norm_(model.parameters(), gen_grad_clip)
                grad_norm_g_epoch += float(_gn_g); n_gen_updates += 1
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

            del est_source, est_source_raw, wav, wav_n, est_n, uv, mel_cond, log_melspc, f0, mask
            if harmonic_penalty_scale > 0:
                del sig_harm, sig_noise

        toc = time.time()

        print(
            'epoch', epoch, round(toc - tic, 2),
            'stft', round(stft_loss_epoch, 3),
            'loss_g', round(loss_g_epoch, 3),
            'loss_real', round(loss_real_epoch, 3),
            'loss_fake', round(loss_fake_epoch, 3),
            'loss_d', round(loss_d_epoch, 3),
            'loss_f', round(loss_f_epoch, 3),
            'loss_mel', round(loss_mel_epoch, 3),
            'loss_env', round(loss_env_epoch, 3),
            'loss_harm_pen', round(loss_harm_pen_epoch, 5),
            '| adv_' + sub_labels[0], round(adv_msd_epoch / max(1, n_batches), 3),
            'adv_' + sub_labels[1], round(adv_grp2_epoch / max(1, n_batches), 3),
            *(('adv_' + sub_labels[2], round(adv_wn_epoch / max(1, n_batches), 3)) if sub_labels[2] else ()),
            '| d_' + sub_labels[0], round(d_msd_epoch / max(1, n_batches), 3),
            'd_' + sub_labels[1], round(d_grp2_epoch / max(1, n_batches), 3),
            *(('d_' + sub_labels[2], round(d_wn_epoch / max(1, n_batches), 3)) if sub_labels[2] else ()),
            '| gnorm_d', round(grad_norm_d_epoch / max(1, n_disc_updates), 5),
            *(('gnorm_d_' + _grp2_label, round(grad_norm_d_grp2_epoch / max(1, n_disc_updates), 5)) if _disc_has_grp2 else ()),
            'gnorm_g', round(grad_norm_g_epoch / max(1, n_gen_updates), 5),
            *(('| d_msd_ema', round(d_msd_ema, 3), 'msd_skip', f'{n_msd_skipped_epoch}/{n_batches}',
               'd_mpd_ema', round(d_mpd_ema, 3), 'mpd_skip', f'{n_mpd_skipped_epoch}/{n_batches}') if adaptive_d_gate else ()),
        )

        n = len(train_loader)
        writer.add_scalar('train/stft', stft_loss_epoch / n, epoch)
        writer.add_scalar('train/loss_g', loss_g_epoch / n, epoch)
        writer.add_scalar('train/loss_real', loss_real_epoch / n, epoch)
        writer.add_scalar('train/loss_fake', loss_fake_epoch / n, epoch)
        writer.add_scalar('train/loss_d', loss_d_epoch / n, epoch)
        writer.add_scalar('train/loss_f', loss_f_epoch / n, epoch)
        # sub-disc 群別の寄与（adv=generator側, d=disc側 real+fake）。ラベルは disc 種別で切替。
        # HiFi-GAN: msd/mpd、自前 disc: msd/stft/wn。
        writer.add_scalar(f'train/adv_{sub_labels[0]}', adv_msd_epoch / n, epoch)
        writer.add_scalar(f'train/adv_{sub_labels[1]}', adv_grp2_epoch / n, epoch)
        writer.add_scalar(f'train/d_{sub_labels[0]}', d_msd_epoch / n, epoch)
        writer.add_scalar(f'train/d_{sub_labels[1]}', d_grp2_epoch / n, epoch)
        if sub_labels[2]:
            writer.add_scalar(f'train/adv_{sub_labels[2]}', adv_wn_epoch / n, epoch)
            writer.add_scalar(f'train/d_{sub_labels[2]}', d_wn_epoch / n, epoch)
        if mel_loss_scale > 0:
            writer.add_scalar('train/loss_mel', loss_mel_epoch / n, epoch)
        if envelope_scale > 0:
            writer.add_scalar('train/loss_env', loss_env_epoch / n, epoch)
        if harmonic_penalty_scale > 0:
            writer.add_scalar('train/loss_harm_pen', loss_harm_pen_epoch / n, epoch)

        eval_loss, eval_env_loss = evaluate(model, test_list, stft_loss_fn, cfg, device)
        writer.add_scalar('test/stft', eval_loss, epoch)
        if envelope_scale > 0:
            writer.add_scalar('test/loss_env', eval_env_loss, epoch)

        if epoch % cfg['training']['save_interval'] == 0:
            save_path = snapshot_dir / f"{epoch:06d}epoch.pth"
            save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, save_path)
            print(f"Saved model at {save_path}")
            inference_test_data(model, device, writer, epoch, cfg, logged_real_mels, discriminator)
            if pinpoint_writer is not None:
                inference_pinpoint_files(model, device, pinpoint_writer, epoch, cfg, pinpoint_logged_reals, discriminator)

        # epoch 末に CUDA キャッシュを解放（可変 segment 長による reserved の累積=断片化を抑える）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    if pinpoint_writer is not None:
        pinpoint_writer.close()
    print("Training finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NHVSing V3 training script (MRD+MPD GAN)")
    parser.add_argument("--config", type=str, default="config_v3.yaml")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--finetune_from", type=str, default=None)
    args = parser.parse_args()

    # 同期確認用バージョン。train.py を変更したらこの文字列を更新すること。
    print("=" * 70)
    print(">>> train.py VERSION: 2026-06-24e  (bf16 mixed[RTX PRO 6000] / padding除外 / clamp1e-5 / gen_grad_clip=100)")
    print("=" * 70)

    MAX_NAN_RESTARTS = 5
    force_restart = False
    for nan_count in range(MAX_NAN_RESTARTS + 1):
        try:
            run(args, force_restart=force_restart)
            break
        except NaNStop as e:
            print(f"[NaN] epoch {e.epoch} でNaN検出 → 学習を停止します。")
            raise
        except NaNDetected as e:
            if nan_count >= MAX_NAN_RESTARTS:
                print(f"[NaN Auto-Restart] {MAX_NAN_RESTARTS}回再スタートしてもNaNが続くため停止します。")
                raise
            print(f"\n[NaN Auto-Restart] ({nan_count + 1}/{MAX_NAN_RESTARTS})"
                  f" epoch {e.epoch} でNaN検出 → epoch 0 から再スタートします...")
            cfg = load_config(args.config)
            snapshot_dir = Path(cfg['training']['snapshot_dir'])
            for pth in sorted(snapshot_dir.glob("*.pth")):
                pth.unlink()
                print(f"  削除: {pth}")
            force_restart = True
            print()


if __name__ == "__main__":
    main()
