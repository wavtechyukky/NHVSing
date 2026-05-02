import os
import argparse
import time
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

from dataset import VocoderDataset, collate_fn_padd, norm_interp_f0
from model import NHVSing, NHVSingV2
from discriminator import DiscriminatorWithComplexSTFT
from dsp import stft_loss as stft_loss_fn, envelope_loss as envelope_loss_fn
import glob


def make_capped_collate(max_frames, hop_size: int):
    if max_frames is None:
        return collate_fn_padd

    def capped_collate(batch):
        capped = []
        for (f0, melspc, wav, uv) in batch:
            T = min(max_frames, f0.shape[1])
            capped.append((
                f0[:, :T],
                melspc[:T, :],
                wav[:T * hop_size],
                uv[:, :T],
            ))
        return collate_fn_padd(capped)

    return capped_collate


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
    envelope_scale = cfg['training'].get('envelope_scale', 0.0)
    hop_size = cfg['preprocess']['hop_size']
    envelope_kernel = cfg['training'].get('envelope_kernel_size', hop_size * 2)
    envelope_stride = cfg['training'].get('envelope_stride', hop_size)

    with torch.no_grad():
        for f0, log_melspc, wav, _, mask in test_loader:
            f0 = torch.from_numpy(f0).float().to(device)
            log_melspc = torch.from_numpy(log_melspc).float().to(device)
            wav = torch.from_numpy(wav).float().to(device)
            mask = mask.to(device)

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
                                window_lengths, hop_lengths, 'log_linear')
            stft_loss_eval += stft_loss.item()
            del stft_loss

            if envelope_scale > 0:
                env_loss = envelope_loss_fn(output, wav_for_loss, envelope_kernel, envelope_stride)
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


def inference(model, npz_path, device, cfg):
    model.eval()

    with torch.no_grad():
        data = np.load(npz_path)
        f0_np, _ = norm_interp_f0(data['f0'])
        f0_np = f0_np[np.newaxis][np.newaxis]
        log_melspc_np = data['log_melspc'][np.newaxis]
        real_mel = data['log_melspc']
        del data

        f0 = torch.Tensor(f0_np).to(device)
        log_melspc = torch.Tensor(log_melspc_np).to(device)
        del f0_np, log_melspc_np

        synthesized_tensor = model(log_melspc, f0)
        del f0, log_melspc

        synthesized = torch.squeeze(synthesized_tensor).to('cpu').detach().numpy().copy()
        del synthesized_tensor

    p_cfg = cfg['preprocess']
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


def inference_pinpoint_files(model, device, writer, epoch, cfg, logged_real_mels: set):
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
                                               vmin=-80, vmax=0), 0)
            real_wav = np.load(npz_path)['wav']
            writer.add_audio('audio/' + basename + '/real',
                             torch.from_numpy(real_wav).unsqueeze(0), 0, sample_rate)
            logged_real_mels.add(basename)

        writer.add_figure('plot/' + basename + '/fake',
                          plot_spectrogram(fake_mel.T, title=f"fake: {basename}  ep{epoch}",
                                           vmin=-80, vmax=0), epoch)
        writer.add_audio('audio/' + basename + '/fake',
                         torch.from_numpy(wav).unsqueeze(0), epoch, sample_rate)

        min_T = min(real_mel.shape[0], fake_mel.shape[0])
        diff = fake_mel[:min_T] - real_mel[:min_T]  # (T, mel_dim)
        writer.add_figure('mel_diff/' + basename,
                          plot_mel_diff(diff.T, title=f"diff: {basename}  ep{epoch}"), epoch)


def inference_test_data(model, device, writer, epoch, cfg, logged_real_mels: set):
    test_data_folder = cfg['training']['test_dir']
    all_npz_paths = sorted(glob.glob(os.path.join(test_data_folder, '**/*.npz'), recursive=True))

    n_per_singer = cfg['training'].get('tb_files_per_singer', 0)
    if n_per_singer > 0:
        all_npz_paths = _select_tb_files(all_npz_paths, n_per_singer)

    inference_output_base_dir = Path(cfg['training'].get('inference_output_dir', 'dataset/inference'))
    save_dir = inference_output_base_dir / f"{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = cfg['preprocess']['sample_rate']
    for i, npz_path in enumerate(all_npz_paths):
        wav, real_mel, fake_mel = inference(model, npz_path, device, cfg)

        basename = os.path.splitext(os.path.basename(npz_path))[0]

        # real_mel / ground truth audio は固定なので初回のみ記録
        if basename not in logged_real_mels:
            writer.add_figure('plot/' + basename + '/real',
                              plot_spectrogram(real_mel.T, title=f"real: {basename}",
                                               vmin=-80, vmax=0), 0)
            real_wav = np.load(npz_path)['wav']
            writer.add_audio('audio/' + basename + '/real',
                             torch.from_numpy(real_wav).unsqueeze(0), 0, sample_rate)
            logged_real_mels.add(basename)

        writer.add_figure('plot/' + basename + '/fake',
                          plot_spectrogram(fake_mel.T, title=f"fake: {basename}  ep{epoch}",
                                           vmin=-80, vmax=0), epoch)
        writer.add_audio('audio/' + basename + '/fake',
                         torch.from_numpy(wav).unsqueeze(0), epoch, sample_rate)

        min_T = min(real_mel.shape[0], fake_mel.shape[0])
        diff = fake_mel[:min_T] - real_mel[:min_T]  # (T, mel_dim)
        writer.add_figure('mel_diff/' + basename,
                          plot_mel_diff(diff.T, title=f"diff: {basename}  ep{epoch}"), epoch)

        save_path = save_dir / f"{i:03d}.wav"
        sf.write(save_path, wav, sample_rate)


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
    use_v2 = ltv_filter_cfg.get('use_shared_trunk', False)
    ModelClass = NHVSingV2 if use_v2 else NHVSing
    print(f"Model: {ModelClass.__name__}")
    model = ModelClass(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=ltv_filter_cfg,
    ).to(device)

    disc_cfg = cfg.get('discriminator', {})
    use_msd = disc_cfg.get('use_msd', True)
    stft_filters = disc_cfg.get('stft_filters', 32)
    discriminator = DiscriminatorWithComplexSTFT(
        use_msd=use_msd, stft_filters=stft_filters
    ).to(device)
    print(f"DiscriminatorWithComplexSTFT: use_msd={use_msd}, stft_filters={stft_filters}")

    optimizer_g = torch.optim.RAdam(model.parameters(), lr=cfg['training']['lr_g'], eps=1e-4)
    optimizer_d = torch.optim.RAdam(discriminator.parameters(), lr=cfg['training']['lr_d'])

    use_amp = cfg['training'].get('use_amp', False) and device.type == 'cuda'
    scaler_g = torch.amp.GradScaler('cuda', enabled=use_amp)
    scaler_d = torch.amp.GradScaler('cuda', enabled=use_amp)
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
    if amp_augment:
        print(f"amp_augment: enabled, range={amp_aug_range}")
    train_dataset = VocoderDataset(dataset_dir=cfg['training']['train_dir'], hop_size=hop_size,
                                   augment=amp_augment, amp_aug_range=amp_aug_range)
    max_train_frames = cfg['training'].get('max_train_frames', None)
    collate_train = make_capped_collate(max_train_frames, hop_size=hop_size)
    if max_train_frames:
        print(f"max_train_frames={max_train_frames}: VRAM 上限を固定")

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
    test_dataset = VocoderDataset(dataset_dir=cfg['training']['test_dir'], hop_size=hop_size)
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
            discriminator.load_state_dict(snapshot['discriminator'])
            optimizer_g.load_state_dict(snapshot['optimizer_g'])
            optimizer_d.load_state_dict(snapshot['optimizer_d'])
            start_epoch = snapshot['epoch'] + 1
            print(f"Starting from epoch {start_epoch}")

        elif args.finetune_from:
            print(f"Fine-tuning from: {args.finetune_from}")
            snapshot = torch.load(args.finetune_from, map_location=device)
            model.load_state_dict(snapshot['model'])
            print("  Generator weights loaded.")
            if 'optimizer_g' in snapshot:
                optimizer_g.load_state_dict(snapshot['optimizer_g'])
                print("  optimizer_g state loaded (Adam momentum preserved).")
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
    adversarial_scale = cfg['training']['adversarial_scale']
    print("adversarial_scale:", adversarial_scale)
    feature_matching_scale = cfg['training']['feature_matching_scale']
    adversarial_warmup_epochs = cfg['training'].get('adversarial_warmup_epochs', 0)
    envelope_scale = cfg['training'].get('envelope_scale', 0.0)
    envelope_start = cfg['training'].get('envelope_start', 0)
    envelope_kernel = cfg['training'].get('envelope_kernel_size', hop_size * 2)
    envelope_stride = cfg['training'].get('envelope_stride', hop_size)
    if envelope_scale > 0:
        print(f"envelope_loss: scale={envelope_scale}, start={envelope_start}, kernel={envelope_kernel}, stride={envelope_stride}")
    window_lengths = cfg['training']['window_lengths']
    fft_lengths = [int(2 * i) for i in window_lengths]
    hop_lengths = [int(i / 4) for i in window_lengths]
    nan_restart_until = cfg['training'].get('nan_restart_until_epoch', 30)

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

            if harmonic_penalty_scale > 0:
                est_source, sig_harm, sig_noise = model.forward_train(
                    log_melspc, f0, noise_std=effective_noise_std)
            else:
                est_source = model(log_melspc, f0, noise_std=effective_noise_std)
            est_source = est_source.unsqueeze(1)
            wav = wav.unsqueeze(1)
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

            # ----------------------------------------------------------------
            # Generator update
            # ----------------------------------------------------------------
            with torch.amp.autocast('cuda', enabled=use_amp):
                stft_loss = stft_loss_fn(
                    est_n, wav_n, fft_lengths,
                    window_lengths, hop_lengths, 'log_linear'
                )
                if torch.isnan(stft_loss):
                    if epoch < nan_restart_until:
                        raise NaNDetected(epoch)
                    else:
                        raise NaNStop(epoch)

                # accumulation: divide loss by accum_steps to normalize gradients
                total_loss = stft_loss / accum_steps
                stft_loss_epoch += stft_loss.item()  # log at original scale
                del stft_loss

                if envelope_scale > 0 and epoch > envelope_start:
                    env_loss = envelope_loss_fn(
                        est_n, wav_n, envelope_kernel, envelope_stride
                    )
                    loss_env_epoch += env_loss.item()
                    total_loss = total_loss + env_loss * envelope_scale / accum_steps
                    del env_loss

                if harmonic_penalty_scale > 0 and epoch > harmonic_penalty_start:
                    # uv: (B, 1, T_frames) -> (B, 1, T_samples)
                    uv_resampled = uv.repeat_interleave(hop_size, dim=-1)
                    T = min(sig_harm.size(-1), uv_resampled.size(-1))
                    # Exclude padding regions (mask: True = padding)
                    valid_mask = (~mask.unsqueeze(1)).float()[..., :T]
                    # L1 norm: sig_harm is ~1e-4 scale; pow(2) would give ~1e-8 gradients.
                    # abs().mean() keeps values intact; gradient = sign(sig_harm) = ±1.
                    harm_pen = (sig_harm[..., :T] * uv_resampled[..., :T] * valid_mask).abs().mean()
                    loss_harm_pen_epoch += harm_pen.item()
                    total_loss = total_loss + harm_pen * harmonic_penalty_scale / accum_steps
                    del harm_pen, uv_resampled, valid_mask

                if epoch > adversarial_start:
                    discriminator.requires_grad_(False)
                    est_p = discriminator(est_source)

                    adversarial_loss = 0.0
                    for ii in range(len(est_p)):
                        adversarial_loss += nn.MSELoss()(
                            est_p[ii][-1], est_p[ii][-1].new_ones(est_p[ii][-1].size())
                        )
                    adversarial_loss /= float(len(est_p))

                    if adversarial_warmup_epochs > 0:
                        adv_ramp = min(1.0, (epoch - adversarial_start) / adversarial_warmup_epochs)
                    else:
                        adv_ramp = 1.0
                    total_loss = total_loss + adversarial_loss * adversarial_scale * adv_ramp / accum_steps

                    with torch.no_grad():
                        p = discriminator(wav)
                    feature_map_loss = 0.0
                    n_pairs = 0
                    for real_fmaps, fake_fmaps in zip(p, est_p):
                        for real, fake in zip(real_fmaps[:-1], fake_fmaps[:-1]):
                            ref = real.detach()
                            feature_map_loss += (fake - ref).abs().mean() / ref.abs().mean().clamp(min=1e-8)
                            n_pairs += 1
                    feature_map_loss /= n_pairs
                    total_loss = total_loss + feature_map_loss * feature_matching_scale * adv_ramp / accum_steps
                    loss_f_epoch += feature_map_loss.item()

                    del adversarial_loss, feature_map_loss, est_p, p
                    discriminator.requires_grad_(True)

            loss_g_epoch += total_loss.item() * accum_steps  # logging は元のスケール
            scaler_g.scale(total_loss).backward()
            del total_loss

            if is_update_step:
                scaler_g.unscale_(optimizer_g)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

            # ----------------------------------------------------------------
            # Discriminator update
            # ----------------------------------------------------------------
            if epoch > adversarial_start:
                del log_melspc, f0, mask

                with torch.amp.autocast('cuda', enabled=use_amp):
                    p = discriminator(wav)
                    est_p_for_d = discriminator(est_source.detach())
                    real_loss = 0.0
                    fake_loss = 0.0
                    for ii in range(len(p)):
                        real_loss += nn.MSELoss()(
                            p[ii][-1], p[ii][-1].new_ones(p[ii][-1].size())
                        )
                        fake_loss += nn.MSELoss()(
                            est_p_for_d[ii][-1],
                            est_p_for_d[ii][-1].new_zeros(est_p_for_d[ii][-1].size())
                        )
                    real_loss /= float(len(p))
                    fake_loss /= float(len(p))
                    discriminator_loss = (real_loss + fake_loss) / accum_steps

                loss_real_epoch += real_loss.item()
                loss_fake_epoch += fake_loss.item()
                loss_d_epoch += discriminator_loss.item() * accum_steps  # logging は元のスケール
                scaler_d.scale(discriminator_loss).backward()

                if is_update_step:
                    scaler_d.unscale_(optimizer_d)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()

                del est_source, p, est_p_for_d, real_loss, fake_loss, discriminator_loss
            else:
                del est_source, log_melspc, f0, mask
            del wav, wav_n, est_n, uv
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
            'loss_env', round(loss_env_epoch, 3),
            'loss_harm_pen', round(loss_harm_pen_epoch, 5),
        )

        n = len(train_loader)
        writer.add_scalar('train/stft', stft_loss_epoch / n, epoch)
        writer.add_scalar('train/loss_g', loss_g_epoch / n, epoch)
        writer.add_scalar('train/loss_real', loss_real_epoch / n, epoch)
        writer.add_scalar('train/loss_fake', loss_fake_epoch / n, epoch)
        writer.add_scalar('train/loss_d', loss_d_epoch / n, epoch)
        writer.add_scalar('train/loss_f', loss_f_epoch / n, epoch)
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
            inference_test_data(model, device, writer, epoch, cfg, logged_real_mels)
            if pinpoint_writer is not None:
                inference_pinpoint_files(model, device, pinpoint_writer, epoch, cfg, pinpoint_logged_reals)

    writer.close()
    if pinpoint_writer is not None:
        pinpoint_writer.close()
    print("Training finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NHVSing training script")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--finetune_from", type=str, default=None)
    args = parser.parse_args()

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
