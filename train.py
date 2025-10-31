import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import soundfile as sf

from dataset import VocoderDataset, collate_fn_padd, norm_interp_f0
from model import NHVSing
from discriminator import Discriminator
from dsp import stft_loss as stft_loss_fn
from dsp import frame_center_log_mel_spectrogram
import glob

# --- Helper Functions ---

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

def evaluate(model, test_loader, loss_fn, cfg, device, writer, epoch):
    model.eval()
    stft_loss_eval = 0
    
    frame_size = cfg['training']['frame_size']
    noise_std = cfg['model']['vocoder']['noise_std']
    window_lengths = cfg['training']['window_lengths']
    fft_lengths = [int(2 * i) for i in window_lengths]
    hop_lengths = [int(i / 4) for i in window_lengths]

    with torch.no_grad():
        i = 0
        for f0, log_melspc, wav, _, mask in test_loader:
            f0 = torch.from_numpy(f0).to(device)
            log_melspc = torch.from_numpy(log_melspc).to(device)
            wav = torch.from_numpy(wav).to(device)
            mask = mask.to(device)

            output = model(log_melspc, f0).masked_fill(mask, 0)
            
            est_source = torch.masked_select(output, torch.logical_not(mask)).view(1, 1, -1)
            wav_for_loss = torch.masked_select(wav, torch.logical_not(mask)).view(1, 1, -1)
            
            stft_loss = loss_fn(
                est_source, wav_for_loss, fft_lengths,
                window_lengths, hop_lengths, 'log_linear'
            )
            stft_loss_eval += stft_loss.item()

            output = torch.squeeze(output).to('cpu').detach().numpy().copy()
            writer.add_audio('test/' + str(i), output, global_step=epoch, sample_rate=int(44100))
            i += 1

    model.train()
    return stft_loss_eval / len(test_loader)


def inference(model, npz_path, device, writer, epoch, cfg):
    model.eval()

    with torch.no_grad():
        data = np.load(npz_path)

        f0, _ = norm_interp_f0(data['f0'])
        f0 = f0[np.newaxis][np.newaxis]
        log_melspc = data['log_melspc'][np.newaxis]

        f0 = torch.Tensor(f0).to(device)
        log_melspc = torch.Tensor(log_melspc).to(device)

        synthesized = model(log_melspc, f0)
        synthesized = torch.squeeze(synthesized).to('cpu').detach().numpy().copy()

    model.train()
    return synthesized

def inference_test_data(model, device, writer, epoch, cfg):
    test_data_folder = cfg['training']['test_dir']
    test_npz_path_list = glob.glob(os.path.join(test_data_folder, '**/*.npz'), recursive=True)
    
    # Create save directory
    save_dir = Path(f"dataset/inference/{epoch}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, test_npz_path in enumerate(test_npz_path_list):
        wav = inference(model, test_npz_path, device, writer, epoch, cfg)

        # Save the audio file
        save_path = save_dir / f"{i:03d}.wav"
        sf.write(save_path, wav, 44100)


# --- Main Execution Function ---

def run(args):
    cfg = load_config(args.config)
    log_dir = Path(cfg['training']['log_dir'])
    snapshot_dir = Path(cfg['training']['snapshot_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NHVSing(
        vocoder_cfg=cfg['model']['vocoder'],
        ltv_filter_cfg=cfg['model']['ltv_filter'],
    ).to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_g = torch.optim.RAdam(model.parameters(), lr=cfg['training']['lr_g'])
    optimizer_d = torch.optim.RAdam(discriminator.parameters(), lr=cfg['training']['lr_d'])
    
    train_dataset = VocoderDataset(
        dataset_dir=cfg['training']['train_dir']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_padd,
        drop_last=True, pin_memory=True
    )
    test_dataset = VocoderDataset(
        dataset_dir=cfg['training']['test_dir']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg['training']['num_workers'], collate_fn=collate_fn_padd
    )
    test_list= list(test_loader)

    start_epoch = 0
    if args.resume_path:
        print(f"Resuming from checkpoint: {args.resume_path}")
        snapshot = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(snapshot['model'])
        discriminator.load_state_dict(snapshot['discriminator'])
        optimizer_g.load_state_dict(snapshot['optimizer_g'])
        optimizer_d.load_state_dict(snapshot['optimizer_d'])
        start_epoch = snapshot['epoch'] + 1
        print(f"Starting from epoch {start_epoch}")
    else:
        print("Starting new training.")

    frame_size = cfg['training']['frame_size']
    noise_std = cfg['model']['vocoder']['noise_std']
    adversarial_start = cfg['training']['adversarial_start']
    adversarial_scale = cfg['training']['adversarial_scale']
    feature_matching_scale = cfg['training']['feature_matching_scale']
    window_lengths = cfg['training']['window_lengths']
    fft_lengths = [int(2*i) for i in window_lengths]
    hop_lengths = [int(i/4) for i in window_lengths]
    
    for epoch in range(start_epoch, cfg['training']['n_epoch']):
        tic = time.time()
        stft_loss_epoch, loss_g_epoch = 0, 0
        loss_real_epoch, loss_fake_epoch, loss_d_epoch, loss_f_epoch = 0, 0, 0, 0

        for f0, log_melspc, wav, _, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            f0 = torch.from_numpy(f0).to(device)
            log_melspc = torch.from_numpy(log_melspc).to(device)
            wav = torch.from_numpy(wav).to(device)
            mask = mask.to(device)
            
            est_source = model(log_melspc, f0)
            est_source = est_source.masked_fill(mask.unsqueeze(1), 0)
            
            est_source = torch.masked_select(est_source, torch.logical_not(mask)).view(1, 1, -1)
            wav = torch.masked_select(wav, torch.logical_not(mask)).view(1, 1, -1)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            
            total_loss = 0.
            stft_loss = stft_loss_fn(
                est_source, wav, fft_lengths,
                window_lengths, hop_lengths, 'log_linear'
            )
            total_loss = total_loss + stft_loss
            stft_loss_epoch += stft_loss.item()
            
            if epoch > adversarial_start:
                est_p = discriminator(est_source)
                adversarial_loss = 0.0
                for ii in range(len(est_p)):
                    adversarial_loss += nn.MSELoss()(est_p[ii][-1], est_p[ii][-1].new_ones(est_p[ii][-1].size()))
                adversarial_loss /= float(len(est_p))
                total_loss = total_loss + adversarial_loss * adversarial_scale
                
                with torch.no_grad():
                    p = discriminator(wav)
                feature_map_loss = 0.0
                for ii in range(len(est_p)):
                    for jj in range(len(est_p[ii]) - 1):
                        feature_map_loss += nn.L1Loss()(est_p[ii][jj], p[ii][jj].detach())
                feature_map_loss /= (float(len(est_p)) * float(len(est_p[0]) - 1))
                total_loss = total_loss + feature_map_loss * feature_matching_scale
                loss_f_epoch += feature_map_loss.item()
            
            loss_g_epoch += total_loss.item()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_g.step()

            if epoch > adversarial_start:
                optimizer_d.zero_grad()
                with torch.no_grad():
                    est_source_for_d = model(log_melspc, f0)
                p = discriminator(wav)
                est_p_for_d = discriminator(est_source_for_d.unsqueeze(1).detach())
                real_loss = 0.0
                fake_loss = 0.0
                for ii in range(len(p)):
                    real_loss += nn.MSELoss()(p[ii][-1], p[ii][-1].new_ones(p[ii][-1].size()))
                    fake_loss += nn.MSELoss()(est_p_for_d[ii][-1], est_p_for_d[ii][-1].new_zeros(est_p_for_d[ii][-1].size()))
                real_loss /= float(len(p))
                fake_loss /= float(len(p))
                discriminator_loss = real_loss + fake_loss
                loss_real_epoch += real_loss.item()
                loss_fake_epoch += fake_loss.item()
                loss_d_epoch += discriminator_loss.item()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_d.step()
        
        toc = time.time()

        writer.add_scalar('train/stft', stft_loss_epoch/2, epoch)
        writer.add_scalar('train/loss_g', loss_g_epoch/2, epoch)
        writer.add_scalar('train/loss_real', loss_real_epoch/2, epoch)
        writer.add_scalar('train/loss_fake', loss_fake_epoch/2, epoch)
        writer.add_scalar('train/loss_d', loss_d_epoch/2, epoch)
        writer.add_scalar('train/loss_f', loss_f_epoch/2, epoch)

        print(
            'epoch', epoch, round(toc-tic, 2),
            'stft', round(stft_loss_epoch, 3),
            'loss_g', round(loss_g_epoch, 3),
            'loss_real', round(loss_real_epoch, 3),
            'loss_fake', round(loss_fake_epoch, 3),
            'loss_d', round(loss_d_epoch, 3),
            'loss_f', round(loss_f_epoch, 3)
        )


        # Write to TensorBoard
        writer.add_scalar('train/stft', stft_loss_epoch / len(train_loader), epoch)
        writer.add_scalar('train/loss_g', loss_g_epoch / len(train_loader), epoch)
        writer.add_scalar('train/loss_d', loss_d_epoch / len(train_loader), epoch)
        writer.add_scalar('train/loss_f', loss_f_epoch / len(train_loader), epoch)
        
        # Evaluation
        eval_loss = evaluate(model, test_list, stft_loss_fn, cfg, device, writer, epoch)
        writer.add_scalar('test/stft', eval_loss, epoch)
        
        # Save model
        if epoch % cfg['training']['save_interval'] == 0:
            save_path = snapshot_dir / f"{epoch:06d}epoch.pth"
            save_checkpoint(model, discriminator, optimizer_g, optimizer_d, epoch, save_path)
            print(f"Saved model at {save_path}")
            inference_test_data(model, device, writer, epoch, cfg)

    writer.close()
    print("Training finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to the checkpoint to resume from.")
    args = parser.parse_args()
    
    run(args)

if __name__ == "__main__":
    main()
