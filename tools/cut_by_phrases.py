"""
cut_by_phrases.py

音声ファイルをフレーズ単位でカットするスクリプト。
pydub の split_on_silence（静音ごとにぶつ切り）に替わる実装。

■ アルゴリズム:
  1. -50dBFS 未満が min_silence_dur 秒以上続く区間 → 「静音区間」
     それより短い無音（スタッカートの息継ぎ等）→「有音区間」の一部として扱う

  2. 有音区間からスタートし、以下の条件を満たす限り
     「静音区間 + 有音区間」を追加していく:
       - 追加後の合計が max_dur 秒以内
       - 静音区間が long_silence 秒未満

  3. 条件を超えたら、直前の有音区間の終わりまでを 1 セグメントとする

  4. セグメントの前後に pad 秒の余白を追加する

■ 使い方:
  # 基本
  python tools/cut_by_phrases.py \
    --src dataset/kiritan_wav_44100 \
    --dst dataset/kiritan_cut_new

  # パラメータ調整
  python tools/cut_by_phrases.py \
    --src dataset/kiritan_wav_44100 \
    --dst dataset/kiritan_cut_new \
    --silence_thresh -45 \
    --min_silence_dur 0.08 \
    --max_dur 9.0 \
    --long_silence 1.0 \
    --pad 0.1 \
    --dry_run
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

@dataclass
class Region:
    start: int   # samples
    end: int     # samples
    is_silence: bool

    @property
    def n_samples(self) -> int:
        return self.end - self.start


def detect_regions(
    audio: np.ndarray,
    sr: int,
    silence_thresh_db: float = -50.0,
    min_silence_dur: float = 0.10,
    frame_dur: float = 0.01,          # 10ms フレーム
) -> List[Region]:
    """
    音声を「静音区間」と「有音区間」に分割して返す。

    min_silence_dur 秒未満の静音は有音区間として扱う（スタッカート対策）。
    """
    frame_size = max(1, int(sr * frame_dur))
    n_total = len(audio)
    min_silence_frames = max(1, round(min_silence_dur / frame_dur))

    # --- フレームごとの dBFS ---
    silent_flags: List[bool] = []
    pos = 0
    while pos < n_total:
        frame = audio[pos: pos + frame_size]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20.0 * np.log10(max(rms, 1e-10))
        silent_flags.append(db < silence_thresh_db)
        pos += frame_size

    n_frames = len(silent_flags)
    if n_frames == 0:
        return []

    # --- 連続するフレームを raw region にまとめる ---
    raw: List[Tuple[int, int, bool]] = []   # (start_frame, end_frame, is_silence)
    seg_start = 0
    for i in range(1, n_frames + 1):
        if i == n_frames or silent_flags[i] != silent_flags[seg_start]:
            raw.append((seg_start, i, silent_flags[seg_start]))
            seg_start = i

    # --- 短い静音 → 有音に変換 ---
    merged: List[Tuple[int, int, bool]] = []
    for (sf_i, ef_i, is_sil) in raw:
        if is_sil and (ef_i - sf_i) < min_silence_frames:
            merged.append((sf_i, ef_i, False))   # 短い静音 → 有音
        else:
            merged.append((sf_i, ef_i, is_sil))

    # --- 隣接する同種 region を再統合 ---
    coalesced: List[Tuple[int, int, bool]] = []
    for item in merged:
        if coalesced and coalesced[-1][2] == item[2]:
            coalesced[-1] = (coalesced[-1][0], item[1], item[2])
        else:
            coalesced.append(list(item))   # mutable にする

    # --- フレームインデックス → サンプル位置に変換 ---
    regions: List[Region] = []
    for idx, (sf_i, ef_i, is_sil) in enumerate(coalesced):
        start_s = sf_i * frame_size
        end_s   = ef_i * frame_size if idx < len(coalesced) - 1 else n_total
        end_s   = min(end_s, n_total)
        if start_s < end_s:
            regions.append(Region(start=start_s, end=end_s, is_silence=is_sil))

    return regions


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def build_segments(
    regions: List[Region],
    sr: int,
    max_dur: float = 9.0,
    long_silence: float = 1.0,
    pad: float = 0.10,
    total_samples: int = 0,
) -> List[Tuple[int, int]]:
    """
    Region リストからセグメント (start_sample, end_sample) のリストを作る。
    """
    max_samples          = int(max_dur * sr)
    long_silence_samples = int(long_silence * sr)
    pad_samples          = int(pad * sr)

    segments: List[Tuple[int, int]] = []
    i = 0

    # 先頭の静音をスキップ
    while i < len(regions) and regions[i].is_silence:
        i += 1

    while i < len(regions):
        # 有音区間からスタート
        if regions[i].is_silence:
            i += 1
            continue

        seg_voiced_start = regions[i].start
        seg_voiced_end   = regions[i].end
        seg_len          = regions[i].n_samples
        i += 1

        # 「静音 + 有音」を追加できる限り追加
        while i < len(regions):
            if not regions[i].is_silence:
                # 隣接した有音（統合漏れ）→ 吸収
                seg_voiced_end = regions[i].end
                seg_len       += regions[i].n_samples
                i += 1
                continue

            sil = regions[i]

            # 条件②: 長い静音 → ここで切る
            if sil.n_samples >= long_silence_samples:
                break

            # 次の有音区間が存在するか確認
            if i + 1 >= len(regions):
                break
            next_voiced = regions[i + 1]
            if next_voiced.is_silence:
                i += 1
                continue

            next_len = sil.n_samples + next_voiced.n_samples

            # 条件②: max_dur を超えそう → ここで切る
            if seg_len + next_len > max_samples:
                break

            # 追加 OK
            seg_voiced_end = next_voiced.end
            seg_len       += next_len
            i += 2

        # 条件③: pad を前後に追加
        padded_start = max(0, seg_voiced_start - pad_samples)
        padded_end   = min(total_samples, seg_voiced_end + pad_samples)

        if padded_end > padded_start:
            segments.append((padded_start, padded_end))

        # 次の有音区間まで進む
        while i < len(regions) and regions[i].is_silence:
            i += 1

    return segments


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def fmt(sec: float) -> str:
    """秒を mm:ss.s 形式で返す"""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:04.1f}"


def process_file(
    wav_path: Path,
    dst_dir: Path,
    silence_thresh_db: float,
    min_silence_dur: float,
    max_dur: float,
    long_silence: float,
    pad: float,
    dry_run: bool,
) -> dict:
    """1 ファイルを処理してセグメントを書き出す。統計 dict を返す。"""
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    total_dur = len(audio) / sr

    regions = detect_regions(audio, sr,
                             silence_thresh_db=silence_thresh_db,
                             min_silence_dur=min_silence_dur)

    # 有音区間の合計・過大区間（単独で max_dur 超え）を集計
    max_samples = int(max_dur * sr)
    voiced_dur = 0.0
    oversize_voiced_dur = 0.0
    for r in regions:
        if not r.is_silence:
            d = r.n_samples / sr
            voiced_dur += d
            if r.n_samples > max_samples:
                oversize_voiced_dur += d

    segments = build_segments(regions, sr,
                              max_dur=max_dur,
                              long_silence=long_silence,
                              pad=pad,
                              total_samples=len(audio))

    seg_dur = sum((e - s) / sr for s, e in segments)

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, (s, e) in enumerate(segments):
        chunk = audio[s:e]
        out_path = dst_dir / f"{wav_path.stem}_{idx:04d}.wav"
        if not dry_run:
            sf.write(str(out_path), chunk, sr)

    return {
        'n_segments':          len(segments),
        'total_dur':           total_dur,
        'voiced_dur':          voiced_dur,
        'seg_dur':             seg_dur,
        'oversize_voiced_dur': oversize_voiced_dur,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src', required=True,
                        help='入力 WAV ディレクトリ（再帰検索）')
    parser.add_argument('--dst', required=True,
                        help='出力ディレクトリ')
    parser.add_argument('--silence_thresh', type=float, default=-50.0,
                        help='静音と判定する dBFS 閾値（デフォルト: -50）')
    parser.add_argument('--min_silence_dur', type=float, default=0.10,
                        help='静音区間と見なす最短秒数（デフォルト: 0.10 秒）')
    parser.add_argument('--max_dur', type=float, default=9.0,
                        help='セグメントの最大秒数（デフォルト: 9.0 秒）')
    parser.add_argument('--long_silence', type=float, default=1.0,
                        help='セグメントを切る長い静音の閾値秒数（デフォルト: 1.0 秒）')
    parser.add_argument('--pad', type=float, default=0.10,
                        help='前後に追加する余白秒数（デフォルト: 0.10 秒）')
    parser.add_argument('--dry_run', action='store_true',
                        help='ファイルを書き出さず統計だけ表示する')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f'エラー: 入力ディレクトリが見つかりません: {src_dir}')
        sys.exit(1)

    wav_files = sorted(src_dir.rglob('*.wav'))
    if not wav_files:
        print(f'エラー: {src_dir} に WAV ファイルが見つかりません')
        sys.exit(1)

    print(f'入力: {src_dir}  ({len(wav_files)} ファイル)')
    print(f'出力: {dst_dir}')
    print(f'設定:')
    print(f'  silence_thresh  = {args.silence_thresh} dBFS')
    print(f'  min_silence_dur = {args.min_silence_dur} 秒  （これ未満の静音は有音扱い）')
    print(f'  max_dur         = {args.max_dur} 秒')
    print(f'  long_silence    = {args.long_silence} 秒  （これ以上の静音でカット）')
    print(f'  pad             = {args.pad} 秒  （前後の余白）')
    if args.dry_run:
        print('  [DRY RUN]')
    print()

    total_segments        = 0
    total_input_dur       = 0.0
    total_voiced_dur      = 0.0
    total_seg_dur         = 0.0
    total_oversize_dur    = 0.0

    for wav_path in tqdm(wav_files, unit='files', disable=args.dry_run):
        stats = process_file(
            wav_path, dst_dir,
            silence_thresh_db=args.silence_thresh,
            min_silence_dur=args.min_silence_dur,
            max_dur=args.max_dur,
            long_silence=args.long_silence,
            pad=args.pad,
            dry_run=args.dry_run,
        )
        total_segments     += stats['n_segments']
        total_input_dur    += stats['total_dur']
        total_voiced_dur   += stats['voiced_dur']
        total_seg_dur      += stats['seg_dur']
        total_oversize_dur += stats['oversize_voiced_dur']

        if args.dry_run:
            seg_pct = stats['seg_dur'] / stats['total_dur'] * 100 if stats['total_dur'] > 0 else 0
            over_str = (f"  ⚠ 単独{args.max_dur}秒超え有音: {stats['oversize_voiced_dur']:.1f}s"
                        if stats['oversize_voiced_dur'] > 0 else "")
            print(f"  {wav_path.name}:")
            print(f"    入力 {fmt(stats['total_dur'])}  "
                  f"有音 {fmt(stats['voiced_dur'])}  "
                  f"→ {stats['n_segments']}セグメント {fmt(stats['seg_dur'])} ({seg_pct:.0f}%)"
                  + over_str)

    discarded_dur  = total_input_dur - total_seg_dur
    voiced_out_pct = (total_seg_dur / total_voiced_dur * 100) if total_voiced_dur > 0 else 0
    silence_dur    = total_input_dur - total_voiced_dur

    print(f'\n=== 集計 ===')
    print(f'  入力合計    : {fmt(total_input_dur)}')
    print(f'  有音合計    : {fmt(total_voiced_dur)}  '
          f'({total_voiced_dur / total_input_dur * 100:.1f}% of 入力)')
    print(f'  静音合計    : {fmt(silence_dur)}')
    print(f'  セグメント計: {fmt(total_seg_dur)}  '
          f'({voiced_out_pct:.1f}% of 有音)  '
          f'({total_segments} セグメント)')
    print(f'  破棄合計    : {fmt(discarded_dur)}  '
          f'({discarded_dur / total_input_dur * 100:.1f}% of 入力)')
    if total_oversize_dur > 0:
        print(f'  ⚠ 単独{args.max_dur}秒超え有音区間: {fmt(total_oversize_dur)}  '
              f'→ セグメントに含まれるが max_dur 超え')
    if not args.dry_run:
        print(f'\n出力先: {dst_dir}')
    print('\n次のステップ（preprocess.py の npz 作成）:')
    print(f'  python preprocess.py --config config_kiritan.yaml --step npz')


if __name__ == '__main__':
    main()
