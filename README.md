[日本語](./README.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)を、**歌声合成向けにチューニングした**モデルです。PyTorchで実装されており、JITコンパイルおよび単一ファイルONNXエクスポートに対応しています。

本リポジトリには最新の **NHVSing V3 / V3X**（音質改善版のV3.2・V3.2Xを推奨）と、レガシーの **NHVSing**（V1・単一話者）/ **NHVSingV2**（多話者）が含まれます。V3.1は特定の周波数で、フレーム単位で波形の位相が反転し、隣接するフレームの波形を打ち消す動作が起こっていたため非推奨です。

🎧 **試聴はこちら → [NHVSing V3.2 デモページ](https://wavtechyukky.github.io/NHVSing/v3_2.html)**（最新の配布ウェイト。NSF-HiFiGANとの copy-synthesis聴き比べ）
&nbsp;·&nbsp; [V3 デモページ](https://wavtechyukky.github.io/NHVSing/v3.html)（RTF・モデルサイズの比較、DiffSinger音響モデルでの歌声合成）

***

## NHVSing V3 / V3X（最新・推奨）

**V3** は NHV を歌声向けに突き詰めた最新モデルです（44.1kHz / hop256 / 128-mel[40–16000Hz, **ln**]）。V2 からの本質的な改良は3点:

- **MRD (Multi-Resolution Discriminator) の採用**（+ MPD）: 品質向上の最大の要因。UnivNet系の多解像度スペクトログラム magnitude 判別。
- **インパルス応答生成の高速化 (`fft_corr`)**: LTV-FIRにFFTを使用し、時間領域で処理するバージョンとビット等価で、CPUで約8倍速に。
- **教師データの選別、水増し**: 44.1kHzの高音質な音声を集め、さらに学習中に音量と音高をランダムに変化させながら与えた。普通の波形から逸脱したものを学習で与えることになりそうだが、ボコーダーの汎化性能はこれで有意に向上した。
- **励起・条件の整理**: 高音質化を図る対策を沢山講じたが、上の2つの方法の効果が大きく、一旦構成を元論文準拠のものにした。構成を大幅に変えることがあればV4以降になると思われる。励起インパルスは正弦波を200倍音まで加算 / quef_norm α=1.0 / **melのみ入力**（F0 embedding 廃止）/ F0は線形補間（元論文NHVへの回帰）。

**V3X** はV3を **hop512入力**で使えるようにした派生です。hop512のmel/F0を受け、内部で整列中点補間してhop256グリッドへ戻してからV3を通します（重み・state_dict はV3と共有）。**V1/V2の「hop512だと品質が大きく落ちる」問題を解消**します。

| クラス | 用途 | config |
|---|---|---|
| `NHVSing` | V1 legacy（単一話者） | `config.yaml` |
| `NHVSingV2` | V2 legacy（多話者） | `config_v2.yaml` |
| **`NHVSingV3`** | **最終モデル（hop256 native）** | **`config_v3.yaml`** |
| **`NHVSingV3X`** | **V3 の hop512 入力版** | `config_v3.yaml` + `ltv_filter.use_v3x: true` |

> V3.2 も同じ `NHVSingV3` / `NHVSingV3X` を使い、config だけ **`config_v3_2.yaml`**（`ltv_filter.ola_mode: hann`）に差し替えます。

### V3.2 — 最新の配布ウェイト（2026-09）

**V3.2** のV3からの変更点は以下のとおりです。アーキテクチャは不変（クラス・モデルサイズ・RTFは同じ）で、変わったのは一部のバグ修正と学習の仕方です:

- **LTVフィルタの生成波形にHann窓をかけた（Hann WOLA・50%重複）**: フレーム単位で急激に波形が弱まる現象の主因は、時変FIRを矩形窓の非重複OLAで畳んでいたため、あるフレームの畳み込み波形が隣フレームへ大振幅ではみ出し、隣フレームがそれを180°逆位相で打ち消していたことでした。窓を2×frame_sizeのHann窓に変えてはみ出しを境界でテーパーし、波形の重複を実質4回から2回分の強度に変更しました。急峻なmelの変化があった場合に近接フレームの波形が重なり、melと波形が不整合になることを低減する狙いもあります。配布ONNX（`export.py`→`LTVFirONNX`）も同時にHann窓化しており、学習と配布で計算が一致します。
- **励起位相のfloat64化**: インパルス列の位相をfloat32で累積していたため、長尺で精度が落ちて倍音が滲んでいました。位相をfloat64で累積し、mod 1.0で折り返してからfloat32のcosに渡す方式に変更（PyTorchとONNXでほぼ計算が一致）。
- **pitch augmentationの連続倍率化**: `torchaudio.resample` は有理数比しか扱えず、旧実装はピッチシフト倍率が22段階に量子化されていました。連続倍率の線形補間リサンプルに変更。
- **訓練中の励起インパルスの開始タイミングのランダム化**: V3.1で瞬間的にインパルス応答の位相が反転し、隣接するフレームの波形を打ち消してしまう動作が起こったため、局所的に特定の位相に偏る学習が行われないことを目的に機能を追加しました。
- **多重解像度STFT損失の短時間の解像度のウィンドウ追加**: これも上と同様、V3.1で起こった隣接するフレームの波形を打ち消す動作の対策として追加しました。元の解像度では1フレームの波形の弱まりを検知できないことが分かったためです。
- **低学習率の仕上げのステップを追加**: 過学習しないよう早めに学習を切り上げつつ、melの再現度を高めるように追加しました。

**配布ONNXのメモリ修正（2026-09-19・重みは不変）**

配布ONNXは入力長に比例してメモリを消費し、**24秒のフレーズ1本で3.4GBを超えて落ちていました**（16GB機でスワップが埋まりOSごと重くなります）。原因は3箇所で、いずれも入力長ぶんを一度に展開していたためです。

| 箇所 | 中間テンソル | 対策 |
|---|---|---|
| 励起インパルス（`impulse_train_onnx`） | `cos` が `[B, 200, 256T]` | 倍音200本を**ループ**で `[B,1,n]` の累算器へ加算 |
| 時変FIRのFFT（`ltv_fir_onnx`） | `[B, T, 2048, 2]` ×10 | フレーム方向に**ブロック分割**（FFTはフレーム内なので各フレームの結果は不変） |
| ケプストラム→IR（`complex_cepstrum_to_imp_onnx`） | `[B, T, 1024, 2]` ×6 | 同上 |

`torch._higher_order_ops.scan`（ONNXの`Scan`）で動的長のループにしています。**約125MB/秒 → 約8MB/秒**になり、24秒で0.33GB、240秒でも2.05GBに収まります。出力は配布済みV3.2と `harmonic` 出力で **−142dB以内**（float32の丸め相当・16bit量子化ノイズより46dB小）で一致します。

**速度の代償**（対策前後の配布ONNXを同一環境で直接比較。3秒入力・9回中央値・1プロセス1測定）:

| スレッド数 | 対策前 | 対策後 | |
|---|---|---|---|
| 1 | 0.071 | 0.073 | +4% |
| 2 | 0.063 | 0.065 | +4% |
| 4 | 0.059 | 0.063 | +7% |
| 8 | 0.061 | 0.081 | **+33%** |

ループ化した部分はスレッドで分割できないので、コアが多いほど不利になります。1〜4スレッドでの数%は、長尺入力でプロセスが落ちなくなることに対して妥当な代償と判断しました。多コアで最速を求めるなら、`Scan`を使わないtorch経路（下のRTF比較表）の方が速いです。

> `v3` の配布ONNXは、上の「励起位相のfloat64化」が反映されていない古い書き出しのままでした。今回あわせて書き出し直しています（長いほど差が開き、24秒で−27.8dB）。`v3_1` / `v3_2` は修正済みだったため影響ありません。

**学習側（`dsp.py` / `model.py`）は既定では1ビットも変わりません。** 同じ問題は `dsp.py` のeager実装にもありますが（実測 約150MB/秒・24秒で3GB超）、学習は372msのクロップなので踏みません。長尺をtorchで推論する場合のみ `vocoder.harm_block: 20`（または環境変数 `NHV_HARM_BLOCK=20`）で倍音和をループ化できます（既定0＝従来と完全に同一）。

> ※「励起位相のfloat64化」と「pitch augmentation の連続倍率化」は元々 **V3.1** で入れた修正で、V3.2 にも引き継いでいます。V3.2 で新規に加えたのは Hann窓化・励起位相のランダム化・STFT の短時間ウィンドウ追加・低学習率の仕上げです。

配布ファイルは `exported_models/v3_2/`（標準の `export.py` 出力をリネームしたもの）:

- **`nhv_v3_2.pth`** — V3/V3X 共有の重み
- **`nhv_v3_2.onnx` / `nhv_v3_2x.onnx`** — 重み内蔵の単一ファイル ONNX（I/O 契約は `nhv_v3.onnx` / `nhv_v3x.onnx` と同一）

学習レシピは **`config_v3_2.yaml`**。

### 性能（V3）

**モデルサイズ**: `nhv_v3.onnx` は **約 7.8 MB**（量子化なし）。比較対象のNSF-HiFiGAN（pc-nsf-hifigan, 56.7MB）の **約1/7** です。

> V3.2 までは約 2.2 MB でした。長尺入力でメモリが破綻する問題（上記）を直す際に、励起と時変FIRを `Scan` でループ化したため、`scatter_add` の添字テーブルが定数としてグラフに埋まり約 5.6 MB 増えています。**重みは 0.478 M パラメータのまま**で、増えたのは添字だけです。

**RTF**（Real-Time Factor = 音声1秒あたりの計算秒数。小さいほど速く、< 1 で実時間より速い）。測定環境はM4 MacBook Airの10コアCPU（高性能4＋高効率6）/ 約5秒入力 / バッチ1 / 9回の中央値。**下記はすべて同一セッションで交互に測った値です**（ORTセッションを同一プロセスに複数作ると1.4倍ほどブレるため、1プロセス1測定）。

ONNX Runtime（CPU）でNSF-HiFiGANと並べると:

| スレッド数 | NHVSing V3 | NSF-HiFiGAN | NHVSing の速さ |
|---|---|---|---|
| 1 | 0.073 (14×) | 0.589 (2×) | **8.0×** |
| 2 | 0.061 (16×) | 0.314 (3×) | 5.2× |
| 4 | 0.058 (17×) | 0.196 (5×) | 3.4× |
| 8 | 0.075 (13×) | 0.199 (5×) | 2.6× |

**コア数への依存が両者で大きく異なります。** NHVSing V3は各フレームのインパルス応答生成が完全に独立（*embarrassingly parallel*）ですが、**ONNX Runtimeはこの並列性をほとんど活かせず**、実質シングルコア律速です（14×→17× で頭打ち）。一方NSF-HiFiGANは大きな転置畳み込みが良く並列化し、コアを増やすほど速くなります（2×→5×）。この結果、**NHVSingの速度優位は低コア環境で最大（〜8×）、多コアでは〜3×に縮小**しますが、どの条件でも上回ります。

なお8スレッドでは両者とも4スレッドより悪化しますが、**要因は同じではありません**。NSF-HiFiGANの悪化（+1.5%）は高効率コアに踏み込むためですが、NHVSingの悪化（+29%）は大半が前述のメモリ対策で入れた**逐次ループ**によるものです。スレッドを増やしてもここは分割できません。

**PyTorch（ネイティブ）では並列性が活きます。** 同じV3をtorchで回すと、per-frameの独立性によりコア数でスケールします:

| スレッド数 | ONNX Runtime | PyTorch |
|---|---|---|
| 1 | 0.073 (14×) | 0.069 (15×) |
| 2 | 0.061 (16×) | 0.051 (20×) |
| 4 | 0.058 (17×) | **0.039 (26×)** |
| 8 | 0.075 (13×) | 0.049 (20×) |

多コアでは **torchの方が自前ONNXより速い**（4スレッドで約1.5倍）という逆転が起きます。極小・FFT主体のモデルゆえ、ORTのグラフ最適化よりもtorchのバッチFFT並列化のほうが効くためです。したがって「NHVSingは遅い/速い」は単一の数字では語れず、**ランタイムとコア数の組み合わせで決まります**。

> このtorch列は `export.py` の `FullVocoderV3`（ONNX用部品をそのままeagerで実行したもの）で、学習側の `model.py::NHVSingV3` ではありません。

> RTF は計算量のみに依存し、モデルの重み値には依存しません（どのckptでも同じ）。

ONNXは `LTVFirONNX` のFFT長を **2の冪へpad** して高速化しています（ONNX RuntimeのDFTは2の冪サイズのみ高速）。V3Xも同等です。LTV-FIRの時間相関自体も `fft_corr` でFFT化済み（時間領域版とビット等価でCPUで約8倍速。上記「主な変更点」参照）。

### 使い方（V3）

**前処理**（F0 = RMVPE単体で、1フレーム単位の推定誤差を弾く処理を追加。初回に `rmvpe.pt` を自動DL）:
```bash
python preprocess.py --indir <歌唱wavディレクトリ> --out <npz_dir> --config config_v3.yaml
```

> **train/testの分け方**: `preprocess.py` は `--out` に全 shard を出力するだけで、train/testの自動振り分けはしません。**wavを学習用・検証用に分けて2回実行**し、それぞれ別ディレクトリへ出力してください（少数をtestに回せば十分）:
> ```bash
> python preprocess.py --indir wavs/train --out dataset/train --config config_v3.yaml
> python preprocess.py --indir wavs/eval  --out dataset/test  --config config_v3.yaml
> ```
> `config_v3.yaml` の `training.train_dir` / `test_dir` をそれぞれのディレクトリに設定します。`VocoderDataset` はshard（`<sid>|f0` / `|log_melspc` / `|wav`）・単一セグメントのnpzの両方を再帰的に読むので、どちらの形式でも構いません。

**学習**（MRD + MPD GAN。`config_v3.yaml` の `training.train_dir` / `test_dir` / `snapshot_dir` 等を設定）:
```bash
python train_v3.py --config config_v3.yaml
```

**ONNXエクスポート**（V3・V3Xの両方。3出力: `waveform` / `harmonic` / `noise`）:
```bash
python export.py --config config_v3.yaml --ckpt <weights.ckpt> --out exported_models
```
既定で `exported_models/v3/` に以下を出力します:

- **`nhv_v3.pth`** — V3/V3X共有の重み（`NHVSingV3(vc, lc).load_state_dict(torch.load('nhv_v3.pth'))` でロード可）。**V3Xは重みをV3と共有するので `.pth` は無し**（onnxのみ）。
- **`nhv_v3.onnx` / `nhv_v3x.onnx`** — 各 **NN + DSP全部入りの単一ONNX**（重み内蔵・`.onnx.data` 等の外部ファイル無し・ONNXRuntimeのみで推論可能。V1/V2の `full_vocoder.onnx` と同形式）。入力 `mel` / `f0` / `uv` → 出力 `waveform` / `harmonic` / `noise` の3出力で、`clamp(harmonic + noise) == waveform`。入力長Tは動的（任意長）。

> **V3.2** を出力するときは `--config config_v3_2.yaml` を指定し、出力された `nhv_v3.*` を `nhv_v3_2.*` にリネームします（配布物 `exported_models/v3_2/` はこの手順で作成）。

**推論（Python）**:
```python
from nhv_vocoder import NHVVocoder
voc = NHVVocoder('weights.ckpt', 'config_v3.yaml')      # config の use_v3x で V3/V3X 自動選択
cf0, uv = NHVVocoder.prep_f0(f0_hz)                       # 0=無声 の生F0 → 連続F0 + uv
wav = voc.infer(mel, cf0, uv)                            # mel: [T, 128] ln-mel
```

### F0 抽出について

前処理のF0は **RMVPE 単体 + 跳躍除外の後処理**（`tools/f0`）を使います。RMVPEのモデル重み `rmvpe.pt`（~173MB）はリポジトリに含めず、**初回実行時に HuggingFace から自動ダウンロード**されます。

### 配布重みのライセンス

配布する学習済み重みは、学習データの都合上 **非商用（non-commercial）** とします。

***

# 以下はレガシー版（NHVSing V1 / NHVSingV2）の説明です

> ⚠️ **ここから下のセクション（音声サンプル・アーキテクチャ・性能・使い方など）は、すべて旧世代 V1 / V2 の内容**です。新規利用は上記の **V3 / V3X（推奨）** を参照してください。V1 / V2 は互換のために残しています。

## 音声サンプル

→ **[NHVSingV2 デモページで試聴する](https://wavtechyukky.github.io/NHVSing/v2.html)**

→ [NHVSing (V1) デモページ](https://wavtechyukky.github.io/NHVSing/)

きりたん・夏目悠李（NHVSing）、およびM4Singer・GTSinger評価データ（NHVSingV2）による生成音声を聴き比べできます。

***

## NHVSing と NHVSingV2 の違い

| | NHVSing | NHVSingV2 |
|---|---|---|
| CNN backbone | Dual-branch（Harmonic/Noise独立） | Shared trunk（CNNを共有） |
| F0入力 | メルスペクトログラムのみ | F0 embedder（256 bins, log₂スケール, 128-dim）を追加結合 |
| quef_norm | オフ（V1では高周波学習が阻害されたため） | alpha=0.3のソフトスケーリング（高周波を犠牲にせず安定化） |
| 学習時の振幅拡張 | なし | 0.5〜2.0×（log-uniform）でランダムスケール |
| 話者汎用性 | **単一話者特化**。学習話者以外ではアーティファクトが出やすい | **多話者対応**。M4Singer・ACE-Opencpop等の多話者コーパスで学習可能 |
| 設定ファイル | `config.yaml` | `config_v2.yaml` |

NHVSingは単一話者のデータセットで学習することで、その話者の声質を忠実に再現することに優れています。一方NHVSingV2は多話者コーパスで学習することにより、様々な話者の音響特徴量を高品質に再合成することが可能です。

***

## 性能（V1 / V2）

以下の環境・条件で測定しました。

- **測定環境:** Apple M-series CPU（MacBook）
- **測定条件:** 入力44.1kHz・約26秒・バッチサイズ1

### NHVSing（V1）

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

## アーキテクチャ（V1 / V2）

元の論文の実装から以下の点を変更しています。（V3 の disc は MRD + MPD で、下記の MSD + 複素STFT とは別物です）

* **サンプリング周波数**: **44.1kHz**に対応
* **複素ケプストラム**: 次元数を**512次元**に拡張
* **FIR（postfilter）の削除**: STFT損失の低下には寄与するものの、波形の学習に寄与しないと判断
* **Discriminator**: Multi-Scale Waveform Discriminator + Multi-Scale Complex STFT Discriminatorを使用。Adversarial lossにwarmup期間を設けており、学習途中からのFine-tuningにも対応（`adversarial_warmup_epochs`）
* **損失関数の追加**:
    * **Envelope loss**: 1D max-poolingで上下包絡を抽出しMAEを計算（RefineGAN §2.5.1）。振幅包絡の不安定化を抑制する（`envelope_scale`）
    * **Harmonic penalty loss**: 無声区間（F0=0のフレーム）で有声成分（`sig_harm`）が出力されることへのL1ペナルティ。無声区間でのブザー音を抑制する（`harmonic_penalty_scale`）
* **F0入力**: **無声区間を線形補間**したF0を入力とし、Unvoiced/Voicedフラグを不要にした。歌声合成では無声区間も含めてF0カーブを描けることが重要で、UVフラグによる挙動切り替えでは無声→有声のなめらかな推移が再現できない
* **logメルスペクトログラム**: **40Hz〜22050Hz**の全帯域を入力とする。高周波数帯の再現度が直感的な品質向上に寄与すると判断した

### NHVSingV2 固有の変更点

* **Shared trunk CNN** (`use_shared_trunk: true`): HarmonicとNoiseの両ブランチが共有の幹CNNを通ってからそれぞれのヘッドへ分岐する。これによって入力された音響特徴量を、HarmonicとNoiseのどちらを支配的にして生成すべきかを、両者が独立して学習する必要がなくなった。
* **F0 Embedder** (`use_f0_embed: true`): 連続F0を256ビンのlog₂スケールで離散化し、128次元に埋め込んでメルスペクトログラムと結合する。ネットワークに明示的な音高情報を与えることで、生成すべき一周期分の波形の手がかりが増える。
* **quef_norm** (`use_quef_norm: true`, `quef_norm_alpha: 0.3`): ケフレンシー成分に1/|n|^αの緩やかなスケーリングをかけ、高次倍音を抑制しすぎずに学習を安定化させる。過去の検証ではオンにすると高周波数帯が阻害されたが、alphaを0.3と小さく設定することで高周波を犠牲にせず安定化できることが確認された

### エクスポート形式

*  **PyTorchネイティブ** (`model.pth`): 学習時に使うモデルと同じ。
*  **TorchScript** (`model_jit.pt`): JITコンパイルにより他の言語から実行できるようになる。
*  **Unified ONNX** (`full_vocoder.onnx`): ボコーダー全体（NN + DSP）を単一のONNXファイルにエクスポート。ONNXRuntimeのみで推論可能。

***

## 環境

* Python 3.10で検証

```bash
pip install -r requirements.txt
```

***

## 使い方（V1 / V2）

### NHVSing（V1）の場合

#### 1. 前処理

```bash
# WAVカット → F0/メル抽出 → train/test振り分けをまとめて実行
python preprocess.py --config config.yaml --step all
```

#### 2. 学習

```bash
python train.py --config config.yaml
```

#### 3. Fine-tuning（別話者への転移学習）

学習済みモデルのウェイトのみを引き継ぎ、別話者のデータセットで再学習します。DiscriminatorとOptimizerは新規初期化されます。

```bash
python prepare_finetune.py \
  --weights exported_models/natsume/model.pth \
  --config config.yaml \
  --output snapshots_kiritan/000000epoch.pth

python train.py --resume_path snapshots_kiritan/000000epoch.pth --config config_fine_tuning.yaml
```

#### 4. エクスポート

```bash
python export.py \
  --checkpoint snapshots/000990epoch.pth \
  --config config.yaml \
  --output_dir exported_models/kiritan \
  --all
```

#### 5. 推論

```bash
python inference.py input.wav \
  --snapshot exported_models/kiritan/model.pth \
  --config config.yaml \
  --output_dir output \
  --onnx exported_models/kiritan/full_vocoder.onnx
```

---

### NHVSingV2 の場合

`config_v2.yaml` を起点として使います。パス・話者プレフィックス・学習パラメータを環境に合わせて編集してください。

#### 1. 前処理

F0抽出にRMVPEを使用します（初回実行時にモデルを自動ダウンロード）。

```bash
python preprocess.py --config config_v2.yaml --step all
```

#### 2. 学習

```bash
python train.py --config config_v2.yaml
```

#### 3. Fine-tuning（別話者・別コーパスへの転移学習）

```bash
python prepare_finetune.py \
  --weights exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output snapshots_finetune/000000epoch.pth

python train.py --resume_path snapshots_finetune/000000epoch.pth --config config_v2.yaml
```

#### 4. エクスポート

```bash
python export.py \
  --checkpoint snapshots_v2/000900epoch.pth \
  --config config_v2.yaml \
  --output_dir exported_models/v2 \
  --all
```

#### 5. 推論

```bash
python inference.py input.wav \
  --snapshot exported_models/v2/model.pth \
  --config config_v2.yaml \
  --output_dir output \
  --onnx exported_models/v2/full_vocoder.onnx
```

WAV入力の場合、`config_v2.yaml` の `target_rms: 0.083` が設定されていれば有声区間のRMSをM4Singer学習データの中央値に自動正規化します。

***

## 学習のベストプラクティス

### 振幅拡張（`amp_augment`）

NHVSingV2では学習時に音量を0.5〜2.0倍（log-uniform）でランダムスケールする振幅拡張を導入しています（`amp_augment: true`, `amp_aug_range: [0.5, 2.0]`）。これにより、入力音量の変動に対してモデルが頑健になり、推論時の音量合わせが多少ずれても品質が劣化しにくくなります。

### quef_norm のスケール（`quef_norm_alpha`）

`use_quef_norm: true` にすることでケフレンシー成分を正規化し、学習の安定性が向上します。ただしalphaを大きくしすぎると高周波数帯（子音・摩擦音）の再現度が低下します。`quef_norm_alpha: 0.3` が高周波を犠牲にせずに安定化できる妥当な値です。

### Harmonic penalty loss の強さ（`harmonic_penalty_scale`）

無声区間（F0=0フレーム）でハーモニック成分が漏れ出すことを抑制するペナルティです。この値は品質に大きく影響します。

* 小さすぎる（0〜10程度）: 無声区間でブザーのような音が発生しやすい
* 大きすぎる（1000以上）: 有声/無声か怪しい音響特徴量が与えられた時に、本来は有声区間であっても、無声音でメルスペクトログラムを再現しようとしてしまう。
* **推奨値: `harmonic_penalty_scale: 100`**

***

## 課題点

*   **学習プロセス:** 複数のGPUの使用を想定していません。
*   **フレームサイズ:** V1/V2 は hop_size=512 にすると品質が大きく落ちます。**V3X が hop512 入力に対応**（内部で hop256 へ整列補間）し、この問題を解消しています。
*   **NHVSing（V1）の話者依存性:** 生成される波形には学習させた話者の特徴が強く反映されます。多話者対応が必要な場合はNHVSingV2を使用してください。

***

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。

## 謝辞

このリポジトリは、Liu, et al.によって発表された以下の論文・リポジトリに基づいています。

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)

## 使用した歌声データベース

### V3

配布するV3のウェイトは以下の非商用データで学習しています。商用利用することはできません。**各データセットのライセンス・入手先・利用条件は、下記リンク先の一次ソースで直接ご確認ください**

*   東北きりたん — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   夏目悠李 — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   波音リツ（Ritsu Singing DB Ver2.0-2.2 / Soft）— [Canon Voice](https://www.canon-voice.com/voicebanks/)
*   Children's Song Dataset (CSD) — [Zenodo](https://zenodo.org/records/4916302)
*   NUS-48E — [Zenodo](https://zenodo.org/records/19595152)
*   御丹宮くるみ（ONIKU_KURUMI うたごえ）— [Onikuru](https://onikuru.info/db-download/)
*   Opencpop — [WeNet Opencpop](https://wenet-e2e.github.io/opencpop/download/)
*   No.7 — [VOICE SEVEN](https://voiceseven.com/7dev/login.php)
*   VocalSet — [Zenodo](https://zenodo.org/records/1193957)
*   ccmusic-database / acapella — [HuggingFace](https://huggingface.co/datasets/ccmusic-database/acapella)

### V1 / V2（レガシー）

*   東北きりたん — [Zunko Project](https://zunko.jp/kiridev/login.php)
*   夏目悠李 — [NJKS Official](https://ksdcm1ng.wixsite.com/njksofficial)
*   M4Singer (CC BY-NC-SA 4.0) — [M4Singer GitHub](https://github.com/M4Singer/M4Singer)
    *   Zhang et al., "M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus," *NeurIPS 2022*.
*   ACE-Opencpop — [HuggingFace](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
