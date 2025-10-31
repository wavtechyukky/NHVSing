[日本語](./readme.md) | [English](./README.en.md)

# NHV-Sing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)を、**歌声合成向けにチューニングした**モデルです。PyTorchで実装されており、JITコンパイルに対応しています。元の論文で提案された構造を踏襲しつつ、歌声合成への適用を考慮した変更を加えています。

***

## 音声サンプル

**Ground Truth:**
<audio controls src="sample_wav/ground_truth.wav"></audio>

**生成された音声:**
<audio controls src="sample_wav/inference_wav.wav"></audio>

## 特徴

### 性能

*   **軽量・高速:** 4MB程度のモデルサイズでありながら、一般的なPCのCPU環境でも非常に高速な推論を実現します。
*   **高い再現性:** 話者の声質を忠実に再現します。
*   **安定した品質:** F0（基本周波数）に忠実で、ロングトーンも破綻なく安定して合成できます。

### オリジナル実装との差異
元の論文の実装から以下の点を変更しています。（一部のパラメータはconfig.yamlから編集可能）

* **サンプリング周波数**: **44.1kHz**に対応
* **複素ケプストラム**: 次元数を**444次元**に拡張
* **FIR（postfilter）の削除**: STFT損失の低下には寄与するものの、処理が遅くなり、直感的に重要となる波形の学習に寄与しないと判断
* **Discriminator**: HiFi-GANのものを使用
* **入力特徴量**:
    * **logメルスペクトログラム**: **40Hz〜22050Hz**のlogメルスペクトログラムを入力とする。論文では高周波数帯をカットしているが、高周波数帯の再現度が直感的な品質の向上に必要であると判断した。論文のように高周波数帯を入力しない場合、FIRによる周波数帯の制御が重要になると思われる。
    * **F0**: **無声区間を線形補完**したF0を入力とし、Unvoiced/Voicedフラグを不要にした。歌声合成では無声区間も含めてF0カーブを描けることが重視されるだけでなく、論文のようにUVフラグによって挙動を変える場合、なめらかな無声区間→有声区間の推移が再現できない。

### エクスポート形式

*  **PyTorchネイティブ**: 学習時に使うモデルと同じ。
*  **TorchScript**: JITコンパイルにより他の言語から実行できるようになる。
*  **ONNX+PyTorch**: ニューラルネットワーク部分をONNXで、DSP部分をPyTorchで実装したもの。ONNXが正常に動くかの確認用。
*  **ONNX+NumPy（試作）**: DSPをNumPyで実装したもの。短い音声では目立たないが、長尺で音声が破綻し、生成が遅い。

***

## 環境

* Python 3.10.18で検証

```bash
pip install -r requrements.txt
```

## 使い方

### 1. 前処理

WAVファイルからモデルの学習に必要な特徴量（npz形式）を抽出します。

```bash
python preprocess.py --step all
```

前処理が完了したら、`dataset/npz`ディレクトリに生成されたnpzファイルを、`dataset/training_normal`（学習用）と`dataset/inference`（検証用）の各フォルダに手動で振り分けてください。

### 2. 学習

モデルの学習を開始します。`config.yaml`がデフォルトの状態では、学習の進行状況や各種ログは`logs_normal`に、モデルのスナップショットは`snapshots_normal`に保存されます。

```bash
python train.py
```

### 3. モデルのエクスポート

学習済みモデル（スナップショット）を、推論で使用できる形式（`.pth`, `.pt`, `.onnx`）にエクスポートします。

```bash
# 例: 990エポック時点のスナップショットをエクスポートする場合
python export.py --checkpoint snapshots_normal/000990epoch.pth --config config.yaml
```

### 4. 推論

エクスポートしたモデルを使って、npzまたはWAVファイルから音声を生成します。

**PyTorchネイティブモデル (.pth) を使う場合:**

```bash
python inference.py <input.wav_or_npz> --snapshot <path/to/model.pth>
```

**各種エクスポート済みモデルの動作を一度に確認する場合:**

```bash
python -m debug.inference_checker <path/to/wav> <path/to/output> --config config.yaml --pth_path exported_models/model.pth --pt_path exported_models/model_jit.pt --onnx_path exported_models/core_model.onnx
```

***

## 課題点

*   **学習プロセス:** 複数のGPUの使用や、2以上のバッチサイズでの学習を想定しておりません。（バッチサイズ1での学習も大きなメモリが必要）
*   **ONNXエクスポート:** ボコーダー全体のONNXエクスポートは非対応です。torch.fft関連のコードと、LTV Filterの処理でエクスポートに躓くようになっています。
*   **低音域の品質:** 記載の設定で試した限り、特に低い声域の男性ボーカルを合成する際に、音質が劣化する傾向があります。原理的に男声の声を再現できないということはないと思うので、ハイパーパラメータや入力するメルスペクトログラムの周波数帯を変えれば品質を上げられるのではと考えています。
*   **話者依存性:** 生成される波形には学習させた話者の特徴が反映され、多人数の声を再現することは難しいです。その代わり、FastSpeech2のような曖昧な音響特徴量を入れてもリアルな質感の声にしてくれます。

## ライセンス

このプロジェクトは [MIT License](LICENCE) のもとで公開されています。

## 謝辞

このリポジトリは、Liu, et al.によって発表された以下の論文やリポジトリに基づいています。

*   Z. Liu, Y. Wang, K. Chen and Y. Jia, "Neural Homomorphic Vocoder," *Proc. Interspeech 2020*, pp. 3500-3504, doi: 10.21437/Interspeech.2020-2325.
*   [https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf)
*   [https://github.com/xcmyz/FastVocoder/tree/main](https://github.com/xcmyz/FastVocoder/tree/main)
*   [https://github.com/zjlww/dsp](https://github.com/zjlww/dsp)
*   [https://pypi.org/project/neural-homomorphic-vocoder/](https://pypi.org/project/neural-homomorphic-vocoder/)

