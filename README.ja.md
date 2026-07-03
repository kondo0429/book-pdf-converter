# Book PDF Converter

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[English](README.md)

**[DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) の Python/Cython 移植版**

スキャンした書籍PDFを、AI画像処理と高度な画像処理技術により、デジタル書籍並みの高品質なドキュメントに変換するツールです。
詳しくは本家[DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter)をご覧ください。

現在本家v1.00の移植まで完了しています。

## インストール

### 必要条件

- Python 3.10〜3.13（3.14以降は依存ライブラリが未対応）
- Cコンパイラ（Cython拡張のビルド用）
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)（ページ番号検出に使用）

### クイックインストール

```bash
# リポジトリをクローン
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

# 依存関係をインストール
pip install -r requirements.txt

# AIモデルをダウンロード・セットアップ（Macでは自動的にCoreMLに変換）
python scripts/setup_model.py

# Cython拡張をビルドしてインストール（モデルもパッケージに含まれる）
pip install .
```

### プラットフォーム別セットアップ

<details>
<summary><b>macOS（Apple Siliconに最適）</b></summary>

```bash
# システム依存関係をインストール
brew install tesseract tesseract-lang

# クローンしてインストール
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py  # CoreMLに自動変換
pip install .
```

CoreMLはM1/M2/M3チップのNeural Engineを使用して高速な推論を提供します。

</details>

<details>
<summary><b>Ubuntu/Debian（Linux）</b></summary>

```bash
# システム依存関係をインストール
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
sudo apt install -y build-essential python3-dev  # Cython用

# クローンしてインストール
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py
pip install .
```

CUDAアクセラレーションを使用するには、[NVIDIAドライバ](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)がインストールされていることを確認してください。

</details>

<details>
<summary><b>Windows</b></summary>

1. [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)をインストール（PATHに追加）
2. [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)をインストール（Cython用）

```bash
# クローンしてインストール
git clone https://github.com/robios/book-pdf-converter.git
cd book-pdf-converter

pip install -r requirements.txt
python scripts/setup_model.py
pip install .
```

CUDAアクセラレーションには、[CUDA付きPyTorch](https://pytorch.org/get-started/locally/)をインストールしてください。

</details>

## 使用方法

### 基本的な使い方

```bash
# AI鮮明化付きでPDFを変換
book-pdf-converter input.pdf output.pdf

# AI鮮明化をスキップ（高速、前処理済みスキャン向け）
book-pdf-converter input.pdf output.pdf --skip-enhancement
```

### バッチ処理

ディレクトリ内の複数のPDFを一括変換できます。フォルダ構造は保持されます。

```bash
# ディレクトリ内のすべてのPDFを変換
book-pdf-converter-batch input_dir/ output_dir/

# 既に変換済みのファイルをスキップ
book-pdf-converter-batch input_dir/ output_dir/ --skip-existing

# エラーが発生しても続行
book-pdf-converter-batch input_dir/ output_dir/ --continue-on-error
```

`book-pdf-converter-batch`は`book-pdf-converter`と同じオプションに加え、`--skip-existing`と`--continue-on-error`が使用できます。

### 詳細オプション

```bash
# 最初/最後のページをバイパス（表紙・裏表紙用）
book-pdf-converter input.pdf output.pdf --bypass-first --bypass-last

# カスタムモデルを指定
book-pdf-converter input.pdf output.pdf --model /path/to/model.pth

# 余白のパーセンテージを調整（デフォルト: 7%）
book-pdf-converter input.pdf output.pdf --margin-percent 5

# 傾き補正の許容角度を広げる（指定した角度まで補正、単位: 度）
book-pdf-converter input.pdf output.pdf --max-deskew-degree 10

# 特定ページの傾き補正をスキップ（1始まり、範囲指定可）
book-pdf-converter input.pdf output.pdf --deskew-exclude-pages 1,4,7-9

# 全ページの傾き補正を無効化
book-pdf-converter input.pdf output.pdf --no-deskew

# 裏映り・背景除去はデフォルトで有効（グレースケール出力）。
# カラー/写真ページは除外（通常の色調整が適用される）、または全体無効化
book-pdf-converter input.pdf output.pdf --bleed-removal-exclude-pages 5,12-14
book-pdf-converter input.pdf output.pdf --no-bleed-removal

# 裏映り除去の調整（ホワイトポイントを下げるほど強力に白色化）
book-pdf-converter input.pdf output.pdf --bleed-white-point 195

# 余白の白色化: 文字のない外周余白バンドをクリア（デフォルトで有効）
book-pdf-converter input.pdf output.pdf --no-margin-whitening
book-pdf-converter input.pdf output.pdf --margin-pad 60

```

### 全オプションリファレンス

```
usage: book-pdf-converter [-h] [--model MODEL] [--scale SCALE] [--tile TILE]
                     [--skip-enhancement] [--dpi DPI]
                     [--margin-percent MARGIN_PERCENT] [--bypass-first]
                     [--bypass-last] [--denoise-strength DENOISE_STRENGTH]
                     [--max-deskew-degree MAX_DESKEW_DEGREE] [--no-deskew]
                     [--deskew-exclude-pages DESKEW_EXCLUDE_PAGES]
                     [--no-bleed-removal]
                     [--bleed-removal-exclude-pages BLEED_REMOVAL_EXCLUDE_PAGES]
                     [--ocr-lang OCR_LANG]
                     [--pdf-format {jpeg,png}] [--jpeg-quality JPEG_QUALITY]
                     [--max-pages MAX_PAGES] [--keep-temp] [--quiet]
                     [--workers WORKERS]
                     input output

位置引数:
  input                 入力PDFファイル
  output                出力PDFファイル

オプション:
  -h, --help            ヘルプメッセージを表示
  --model, -m MODEL     鮮明化モデルのパス（.mlpackageまたは.pth）
  --scale, -s SCALE     アップスケール倍率（デフォルト: 2）
  --tile, -t TILE       鮮明化のタイルサイズ（デフォルト: 512）
  --skip-enhancement    AI鮮明化をスキップ
  --dpi DPI             PDF描画の入力DPI（デフォルト: 300）
  --margin-percent PCT  出力余白のパーセンテージ（デフォルト: 7）
  --bypass-first        最初のページ（表紙）の処理をスキップ
  --bypass-last         最後のページ（裏表紙）の処理をスキップ
  --denoise-strength N  傾き補正用ノイズ除去強度（デフォルト: 20、0で無効）
  --max-deskew-degree D 補正する傾きの最大角度（度）。これを超える検出値は無視
                        （デフォルト: 10）
  --no-deskew           全ページの傾き補正を無効化
  --deskew-exclude-pages PAGES
                        傾き補正をスキップするページ番号（1始まり）例: "1,4,7-9"
  --no-bleed-removal    全ページの裏映り・背景除去を無効化
                        （デフォルトで有効。出力はグレースケール）
  --bleed-removal-exclude-pages PAGES
                        裏映り除去をスキップするページ番号（1始まり）
                        例: "1,4,7-9"（除外ページは通常の色調整が適用。
                        カラー/写真ページに推奨）
  --bleed-bg-ksize N    裏映り除去: 背景推定カーネルサイズ（デフォルト: 151）
  --bleed-black-point N 裏映り除去: この値以下をインク/黒に（デフォルト: 115）
  --bleed-white-point N 裏映り除去: この値以上を紙/白に。下げるほど強力
                        （デフォルト: 205）
  --no-margin-whitening
                        文字のない外周余白バンドの白色化を無効化
  --margin-pad N        余白白色化: 検出した文字領域の周囲に残す
                        ピクセル数（デフォルト: 40）
  --ocr-lang LANG       Tesseractの言語コード（デフォルト: eng+jpn）
  --pdf-format FMT      PDF内の画像形式: jpegまたはpng（デフォルト: jpeg）
  --jpeg-quality N      JPEG品質 0-100（デフォルト: 70）
  --max-pages N         処理する最大ページ数（テスト用）
  --keep-temp           一時ディレクトリを保持
  --quiet, -q           進捗出力を抑制
  --workers N           並列ワーカー数
```

## 本家版との違い

この移植版は本家のC#実装を忠実に再現していますが、以下の意図的な違いがあります：

| 変更点 | 説明 |
|--------|------|
| `--bypass-first/last` | 表紙ページの傾き補正/色調整/クロップをスキップしつつ、AI鮮明化は適用するオプションを追加 |
| 傾き補正の制御 | `--max-deskew-degree`（デフォルト10°、本家の許容上限1°から拡大）、`--no-deskew`、`--deskew-exclude-pages` を追加。なお、Radon変換ベースの角度検出は約7°までしか測定できないため、これを超える傾きは設定に関わらず補正できません |
| 裏映り・背景除去 | デフォルトで有効：本家のグローバル線形色調整の代わりに、各ページの紙背景を局所推定（モルフォロジークロージング＋ブラー）し、フラットフィールド正規化で紙を均一な白に平坦化、さらにコントラストストレッチ（`--bleed-black-point` / `--bleed-white-point`）でインクを黒、裏映りを白へマップします。裏面のゴースト文字と紙の色ムラの両方が消えます（出力はグレースケール）。カラー/写真ページは `--bleed-removal-exclude-pages` で除外（本家の色調整が適用）、`--no-bleed-removal` で全体無効化 |
| 余白の白色化 | デフォルトで有効：外周の余白バンド（最も左の文字より左、最も右の文字より右、最も上の文字より上、最も下の文字より下）を白に塗ります。ページ端に接し、反対側の端まで文字が一切ない帯だけが削除対象になるため、文字が誤って消されることはありません（背表紙の影やページ端の筋は除去されます）。文字と行・列を共有する領域には触れません。`--no-margin-whitening` / `--margin-pad` |
| 傾き検出の外周除外 | 角度検出時に画像の外周6%を除外。書籍スキャンに多い背表紙の影やページ端（長い直線バー）が、本文が疎なページでRadon投影を支配して逆方向回転を引き起こすのを防ぎます |
| PDF抽出時のリサイズ省略 | C#は抽出時にA4サイズ（2480×3508）にリサイズするが、両パイプラインとも内部高解像度（4960×7016）に正規化するため省略 |
| 傾き補正 | C#はImageMagick外部バイナリで高解像度画像に対して処理。本移植版はRadon変換をCythonに移植し、元の抽出画像でノイズ除去後に角度検出、高解像度画像に回転を適用 |

また、移植ミスなどにより、細かい動作が異なる場合があります。悪しからずご了承下さい。

## トラブルシューティング

<details>
<summary><b>Cythonビルドが失敗する</b></summary>

Cコンパイラがインストールされていることを確認してください：
- **macOS**: `xcode-select --install`
- **Linux**: `sudo apt install build-essential`
- **Windows**: Visual Studio Build Toolsをインストール

</details>


<details>
<summary><b>CUDAメモリ不足</b></summary>

タイルサイズを小さくしてみてください：
```bash
book-pdf-converter input.pdf output.pdf --tile 256
```

</details>

## ライセンス

このプロジェクトは、元の [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) と同じ **AGPL-3.0**（GNU Affero General Public License v3.0）でライセンスされています。

ライセンス全文は [LICENSE](LICENSE) を参照してください。

## 謝辞

このプロジェクトは、[登 大遊 (Daiyuu Nobori)](https://github.com/dnobori) 氏による [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) の Python/Cython 移植版です。

傾き補正のRadon変換アルゴリズムは [ImageMagick](https://imagemagick.org/) の `MagickCore/shear.c` から移植しました（Apache 2.0ライセンス）。

## 関連プロジェクト

- [DN_SuperBook_PDF_Converter](https://github.com/dnobori/DN_SuperBook_PDF_Converter) - 元のC#実装
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI画像鮮明化モデル
- [ImageMagick](https://github.com/ImageMagick/ImageMagick) - Radon変換アルゴリズムの移植元
