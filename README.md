# Simple JPEG Encoder & Decoder

A from-scratch Python implementation of a Baseline JPEG encoder and decoder. This project focuses on understanding the JPEG compression pipeline.

## Overview

- **Baseline JPEG Support**: Full implementation of the standard sequential JPEG process.
- **Chroma Subsampling**: Supports 4:4:4, 4:2:2, and 4:2:0 subsampling ratios.
- **Grayscale Support**: Automatic handling of single-channel grayscale images.
- **Quality Control**: Adjustable quality factor (1-100) using standard quantization scaling.
- **Limitations**:
  - **Progressive JPEG** is not supported (Baseline Sequential only).
  - **Restart Intervals** (DRI/RST markers) are not supported.

## Setup

This project uses the `uv` project management tool. To install dependencies and set up the environment, run:

```bash
uv sync
```

## Usage

### 1. Encoding an Image
Convert a standard image (PNG, JPG, etc.) into a custom-encoded JPEG file.

```bash
uv run main_encode.py <input_image_path> --quality 90 --subsampling_ratio 4:2:0 --output_file compressed.jpg
```

**Arguments:**
- `--quality` / `-q`: Compression quality (1-100). Default is 90.
- `--subsampling_ratio` / `-s`: Chroma subsampling ratio (`4:4:4`, `4:2:2`, `4:2:0`).

### 2. Decoding an Image
View the contents of a JPEG file using the custom decoder.

```bash
uv run main_decode.py compressed.jpg
```

### 3. Evaluation and Metrics
Run the evaluation script to compare the custom encoder/decoder against the original image and the industry-standard PIL (Pillow) implementation.

```bash
uv run image_diff.py <input_image_path>
```

This script generates:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality.
- **Error Maps**: Visualizes the difference between the original and the decoded image.
- **Consistency Map**: Visualizes the difference between this implementation and PIL.