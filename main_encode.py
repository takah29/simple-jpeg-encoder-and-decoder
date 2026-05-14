import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from jpg.encoder import jpg_encode

matplotlib.use("QtAgg")


def main():
    parser = argparse.ArgumentParser(description="Jpeg Encoder")
    parser.add_argument("input_file", help="Input image file", type=Path)
    parser.add_argument(
        "--subsampling_ratio",
        "-s",
        help="Chroma subsampling ratio",
        choices=["4:4:4", "4:2:2", "4:2:0"],
        type=str,
        default="4:2:0",
    )

    args = parser.parse_args()

    img = Image.open(args.input_file)
    img = np.array(img)

    subsampling_type = args.subsampling_ratio if img.ndim == 3 else "grayscale"
    jpg_bytes = jpg_encode(img, subsampling_type, quality=50)

    with open("compressed.jpg", "wb") as f:
        f.write(jpg_bytes)


if __name__ == "__main__":
    main()
