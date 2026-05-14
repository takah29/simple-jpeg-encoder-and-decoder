import argparse
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

from jpg.decoder import jpg_decode

matplotlib.use("QtAgg")


def main():
    parser = argparse.ArgumentParser(description="Jpeg Decoder")
    parser.add_argument("input_file", help="Input JPEG file", type=Path)

    args = parser.parse_args()

    with args.input_file.open("rb") as f:
        jpg_bytes = f.read()

    img = jpg_decode(jpg_bytes)
    img = Image.fromarray(img)

    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
