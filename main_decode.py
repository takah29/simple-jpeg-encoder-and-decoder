from pathlib import Path

import matplotlib
from PIL import Image

from jpg.decoder import jpg_decode

matplotlib.use("QtAgg")


def main():
    jpg_path = Path("compressed.jpg")
    with jpg_path.open("rb") as f:
        jpg_bytes = f.read()

    img = jpg_decode(jpg_bytes)

    print(img.shape, img.dtype)
    print(img)
    img = Image.fromarray(img)

    img.show()


if __name__ == "__main__":
    main()
