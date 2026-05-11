import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from encoder import jpg_encode
from jpg.helper import dct_matrix

matplotlib.use("QtAgg")


def main():
    img_path = Path("earthmap.jpg")
    if not img_path.exists():
        print("Image not found. Downloading...")
        try:
            subprocess.run(["wget", "https://raytracing.github.io/images/earthmap.jpg"])
        except Exception:
            print("Failed to download the image.")
            return

    img = Image.open(img_path)
    img = np.array(img.convert("L")).astype(np.int32)

    # plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    # plt.savefig("original.png")
    # plt.show()

    jpg_bytes = jpg_encode(img)
    with open("compressed.jpg", "wb") as f:
        f.write(jpg_bytes)


if __name__ == "__main__":
    main()
