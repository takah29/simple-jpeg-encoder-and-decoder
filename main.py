import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from encoder import jpg_encode

matplotlib.use("QtAgg")

sub_sampling_ratio = {
    "4:2:0": {
        "mcu_size_hw_list": [(2, 2), (1, 1), (1, 1)],
        "sample_step_hw_list": [(1, 1), (2, 2), (2, 2)],
    },
    "4:2:2": {
        "mcu_size_hw_list": [(1, 2), (1, 1), (1, 1)],
        "sample_step_hw_list": [(1, 1), (1, 2), (1, 2)],
    },
    "4:4:4": {
        "mcu_size_hw_list": [(1, 1), (1, 1), (1, 1)],
        "sample_step_hw_list": [(1, 1), (1, 1), (1, 1)],
    },
    "grayscale": {
        "mcu_size_hw_list": [(1, 1)],
        "sample_step_hw_list": [(1, 1)],
    },
}


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
    img = np.array(img).astype(np.int32)

    # plt.imshow(img, vmin=0, vmax=255, cmap="gray")
    # plt.savefig("original.png")
    # plt.show()

    jpg_bytes = jpg_encode(img, **sub_sampling_ratio["4:2:0"])
    with open("compressed.jpg", "wb") as f:
        f.write(jpg_bytes)


if __name__ == "__main__":
    main()
