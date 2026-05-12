import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from encoder import jpg_encode

matplotlib.use("QtAgg")

SAMPLING_SETTINGS = {
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

    if img.ndim == 3:
        sampling_setting = SAMPLING_SETTINGS["4:2:0"]
    elif img.ndim == 2:
        sampling_setting = SAMPLING_SETTINGS["grayscale"]
    else:
        msg = f"Invalid image shape: {img.shape}. Expected GrayScale(ndim=2) or RGB(ndim=3) image."
        raise ValueError(msg)

    jpg_bytes = jpg_encode(img, **sampling_setting, quality=50)

    with open("compressed.jpg", "wb") as f:
        f.write(jpg_bytes)


if __name__ == "__main__":
    main()
