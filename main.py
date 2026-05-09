import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
from encoder import jpg_encode

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
    img = jpg_encode(np.array(img))

    plt.imshow(img)
    plt.show()

    img_arr = jpg_encode(np.array(img))


if __name__ == "__main__":
    main()
