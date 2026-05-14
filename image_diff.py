import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jpg.decoder import jpg_decode
from jpg.encoder import jpg_encode


def calculate_metrics(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0.0, 100.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr


def get_rgb_diff_map(img1: np.ndarray, img2: np.ndarray, scale: int = 5) -> np.ndarray:
    diff = np.abs(img1 - img2) * scale
    return np.clip(diff, 0, 255).astype(np.uint8)


def evaluate_scenario(orig_np, mcu_settings, quality, title):
    jpg_bytes = jpg_encode(orig_np, mcu_settings, quality=quality)
    mine_decoded = jpg_decode(jpg_bytes)

    pil_img = Image.open(io.BytesIO(jpg_bytes))
    pil_decoded = np.array(pil_img)

    h, w = orig_np.shape[:2]
    mine_decoded = mine_decoded[:h, :w]
    pil_decoded = pil_decoded[:h, :w]

    mse_m, psnr_m = calculate_metrics(orig_np.astype(np.float64), mine_decoded.astype(np.float64))

    mse_p, psnr_p = calculate_metrics(
        pil_decoded.astype(np.float64), mine_decoded.astype(np.float64)
    )

    return {
        "mine": mine_decoded,
        "pil": pil_decoded,
        "mse_mine": mse_m,
        "psnr_mine": psnr_m,
        "mse_vs_pil": mse_p,
        "psnr_vs_pil": psnr_p,
        "title": title,
        "size_kb": len(jpg_bytes) / 1024,
    }


def main():
    parser = argparse.ArgumentParser(description="Jpeg Encoder")
    parser.add_argument("input_file", help="Input image file", type=Path)

    args = parser.parse_args()

    if not args.input_file.exists():
        print("Image not found.")
        return

    orig_img = Image.open(args.input_file).convert("RGB")
    orig_np = np.array(orig_img)

    scenarios = [
        {
            "name": "Scenario 1: Perfect Fidelity (4:4:4, Q100)",
            "subsampling_ratio": "4:4:4",
            "quality": 100,
            "diff_scale": 50,  # 誤差が非常に小さいので50倍に強調
        },
        {
            "name": "Scenario 2: Standard Compression (4:2:0, Q50)",
            "subsampling_ratio": "4:2:0",
            "quality": 50,
            "diff_scale": 10,  # 誤差が見えるので10倍に強調
        },
    ]

    results = []
    for sc in scenarios:
        print(f"Evaluating {sc['name']}...")
        res = evaluate_scenario(orig_np, sc["subsampling_ratio"], sc["quality"], sc["name"])
        res["diff_scale"] = sc["diff_scale"]
        results.append(res)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, res in enumerate(results):
        axes[i, 0].imshow(res["mine"])
        axes[i, 0].set_title(f"{res['title']}\nCustom Decoder (PSNR: {res['psnr_mine']:.2f}dB)")

        diff_mine = get_rgb_diff_map(
            orig_np.astype(np.float64), res["mine"].astype(np.float64), scale=res["diff_scale"]
        )
        axes[i, 1].imshow(diff_mine)
        axes[i, 1].set_title(
            f"Error Map (vs Original) x{res['diff_scale']}\nMSE: {res['mse_mine']:.4f}"
        )

        diff_vs_pil = get_rgb_diff_map(
            res["pil"].astype(np.float64), res["mine"].astype(np.float64), scale=res["diff_scale"]
        )
        axes[i, 2].imshow(diff_vs_pil)
        axes[i, 2].set_title(
            f"Consistency Map (vs PIL) x{res['diff_scale']}\nMSE vs PIL: {res['mse_vs_pil']:.4f}"
        )

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("FINAL EVALUATION REPORT")
    print("=" * 50)
    for res in results:
        print(f"\n[{res['title']}]")
        print(f"  - File Size        : {res['size_kb']:.2f} KB")
        print(f"  - PSNR (Custom)    : {res['psnr_mine']:.2f} dB")
        print(f"  - MSE (vs Original): {res['mse_mine']:.4f}")
        print(f"  - MSE (vs PIL)     : {res['mse_vs_pil']:.4f} (Lower is better)")
        if res["mse_vs_pil"] < 1.0:
            print("  - Result: High Consistency with Standard!")
        else:
            print("  - Result: Deviation from Standard found (likely due to Level Shift/Matrices)")


if __name__ == "__main__":
    main()
