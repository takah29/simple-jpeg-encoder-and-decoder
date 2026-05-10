import numpy as np
from huffman_table import HuffmanTable
from quantization_table import QuantizationTable
from pathlib import Path
from helper import to_ycbcr, padding, dct_matrix, flat_zigzag, encode_runlength


def jpg_encode(img: np.ndarray) -> np.ndarray:
    # RGB -> YCbCr
    img = to_ycbcr(img) + np.array([0, 128, 128])

    # padding
    pad_w = 7 - ((img.shape[1] - 1) % 8)
    pad_h = 7 - ((img.shape[0] - 1) % 8)
    img = padding(img, pad_w, pad_h)

    # block splitting
    img = img.reshape(img.shape[0] // 8, 8, img.shape[1] // 8, 8, img.shape[2])
    img = img.transpose(0, 2, 1, 3, 4)

    # DCT
    dct_mat = dct_matrix()
    dct_blocks = dct_mat @ img @ dct_mat.T

    # Quantization
    quantization_table = QuantizationTable.create(0, 0)
    dct_blocks /= quantization_table.values

    # build huffman table
    ydc_ht = HuffmanTable.create_from_file(Path("./huffman_code/ydc_hc.csv"), 1, 1)
    yac_ht = HuffmanTable.create_from_file(Path("./huffman_code/yac_hc.csv"), 1, 2)
    uvdc_ht = HuffmanTable.create_from_file(Path("./huffman_code/uvdc_hc.csv"), 2, 1)
    uvac_ht = HuffmanTable.create_from_file(Path("./huffman_code/uvac_hc.csv"), 2, 2)

    # Runlength encoding
    last_dc = 0
    for row_blocks in dct_blocks:
        for block in row_blocks:
            v = flat_zigzag(block)
            dc_diff = v[0] - last_dc
            last_dc = v[0]

            ac_coeffs = v[1:]
            runlength_enc = encode_runlength(ac_coeffs)

    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
