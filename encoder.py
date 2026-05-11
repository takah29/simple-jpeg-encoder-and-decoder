from pathlib import Path

import numpy as np

from jpg.frame_information import FrameInformation
from jpg.helper import (
    END_OF_IMAGE,
    START_OF_IMAGE,
    dct_matrix,
    encode_runlength,
    get_encval_and_category,
    zigzag_scan,
    padding,
    # to_ycbcr,
)
from jpg.huffman_table import HuffmanTable
from jpg.quantization_table import QuantizationTable
from jpg.start_of_scan import StartOfScan


class JpegBitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.buffer = 0
        self.bit_count = 0

    def write_bits(self, code: int | np.uint8, length: int | np.uint8) -> None:
        code = int(code)
        length = int(length)

        if length == 0:
            return

        self.buffer = (self.buffer << length) | code
        self.bit_count += length

        while self.bit_count >= 8:
            byte = (self.buffer >> (self.bit_count - 8)) & 0xFF
            self.output.append(byte)

            # byte stuffing
            if byte == 0xFF:
                self.output.append(0x00)

            self.bit_count -= 8
            bit_mask = (1 << self.bit_count) - 1
            self.buffer &= bit_mask

    def finalize(self) -> None:
        if self.bit_count > 0:
            shift = 8 - self.bit_count
            byte_padded = ((self.buffer << shift) | ((1 << shift) - 1)) & 0xFF
            self.output.append(byte_padded)

            # byte stuffing
            if byte_padded == 0xFF:
                self.output.append(0x00)

            self.buffer = 0
            self.bit_count = 0


def _encode_dct_blocks(
    dct_blocks: np.ndarray, huffman_table_dc: HuffmanTable, huffman_table_ac: HuffmanTable
) -> bytes:
    bit_writer = JpegBitWriter()
    last_dc = 0
    for row_blocks in dct_blocks:
        for block in row_blocks:
            v = zigzag_scan(block)
            dc_diff = v[0] - last_dc
            last_dc = v[0]

            ac_coeffs = v[1:]
            ac_runlength_enc = encode_runlength(ac_coeffs)

            dc_val, dc_cat = get_encval_and_category(dc_diff)
            code_word, code_len = huffman_table_dc.table[dc_cat]

            bit_writer.write_bits(code_word, code_len)
            bit_writer.write_bits(dc_val, dc_cat)

            for runlen, cat, val in ac_runlength_enc:
                symbol = (runlen << 4) | cat
                code_word, code_len = huffman_table_ac.table[symbol]

                bit_writer.write_bits(code_word, code_len)

                if cat > 0:
                    encval, ac_cat = get_encval_and_category(val)
                    bit_writer.write_bits(encval, ac_cat)

    bit_writer.finalize()

    return bytes(bit_writer.output)


def jpg_encode(img: np.ndarray) -> bytes:
    start_of_scan = StartOfScan.create(img.shape)
    quantization_table = QuantizationTable.create(0, 0, quality=50)
    frame_information = FrameInformation.create(img.shape, 8)
    huffman_table_dc = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 0, 0)
    huffman_table_ac = HuffmanTable.from_file(Path("./huffman_code/yac_hc.csv"), 1, 0)

    # RGB -> YCbCr
    # img = to_ycbcr(img) + np.array([-128, 0, 0])
    img -= 128

    # padding
    pad_w = 7 - ((img.shape[1] - 1) % 8)
    pad_h = 7 - ((img.shape[0] - 1) % 8)
    img = padding(img, pad_w, pad_h)

    # block splitting
    img = img.reshape(img.shape[0] // 8, 8, img.shape[1] // 8, 8)
    img = img.transpose(0, 2, 1, 3)

    # DCT
    dct_mat = dct_matrix()
    dct_blocks = dct_mat @ img @ dct_mat.T

    # Quantization
    dct_blocks = (dct_blocks / quantization_table.values + 0.5).astype(np.int32)

    # Encode DCT blocks
    encoded_dct_blocks = _encode_dct_blocks(dct_blocks, huffman_table_dc, huffman_table_ac)

    return b"".join(
        [
            START_OF_IMAGE.to_bytes(2, "big"),
            quantization_table.to_bytes(),
            frame_information.to_bytes(),
            huffman_table_dc.to_bytes(),
            huffman_table_ac.to_bytes(),
            start_of_scan.to_bytes(),
            encoded_dct_blocks,
            END_OF_IMAGE.to_bytes(2, "big"),
        ]
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
