from pathlib import Path
from typing import Self
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
    to_ycbcr,
    chroma_subsampling,
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


class DctComponent:
    def __init__(
        self,
        quantized_dct_blocks: np.ndarray,
        quantization_table: QuantizationTable,
        dc_huffman_table: HuffmanTable,
        ac_huffman_table: HuffmanTable,
        mcu_size_hw: tuple[int, int],
        sample_step_hw: tuple[int, int],
    ) -> None:
        self.quantized_dct_blocks = quantized_dct_blocks
        self.quantization_table = quantization_table
        self.dc_huffman_table = dc_huffman_table
        self.ac_huffman_table = ac_huffman_table

        self.mcu_size_hw = mcu_size_hw
        self.sample_step_hw = sample_step_hw

        self._last_dc = 0

    @classmethod
    def create(
        cls,
        img: np.ndarray,
        quantization_table: QuantizationTable,
        dc_huffman_table: HuffmanTable,
        ac_huffman_table: HuffmanTable,
        mcu_size_hw: tuple[int, int],
        sample_step_hw: tuple[int, int],
    ) -> Self:
        if img.ndim != 2:
            msg = f"Invalid image shape: {img.shape}. Expected dimension 2."
            raise ValueError(msg)

        # padding
        pad_w = 7 - ((img.shape[1] - 1) % 8)
        pad_h = 7 - ((img.shape[0] - 1) % 8)
        img = padding(img, pad_w, pad_h)

        img = cls._chroma_subsampling(img, sample_step_hw)
        img = cls._block_split(img, mcu_size_hw)

        # DCT
        dct_mat = dct_matrix()
        dct_blocks = dct_mat @ img @ dct_mat.T

        # Quantization
        quantized_dct_blocks = (dct_blocks / quantization_table.values + 0.5).astype(np.int32)

        return cls(
            quantized_dct_blocks,
            quantization_table,
            dc_huffman_table,
            ac_huffman_table,
            mcu_size_hw,
            sample_step_hw,
        )

    @staticmethod
    def _chroma_subsampling(img: np.ndarray, sampling_step_hw: tuple[int, int]) -> np.ndarray:
        h, w = sampling_step_hw

        img = (
            img.reshape(img.shape[0] // h, h, img.shape[1] // w, w)
            .transpose(0, 2, 1, 3)
            .mean(axis=(2, 3))
        )

        return img

    @staticmethod
    def _block_split(img: np.ndarray, mcu_size_hw: tuple[int, int]) -> np.ndarray:
        n = 8
        mcu_h, mcu_w = mcu_size_hw
        img = img.reshape(
            img.shape[0] // (mcu_h * n), mcu_h, n, img.shape[1] // (mcu_w * n), mcu_w, n
        ).transpose(0, 3, 1, 4, 2, 5)  # (n_blocks_h, n_blocks_w, h_size, w_size, n, n)

        return img

    @staticmethod
    def push_bits(block_code, length, code_word, code_len) -> tuple[int, int]:
        code_word = int(code_word)
        code_len = int(code_len)

        if code_len == 0:
            return block_code, length

        block_code = (block_code << code_len) | code_word
        length += code_len
        return block_code, length

    def _encode_mcu(self, mcu: np.ndarray) -> tuple[int, int]:
        block_code = 0
        length = 0
        for row_blocks in mcu:
            for block in row_blocks:
                v = zigzag_scan(block)
                dc_diff = v[0] - self._last_dc
                self._last_dc = v[0]

                ac_coeffs = v[1:]
                ac_runlength_enc = encode_runlength(ac_coeffs)

                dc_val, dc_cat = get_encval_and_category(dc_diff)
                code_word, code_len = self.dc_huffman_table.table[dc_cat]

                block_code, length = self.push_bits(block_code, length, code_word, code_len)
                block_code, length = self.push_bits(block_code, length, dc_val, dc_cat)

                for runlen, cat, val in ac_runlength_enc:
                    symbol = (runlen << 4) | cat
                    code_word, code_len = self.ac_huffman_table.table[symbol]

                    block_code, length = self.push_bits(block_code, length, code_word, code_len)

                    if cat > 0:
                        encval, ac_cat = get_encval_and_category(val)
                        block_code, length = self.push_bits(block_code, length, encval, ac_cat)

        return block_code, length

    def mcu_iterator(self):
        for row_mcu in self.quantized_dct_blocks:
            for mcu in row_mcu:
                yield self._encode_mcu(mcu)


def _encode_dct_blocks(
    dct_component_list: list[DctComponent],
) -> bytes:
    bit_writer = JpegBitWriter()

    for mcu_bits_list in zip(*[x.mcu_iterator() for x in dct_component_list]):
        for mcu_bits in mcu_bits_list:
            bit_writer.write_bits(*mcu_bits)

    bit_writer.finalize()

    return bytes(bit_writer.output)


def jpg_encode(
    img: np.ndarray, mcu_size_hw: list[tuple[int, int]], sample_step_hw: list[tuple[int, int]]
) -> bytes:
    start_of_scan = StartOfScan.create(img.shape)
    main_quantization_table = QuantizationTable.create(0, 0, True, 50)
    sub_quantization_table = QuantizationTable.create(0, 1, False, 50)

    frame_information = FrameInformation.create(img.shape, 8)

    dc_huffman_table0 = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 0, 0)
    dc_huffman_table1 = HuffmanTable.from_file(Path("./huffman_code/uvdc_hc.csv"), 0, 1)

    ac_huffman_table0 = HuffmanTable.from_file(Path("./huffman_code/yac_hc.csv"), 1, 0)
    ac_huffman_table1 = HuffmanTable.from_file(Path("./huffman_code/uvac_hc.csv"), 1, 1)

    img = img.astype(np.float64)
    if img.ndim == 3:
        # RGB -> YCbCr
        img = to_ycbcr(img) + np.array([-128, 0, 0])

    main_component = DctComponent.create(
        img[:, :, 0],
        main_quantization_table,
        dc_huffman_table0,
        ac_huffman_table0,
        mcu_size_hw[0],
        sample_step_hw[0],
    )
    sub_component1 = DctComponent.create(
        img[:, :, 1],
        sub_quantization_table,
        dc_huffman_table1,
        ac_huffman_table1,
        mcu_size_hw[1],
        sample_step_hw[1],
    )
    sub_component2 = DctComponent.create(
        img[:, :, 2],
        sub_quantization_table,
        dc_huffman_table1,
        ac_huffman_table1,
        mcu_size_hw[2],
        sample_step_hw[2],
    )

    # Encode DCT blocks
    encoded_dct_blocks = _encode_dct_blocks([main_component, sub_component1, sub_component2])

    return b"".join(
        [
            START_OF_IMAGE.to_bytes(2, "big"),
            main_quantization_table.to_bytes(),
            sub_quantization_table.to_bytes(),
            frame_information.to_bytes(),
            dc_huffman_table0.to_bytes(),
            dc_huffman_table1.to_bytes(),
            ac_huffman_table0.to_bytes(),
            ac_huffman_table1.to_bytes(),
            start_of_scan.to_bytes(),
            encoded_dct_blocks,
            END_OF_IMAGE.to_bytes(2, "big"),
        ]
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
