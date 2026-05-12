from typing import Self

import numpy as np

from jpg.helper import (
    dct_matrix,
    encode_runlength,
    get_encval_and_category,
    padding,
    zigzag_scan,
)
from jpg.huffman_table import HuffmanTable
from jpg.quantization_table import QuantizationTable


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
        ).transpose(0, 3, 1, 4, 2, 5)  # (n_mcu_h, n_mcu_w, mcu_h, mcu_w, n, n)

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

    def mcu_encoder(self):
        for row_mcu in self.quantized_dct_blocks:
            for mcu in row_mcu:
                yield self._encode_mcu(mcu)
