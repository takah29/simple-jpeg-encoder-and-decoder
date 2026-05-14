from typing import Self

import numpy as np

from jpg.core.helper import (
    dct,
    dct_matrix,
    encode_runlength,
    get_encval_and_category,
    idct,
    padding,
    zigzag_scan,
)
from jpg.core.huffman_table import HuffmanTable
from jpg.core.quantization_table import QuantizationTable


class QuantizedBlocks:
    def __init__(
        self,
        quantized_blocks: np.ndarray,
    ) -> None:
        if quantized_blocks.ndim != 6:
            msg = f"Invalid quantized block shape: {quantized_blocks.shape}. Expected dimension 6."
            raise ValueError(msg)

        self.quantized_blocks = quantized_blocks

        self._last_dc = 0

    @classmethod
    def create(cls, mcu_list: list[np.ndarray], mcu_num_hw: tuple[int, int]) -> Self:
        if mcu_list[0].ndim != 4:
            msg = f"Invalid mcu shape: {mcu_list[0].shape}. Expected dimension 4."
            raise ValueError(msg)

        mcu_h, mcu_w = mcu_list[0].shape[:2]
        return cls(np.stack(mcu_list).reshape(*mcu_num_hw, mcu_h, mcu_w, 8, 8))

    @classmethod
    def from_image_component(
        cls,
        image_component: np.ndarray,
        quantization_table: QuantizationTable,
        mcu_size_hw: tuple[int, int],
        sample_step_hw: tuple[int, int],
    ) -> Self:
        if image_component.ndim != 2:
            msg = f"Invalid image shape: {image_component.shape}. Expected dimension 2."
            raise ValueError(msg)

        # padding
        mcu_h, mcu_w = mcu_size_hw
        pad_w = (mcu_w * 8) - 1 - ((image_component.shape[1] - 1) % (mcu_w * 8))
        pad_h = (mcu_h * 8) - 1 - ((image_component.shape[0] - 1) % (mcu_h * 8))
        img = padding(image_component, pad_w, pad_h)

        img = cls._chroma_subsampling(img, sample_step_hw)
        img = cls._block_split(img, mcu_size_hw)

        # DCT
        dct_mat = dct_matrix()
        dct_blocks = dct(img, dct_mat)

        # Quantization
        quantized_blocks = np.round(dct_blocks / quantization_table.values).astype(np.int32)

        return cls(quantized_blocks)

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
    def _chroma_upsampling(
        img: np.ndarray, sample_step_hw: tuple[int, int], method="nearest"
    ) -> np.ndarray:
        if method == "nearest":
            upsampled = np.kron(img, np.ones(sample_step_hw))
        else:
            raise ValueError("Unsupported upsampling method")

        return upsampled

    @staticmethod
    def _block_split(img: np.ndarray, mcu_size_hw: tuple[int, int]) -> np.ndarray:
        n = 8
        mcu_h, mcu_w = mcu_size_hw
        img = img.reshape(
            img.shape[0] // (mcu_h * n), mcu_h, n, img.shape[1] // (mcu_w * n), mcu_w, n
        ).transpose(0, 3, 1, 4, 2, 5)  # (n_mcu_h, n_mcu_w, mcu_h, mcu_w, n, n)

        return img

    def _merge_block(self, blocks: np.ndarray) -> np.ndarray:
        padded_shape = self.quantized_blocks.shape
        padded_img_h = padded_shape[0] * padded_shape[2] * 8
        padded_img_w = padded_shape[1] * padded_shape[3] * 8
        return blocks.transpose(0, 2, 4, 1, 3, 5).reshape(padded_img_h, padded_img_w)

    @staticmethod
    def _push_bits(block_code, length, code_word, code_len) -> tuple[int, int]:
        code_word = int(code_word)
        code_len = int(code_len)

        if code_len == 0:
            return block_code, length

        block_code = (block_code << code_len) | code_word
        length += code_len
        return block_code, length

    def _encode_mcu(
        self, mcu: np.ndarray, dc_huffman_table: HuffmanTable, ac_huffman_table: HuffmanTable
    ) -> tuple[int, int]:
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
                code_word, code_len = dc_huffman_table.table[dc_cat]

                block_code, length = self._push_bits(block_code, length, code_word, code_len)
                block_code, length = self._push_bits(block_code, length, dc_val, dc_cat)

                for runlen, cat, val in ac_runlength_enc:
                    symbol = (runlen << 4) | cat
                    code_word, code_len = ac_huffman_table.table[symbol]

                    block_code, length = self._push_bits(block_code, length, code_word, code_len)

                    if cat > 0:
                        encval, ac_cat = get_encval_and_category(val)
                        block_code, length = self._push_bits(block_code, length, encval, ac_cat)

        return block_code, length

    def mcu_encoder(self, dc_huffman_table: HuffmanTable, ac_huffman_table: HuffmanTable):
        for row_mcu in self.quantized_blocks:
            for mcu in row_mcu:
                yield self._encode_mcu(mcu, dc_huffman_table, ac_huffman_table)

    def to_image_component(
        self,
        img_shape: tuple[int, int],
        sample_step_hw: tuple[int, int],
        quantization_table: QuantizationTable,
    ) -> np.ndarray:
        dct_blocks = self.quantized_blocks * quantization_table.values
        dct_mat = dct_matrix()
        blocks = idct(dct_blocks, dct_mat)
        img = self._merge_block(blocks)
        img = self._chroma_upsampling(img, sample_step_hw)

        return img[: img_shape[0], : img_shape[1]]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
