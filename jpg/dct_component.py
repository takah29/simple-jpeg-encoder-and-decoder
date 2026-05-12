from typing import Self

import numpy as np

from jpg.helper import zigzag_scan
from jpg.huffman_table import HuffmanTable
from jpg.quantization_table import QuantizationTable


def _dct_matrix(n: int = 8) -> np.ndarray:
    mat = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            if u == 0:
                mat[u, x] = 1 / np.sqrt(n)
            else:
                mat[u, x] = np.sqrt(2 / n) * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return mat


def _dct(blocks: np.ndarray, dct_mat: np.ndarray) -> np.ndarray:
    return dct_mat @ blocks @ dct_mat.T


def _idct(dct_blocks: np.ndarray, dct_mat: np.ndarray) -> np.ndarray:
    return dct_mat.T @ dct_blocks @ dct_mat


def _padding(img: np.ndarray, pad_w: int, pad_h: int) -> np.ndarray:
    if img.ndim == 2:
        w_ext_shape = (1, pad_w)
        h_ext_shape = (pad_h, 1)
    elif img.ndim == 3:
        w_ext_shape = (1, pad_w, 1)
        h_ext_shape = (pad_h, 1, 1)
    else:
        msg = f"Invalid image shape: {img.shape}. Expected GrayScale(ndim=2) or RGB(ndim=3) image."
        raise ValueError(msg)

    w_ext = np.tile(img[:, -1:], w_ext_shape)
    img_ext = np.concatenate((img, w_ext), axis=1)

    h_ext = np.tile(img_ext[-1:], h_ext_shape)
    img_ext = np.concatenate((img_ext, h_ext), axis=0)

    return img_ext


def _get_category(value: int | np.integer) -> int:
    """
    >>> _get_category(0)
    0
    >>> _get_category(1)
    1
    >>> _get_category(-1)
    1
    >>> _get_category(-7)
    3
    >>> _get_category(8)
    4
    >>> _get_category(-2047)
    11
    """

    return abs(int(value)).bit_length()


def _get_encval_and_category(value: int) -> tuple[int, int]:
    """
    >>> val, _ = _get_encval_and_category(0)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = _get_encval_and_category(0b1)
    >>> f'{bin(val)}'
    '0b1'
    >>> val, _ = _get_encval_and_category(-0b1)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = _get_encval_and_category(-0b111)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = _get_encval_and_category(0b1000)
    >>> f'{bin(val)}'
    '0b1000'
    >>> val, _ = _get_encval_and_category(-0b11111111111)
    >>> f'{bin(val)}'
    '0b0'
    """
    if value == 0:
        return 0, 0

    cat = _get_category(value)

    if value < 0:
        enc_val = (1 << cat) - 1 + value
    else:
        enc_val = value

    return enc_val, cat


def _encode_runlength(ac_coeffs: np.ndarray) -> list[tuple[int, int, int]]:
    """
    return [(runlength, category, value), ...]

    >>> _encode_runlength([0] * 63)
    [(0, 0, 0)]

    >>> xs1 = [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    >>> _encode_runlength(xs1)
    [(1, 1, 1), (0, 1, 1), (0, 1, 1), (3, 2, 3), (15, 0, 0), (1, 1, 1), (0, 0, 0)]

    >>> xs2 = [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 8]
    >>> _encode_runlength(xs2)
    [(1, 1, 1), (0, 1, 1), (0, 1, 1), (3, 2, 3), (15, 0, 0), (1, 1, 1), (3, 4, 8)]

    >>> xs3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
    >>> _encode_runlength(xs3)
    [(15, 0, 0), (13, 4, 8)]

    >>> xs4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]
    >>> _encode_runlength(xs4)
    [(15, 0, 0), (12, 4, 8), (0, 0, 0)]

    >>> xs5 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> _encode_runlength(xs5)
    [(0, 1, 1), (0, 0, 0)]
    """
    ac_coeffs = np.asarray(ac_coeffs)

    nonzero_indices = np.nonzero(ac_coeffs != 0)[0]
    last_nonzero_idx = -1 if nonzero_indices.size == 0 else nonzero_indices[-1]

    if last_nonzero_idx == -1:
        return [(0, 0, 0)]

    count_zero = 0
    i = 0
    result = []
    while True:
        coeff_i = int(ac_coeffs[i])

        if last_nonzero_idx == i:
            result.append((count_zero, _get_category(coeff_i), coeff_i))
            if last_nonzero_idx < len(ac_coeffs) - 1:
                result.append((0, 0, 0))  # EOB
            break

        if ac_coeffs[i] == 0:
            count_zero += 1
            if count_zero == 16:
                result.append((15, 0, 0))
                count_zero = 0
        else:
            result.append((count_zero, _get_category(coeff_i), coeff_i))
            count_zero = 0
        i += 1

    return result


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
        mcu_h, mcu_w = mcu_size_hw
        pad_w = (mcu_w * 8) - 1 - ((img.shape[1] - 1) % (mcu_w * 8))
        pad_h = (mcu_h * 8) - 1 - ((img.shape[0] - 1) % (mcu_h * 8))
        img = _padding(img, pad_w, pad_h)

        img = cls._chroma_subsampling(img, sample_step_hw)
        img = cls._block_split(img, mcu_size_hw)

        # DCT
        dct_mat = _dct_matrix()
        dct_blocks = _dct(img, dct_mat)

        # Quantization
        quantized_dct_blocks = np.round(dct_blocks / quantization_table.values).astype(np.int32)

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
    def _push_bits(block_code, length, code_word, code_len) -> tuple[int, int]:
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
                ac_runlength_enc = _encode_runlength(ac_coeffs)

                dc_val, dc_cat = _get_encval_and_category(dc_diff)
                code_word, code_len = self.dc_huffman_table.table[dc_cat]

                block_code, length = self._push_bits(block_code, length, code_word, code_len)
                block_code, length = self._push_bits(block_code, length, dc_val, dc_cat)

                for runlen, cat, val in ac_runlength_enc:
                    symbol = (runlen << 4) | cat
                    code_word, code_len = self.ac_huffman_table.table[symbol]

                    block_code, length = self._push_bits(block_code, length, code_word, code_len)

                    if cat > 0:
                        encval, ac_cat = _get_encval_and_category(val)
                        block_code, length = self._push_bits(block_code, length, encval, ac_cat)

        return block_code, length

    def mcu_encoder(self):
        for row_mcu in self.quantized_dct_blocks:
            for mcu in row_mcu:
                yield self._encode_mcu(mcu)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
