import numpy as np

START_OF_IMAGE = 0xFFD8
END_OF_IMAGE = 0xFFD9

ZIGZAG_ORDER = [
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]  # fmt: skip


def dct_matrix(n: int = 8) -> np.ndarray:
    mat = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            if u == 0:
                mat[u, x] = 1 / np.sqrt(n)
            else:
                mat[u, x] = np.sqrt(2 / n) * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return mat


def dct(block: np.ndarray, dct_mat: np.ndarray) -> np.ndarray:
    return dct_mat @ block @ dct_mat.T


def idct(block_dct: np.ndarray, dct_mat: np.ndarray) -> np.ndarray:
    return dct_mat.T @ block_dct @ dct_mat


def padding(img: np.ndarray, pad_w: int, pad_h: int) -> np.ndarray:
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


def zigzag_scan(arr8x8: np.ndarray) -> np.ndarray:
    return arr8x8.flatten()[ZIGZAG_ORDER]


def zigzag_scan_inv(arr1d: np.ndarray) -> np.ndarray:
    """

    >>> arr8x8 = np.arange(64).reshape(8,8)
    >>> (arr8x8 == zigzag_scan_inv(zigzag_scan(arr8x8))).all()
    True
    """
    return arr1d[np.argsort(ZIGZAG_ORDER)].reshape(8, 8)


def get_category(value: int | np.integer) -> int:
    """
    >>> get_category(0)
    0
    >>> get_category(1)
    1
    >>> get_category(-1)
    1
    >>> get_category(-7)
    3
    >>> get_category(8)
    4
    >>> get_category(-2047)
    11
    """

    return abs(int(value)).bit_length()


def get_encval_and_category(value: int) -> tuple[int, int]:
    """
    >>> val, _ = get_encval_and_category(0)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = get_encval_and_category(0b1)
    >>> f'{bin(val)}'
    '0b1'
    >>> val, _ = get_encval_and_category(-0b1)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = get_encval_and_category(-0b111)
    >>> f'{bin(val)}'
    '0b0'
    >>> val, _ = get_encval_and_category(0b1000)
    >>> f'{bin(val)}'
    '0b1000'
    >>> val, _ = get_encval_and_category(-0b11111111111)
    >>> f'{bin(val)}'
    '0b0'
    """
    if value == 0:
        return 0, 0

    cat = get_category(value)

    if value < 0:
        enc_val = (1 << cat) - 1 + value
    else:
        enc_val = value

    return enc_val, cat


def encode_runlength(ac_coeffs: np.ndarray) -> list[tuple[int, int, int]]:
    """
    return [(runlength, category, value), ...]

    >>> encode_runlength([0] * 63)
    [(0, 0, 0)]

    >>> xs1 = [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    >>> encode_runlength(xs1)
    [(1, 1, 1), (0, 1, 1), (0, 1, 1), (3, 2, 3), (15, 0, 0), (1, 1, 1), (0, 0, 0)]

    >>> xs2 = [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 8]
    >>> encode_runlength(xs2)
    [(1, 1, 1), (0, 1, 1), (0, 1, 1), (3, 2, 3), (15, 0, 0), (1, 1, 1), (3, 4, 8)]

    >>> xs3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
    >>> encode_runlength(xs3)
    [(15, 0, 0), (13, 4, 8)]

    >>> xs4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]
    >>> encode_runlength(xs4)
    [(15, 0, 0), (12, 4, 8), (0, 0, 0)]

    >>> xs5 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> encode_runlength(xs5)
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
            result.append((count_zero, get_category(coeff_i), coeff_i))
            if last_nonzero_idx < len(ac_coeffs) - 1:
                result.append((0, 0, 0))  # EOB
            break

        if ac_coeffs[i] == 0:
            count_zero += 1
            if count_zero == 16:
                result.append((15, 0, 0))
                count_zero = 0
        else:
            result.append((count_zero, get_category(coeff_i), coeff_i))
            count_zero = 0
        i += 1

    return result


def to_ycbcr(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        msg = f"Invalid image shape: {img.shape}. Expected RGB image."
        raise ValueError(msg)

    to_ycbcr = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ]
    )
    return (to_ycbcr @ img[..., None]).squeeze()


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
