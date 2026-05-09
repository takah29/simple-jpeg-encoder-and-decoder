import numpy as np


def _quantization_table(quality: int = 50) -> np.ndarray:
    q_base = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    f = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.clip(np.floor((q_base * f + 50) / 100), 1.0, 255.0)

    return q


def _dct_matrix(n: int = 8) -> np.ndarray:
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


def _pad(img: np.ndarray, pad_w: int, pad_h: int) -> np.ndarray:
    w_ext = np.tile(img[:, -1:], (1, pad_w, 1))
    print(w_ext.shape, img.shape)
    img_ext = np.concatenate((img, w_ext), axis=1)

    h_ext = np.tile(img_ext[-1:], (pad_h, 1, 1))
    img_ext = np.concatenate((img_ext, h_ext), axis=0)

    return img_ext


def _flat_zigzag(block: np.ndarray) -> np.ndarray:
    ZIGZAG_ORDER = [
        0,  1,  5,  6, 14, 15, 27, 28,
        2,  4,  7, 13, 16, 26, 29, 42,
        3,  8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ]  # fmt: skip

    return block.flatten()[ZIGZAG_ORDER]


def _get_category(value: int) -> int:
    """カテゴリ計算

    >>> _get_category(0)
    0
    >>> _get_category(1)
    1
    >>> _get_category(-1)
    1
    >>> _get_category(2)
    2
    >>> _get_category(-3)
    2
    >>> _get_category(-7)
    3
    >>> _get_category(8)
    4
    >>> _get_category(-2047)
    11
    """

    return abs(value).bit_length()


def _encode_runlength(ac_coeffs: np.ndarray) -> list:
    """ランレングス符号化

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


def jpg_encode(img: np.ndarray) -> np.ndarray:
    # RGB -> YCbCr
    to_ycbcr = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
    )
    # img = (to_ycbcr @ img[..., None] + np.array([0, 128, 128])[..., None]).squeeze()

    # padding
    pad_w = 7 - ((img.shape[1] - 1) % 8)
    pad_h = 7 - ((img.shape[0] - 1) % 8)
    img = _pad(img, pad_w, pad_h)

    img = img.reshape(img.shape[0] // 8, 8, img.shape[1] // 8, 8, img.shape[2])
    img = img.transpose(0, 2, 1, 3, 4)

    dct_mat = _dct_matrix()
    dct_blocks = dct_mat @ img @ dct_mat.T
    dct_blocks /= _quantization_table()

    last_dc = 0
    for row_blocks in dct_blocks:
        for block in row_blocks:
            v = _flat_zigzag(block)
            dc_diff = v[0] - last_dc
            last_dc = v[0]

            ac_coeffs = v[1:]
            runlength_enc = _encode_runlength(ac_coeffs)

    return img


if __name__ == "__main__":
    import doctest

    doctest.testmod()
