import numpy as np

START_OF_IMAGE = 0xFFDA
END_OF_IMAGE = 0xFFD9

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
    w_ext = np.tile(img[:, -1:], (1, pad_w, 1))
    print(w_ext.shape, img.shape)
    img_ext = np.concatenate((img, w_ext), axis=1)

    h_ext = np.tile(img_ext[-1:], (pad_h, 1, 1))
    img_ext = np.concatenate((img_ext, h_ext), axis=0)

    return img_ext


def flat_zigzag(arr1d: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        zigzag_order_inv = np.argsort(ZIGZAG_ORDER)
        return arr1d[zigzag_order_inv]

    return arr1d[ZIGZAG_ORDER]


def get_category(value: int) -> int:
    """カテゴリ計算

    >>> get_category(0)
    0
    >>> get_category(1)
    1
    >>> get_category(-1)
    1
    >>> get_category(2)
    2
    >>> get_category(-3)
    2
    >>> get_category(-7)
    3
    >>> get_category(8)
    4
    >>> get_category(-2047)
    11
    """

    return abs(value).bit_length()


def encode_runlength(ac_coeffs: np.ndarray) -> list:
    """ランレングス符号化

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
    to_ycbcr = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ]
    )
    return (to_ycbcr @ img[..., None]).squeeze()

def encode()