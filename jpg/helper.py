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


def zigzag_scan(arr8x8: np.ndarray) -> np.ndarray:
    return arr8x8.flatten()[ZIGZAG_ORDER]


def zigzag_scan_inv(arr1d: np.ndarray) -> np.ndarray:
    """

    >>> arr8x8 = np.arange(64).reshape(8,8)
    >>> bool((arr8x8 == zigzag_scan_inv(zigzag_scan(arr8x8))).all())
    True
    """
    return arr1d[np.argsort(ZIGZAG_ORDER)].reshape(8, 8)


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
