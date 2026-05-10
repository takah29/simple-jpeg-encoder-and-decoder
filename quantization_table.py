from dataclasses import dataclass
import numpy as np
from typing import Self


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


@dataclass
class QuantizationTable:
    precision: int
    table_id: int
    values: np.ndarray

    @classmethod
    def create(cls, precision: int, table_id: int) -> Self:
        if precision not in (0, 1):
            raise ValueError("precision must be 0 or 1")
        if table_id not in (0, 1):
            raise ValueError("table_id must be 0 or 1")

        return cls(precision=precision, table_id=table_id, values=_quantization_table())

    def to_bytes(self) -> bytes:
        pass


if __name__ == "__main__":
    q_table = QuantizationTable.create(0, 0)
    print(q_table)
