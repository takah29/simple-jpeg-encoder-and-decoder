from dataclasses import dataclass
from typing import Self

import numpy as np

from jpg.helper import zigzag_scan, zigzag_scan_inv


def _quantization_table(quality: int) -> np.ndarray:
    if not (0 < quality <= 100):
        msg = f"Invalid quality: {quality}. Expected 0 < quality < 50."
        raise ValueError(msg)

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
    scale = 50 / quality if quality <= 50 else (100 - quality) / 50
    q_scaled = scale * q_base
    q = np.clip(np.floor(q_scaled + 0.5), 1.0, 255.0).astype(np.int32)

    return q


@dataclass
class QuantizationTable:
    MARKER = 0xFFDB

    precision: int
    table_id: int
    values: np.ndarray  # shape = (8, 8)

    @classmethod
    def create(cls, precision: int, table_id: int, quality: int = 90) -> Self:
        if precision not in (0, 1):
            raise ValueError("precision must be 0 or 1")
        if table_id not in (0, 1):
            raise ValueError("table_id must be 0 or 1")

        return cls(precision, table_id, _quantization_table(quality))

    def to_bytes(self) -> bytes:
        marker_bytes = QuantizationTable.MARKER.to_bytes(2, "big")
        segment_length = (3 + self.values.size * (2 if self.precision == 1 else 1)).to_bytes(
            2, "big"
        )
        info = ((self.precision << 4) | self.table_id).to_bytes()

        if self.precision == 0:
            table_bytes = zigzag_scan(self.values).astype(np.uint8).tobytes()
        else:
            table_bytes = zigzag_scan(self.values).astype(">u2").tobytes()

        return marker_bytes + segment_length + info + table_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) < 4:
            raise ValueError("data is too short")

        marker = int.from_bytes(data[0:2], "big")
        if marker != cls.MARKER:
            raise ValueError("invalid marker")

        segment_length = int.from_bytes(data[2:4], "big")
        if len(data) != segment_length + 2:
            raise ValueError("invalid segment length")

        info = data[4]
        precision = (info >> 4) & 0x0F
        table_id = info & 0x0F

        if precision == 0:
            arr1d = np.frombuffer(data[5:], dtype=np.uint8)
        else:
            arr1d = np.frombuffer(data[5:], dtype=">u2")

        values = zigzag_scan_inv(arr1d).reshape(8, 8)

        return cls(precision=precision, table_id=table_id, values=values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantizationTable):
            return NotImplemented

        return (
            self.precision == other.precision
            and self.table_id == other.table_id
            and np.array_equal(self.values, other.values)
        )


if __name__ == "__main__":
    q_table = QuantizationTable.create(0, 0)
    print(q_table)

    assert q_table == QuantizationTable.from_bytes(q_table.to_bytes())
