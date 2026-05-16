import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np


@dataclass
class HuffmanTable:
    MARKER = 0xFFC4

    table_class: int
    table_id: int
    table: dict[int, tuple[int, int]]  # symbol -> (code_word, code_len)

    @classmethod
    def from_file(cls, filepath: Path, table_class: int, table_id: int) -> Self:
        return HuffmanTable(
            table_class=table_class,
            table_id=table_id,
            table=cls._create_huffman_table(filepath),
        )

    @staticmethod
    def _create_huffman_table(filepath: Path):
        result = {}
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                code_len = int(row["code_len"])
                if code_len == 0:
                    continue

                result[i] = (int(row["code_word"]), code_len)

        return result

    def to_bytes(self) -> bytes:
        num_code_len = 16

        marker_bytes = HuffmanTable.MARKER.to_bytes(2, "big")
        info = ((self.table_class << 4) | self.table_id).to_bytes()

        code_length_count_list = [0] * num_code_len
        for _, code_len in self.table.values():
            code_length_count_list[code_len - 1] += 1

        code_length_count_list_bytes = bytes(code_length_count_list)
        sorted_items = sorted(self.table.items(), key=lambda x: (x[1][1], x[0]))
        symbol_bytes = bytes(symbol for symbol, _ in sorted_items)

        segment_length = (3 + num_code_len + len(symbol_bytes)).to_bytes(2, "big")

        return marker_bytes + segment_length + info + code_length_count_list_bytes + symbol_bytes

    @classmethod
    def _read_table_data(cls, data: bytes, table_start_pos: int) -> tuple[Self, int]:
        info = data[table_start_pos]
        table_class = (info >> 4) & 0x0F
        table_id = info & 0x0F

        code_length_count_list = list(data[table_start_pos + 1 : table_start_pos + 17])
        huffman_code_data_length = sum(code_length_count_list)
        table_segment_length = 17 + huffman_code_data_length
        symbol_start = table_start_pos + 17
        symbol_list = list(data[symbol_start : symbol_start + huffman_code_data_length])

        huffman_table = {}
        start_idx = 0
        code_word = 0
        for code_len, count in enumerate(code_length_count_list, start=1):
            for symbol in symbol_list[start_idx : start_idx + count]:
                huffman_table[symbol] = (code_word, code_len)
                code_word += 1

            start_idx += count
            code_word <<= 1

        return cls(table_class, table_id, huffman_table), table_segment_length

    @classmethod
    def from_bytes(cls, data: bytes) -> list[Self]:
        if len(data) < 4:
            raise ValueError("data is too short")

        marker = int.from_bytes(data[0:2], "big")
        if marker != cls.MARKER:
            raise ValueError("invalid marker")

        segment_length = int.from_bytes(data[2:4], "big")
        if len(data) != segment_length + 2:
            raise ValueError("invalid segment length")

        current_pos = 4
        huffman_tables = []
        while current_pos < segment_length + 2:
            huffman_table, table_segment_length = cls._read_table_data(data, current_pos)
            huffman_tables.append(huffman_table)
            current_pos += table_segment_length

        return huffman_tables


@dataclass
class LookupTable:
    symbols: list[int]
    min_codes: dict[int, int]
    max_codes: dict[int, int]
    code_len_start_indices: np.ndarray

    @classmethod
    def from_huffman_table(cls, huffman_table: HuffmanTable) -> Self:
        sorted_items = sorted(huffman_table.table.items(), key=lambda x: (x[1][1], x[1][0]))

        symbols = []
        min_codes = {}
        max_codes = {}
        code_len_start_indices = [0] * 17

        for symbol, val in sorted_items:
            symbols.append(symbol)
            code_word, code_len = val

            if code_len not in min_codes:
                min_codes[code_len] = code_word

            max_codes[code_len] = code_word
            code_len_start_indices[code_len] += 1

        return cls(
            symbols=symbols,
            min_codes=min_codes,
            max_codes=max_codes,
            code_len_start_indices=np.cumsum([0] + code_len_start_indices),
        )

    def contains(self, code, code_len):
        return code_len in self.max_codes and code <= self.max_codes[code_len]

    def get_symbol(self, code: int, code_len: int) -> int:
        idx = self.code_len_start_indices[code_len] + (code - self.min_codes[code_len])
        return self.symbols[idx]


if __name__ == "__main__":
    huffman_table_dc1 = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 0, 0)
    huffman_table_ac1 = HuffmanTable.from_file(Path("./huffman_code/yac_hc.csv"), 1, 0)
    huffman_table_dc2 = HuffmanTable.from_file(Path("./huffman_code/uvdc_hc.csv"), 0, 1)
    huffman_table_ac2 = HuffmanTable.from_file(Path("./huffman_code/uvac_hc.csv"), 1, 1)

    huffman_table_list = {
        "dc1": huffman_table_dc1,
        "ac1": huffman_table_ac1,
        "dc2": huffman_table_dc2,
        "ac2": huffman_table_ac2,
    }

    for name, table in huffman_table_list.items():
        print(f"===== {name}=====")
        print(table)
        table_bytes = table.to_bytes()
        table_from_bytes = HuffmanTable.from_bytes(table_bytes)

        assert table == table_from_bytes[0]
