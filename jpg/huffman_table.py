import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Self


@dataclass
class HuffmanTable:
    MARKER = 0xFFC4

    table_class: int
    table_id: int
    huffman_table: dict[int, tuple[int, int]]  # symbol -> (code_word, code_len)

    @classmethod
    def from_file(cls, filepath: Path, table_class: int, table_id: int) -> Self:
        return HuffmanTable(
            table_class=table_class,
            table_id=table_id,
            huffman_table=cls._create_huffman_table(filepath),
        )

    @staticmethod
    def _create_huffman_table(filepath: Path):
        result = {}
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                result[i] = (int(row["code_word"]), int(row["code_len"]))

        return result

    def to_bytes(self) -> bytes:
        num_code_len = 16

        marker_bytes = HuffmanTable.MARKER.to_bytes(2, "big")
        segment_length = (3 + num_code_len + len(self.huffman_table)).to_bytes(2, "big")

        info = ((self.table_class << 4) | self.table_id).to_bytes()

        code_length_count_list = [0] * num_code_len
        for _, code_len in self.huffman_table.values():
            code_length_count_list[code_len - 1] += 1

        code_length_count_list_bytes = bytes(code_length_count_list)
        sorted_items = sorted(self.huffman_table.items(), key=lambda x: x[1][1])
        symbol_bytes = bytes(symbol for symbol, _ in sorted_items)

        return marker_bytes + segment_length + info + code_length_count_list_bytes + symbol_bytes

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
        table_class = (info >> 4) & 0x0F
        table_id = info & 0x0F

        code_length_count_list = list(data[5:21])
        symbol_list = list(data[21 : 21 + sum(code_length_count_list)])

        huffman_table = {}
        start_idx = 0
        code_word = 0
        for code_len, count in enumerate(code_length_count_list, start=1):
            for symbol in symbol_list[start_idx : start_idx + count]:
                huffman_table[symbol] = (code_word, code_len)
                code_word += 1

            start_idx += count
            code_word <<= 1

        return cls(table_class, table_id, huffman_table)


if __name__ == "__main__":
    huffman_table = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 1, 1)
    print(huffman_table)
    huffman_table_bytes = huffman_table.to_bytes()
    print(huffman_table_bytes)
    huffman_table_from_bytes = HuffmanTable.from_bytes(huffman_table_bytes)
    print(huffman_table_from_bytes)
    assert huffman_table == huffman_table_from_bytes
