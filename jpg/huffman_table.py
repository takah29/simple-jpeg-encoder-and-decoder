import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Self


@dataclass
class HuffmanTable:
    table_class: int
    table_id: int
    huffman_table: dict[int, tuple[int, int]]

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
        pass


if __name__ == "__main__":
    huffman_table = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 1, 1)
    print(huffman_table)
