from dataclasses import dataclass
from typing import Self


@dataclass
class HuffmanTableId:
    component_id: int
    dc_huffman_table_id: int
    ac_huffman_table_id: int

    def to_bytes(self) -> bytes:
        return (
            self.component_id.to_bytes()
            + ((self.dc_huffman_table_id << 4) | self.ac_huffman_table_id).to_bytes()
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) != 2:
            raise ValueError("data must be 2 bytes long")

        component_id = data[0]
        dc_huffman_table_id = (data[1] >> 4) & 0x0F
        ac_huffman_table_id = data[1] & 0x0F

        return cls(component_id, dc_huffman_table_id, ac_huffman_table_id)


@dataclass
class StartOfScan:
    num_components: int
    huffman_table_ids: list[HuffmanTableId]

    start_of_spectral_selection: int = 0x00
    end_of_spectral_selection: int = 0x3F
    successive_approximation_bit_position_high: int = 0
    successive_approximation_bit_position_low: int = 0

    @classmethod
    def create(
        cls,
        image_shape: tuple[int, ...],
    ) -> Self:
        if len(image_shape) == 3:
            num_components = image_shape[2]
            huffman_table_ids = [
                HuffmanTableId(
                    component_id=i + 1, dc_huffman_table_id=min(i, 1), ac_huffman_table_id=min(i, 1)
                )
                for i in range(num_components)
            ]
        elif len(image_shape) == 2:
            num_components = 1
            huffman_table_ids = [
                HuffmanTableId(component_id=1, dc_huffman_table_id=0, ac_huffman_table_id=0)
            ]
        else:
            msg = f"Invalid image shape: {image_shape}. Expected GrayScale or RGB image."
            raise ValueError(msg)

        return cls(
            num_components=num_components,
            huffman_table_ids=huffman_table_ids,
        )

    def to_bytes(self) -> bytes:
        marker = 0xFFDA.to_bytes(2, "big")
        segment_length = (3 + self.num_components * 2 + 3).to_bytes(2, "big")

        huffman_table_id_bytes = b"".join(
            huffman_table_id.to_bytes() for huffman_table_id in self.huffman_table_ids
        )

        num_components_byte = self.num_components.to_bytes()
        spectral_selection_and_approximation_bytes = (
            self.start_of_spectral_selection.to_bytes()
            + self.end_of_spectral_selection.to_bytes()
            + (
                (self.successive_approximation_bit_position_high << 4)
                | self.successive_approximation_bit_position_low
            ).to_bytes()
        )

        return (
            marker
            + segment_length
            + num_components_byte
            + huffman_table_id_bytes
            + spectral_selection_and_approximation_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) < 6:
            raise ValueError("data is too short")

        marker = int.from_bytes(data[0:2], "big")
        if marker != 0xFFDA:
            raise ValueError("invalid marker")

        segment_length = int.from_bytes(data[2:4], "big")
        if len(data) != segment_length + 2:
            raise ValueError("invalid segment length")

        num_components = data[4]

        huffman_table_ids = []
        for i in range(num_components):
            start_idx = 5 + i * 2
            end_idx = start_idx + 2
            huffman_table_id_bytes = data[start_idx:end_idx]
            huffman_table_id = HuffmanTableId.from_bytes(huffman_table_id_bytes)
            huffman_table_ids.append(huffman_table_id)

        return cls(num_components, huffman_table_ids)


if __name__ == "__main__":
    sos = StartOfScan.create((256, 256, 3))
    sos_bytes = sos.to_bytes()
    print(sos)
    print(sos_bytes)
    sos_from_bytes = StartOfScan.from_bytes(sos_bytes)
    print(sos_from_bytes)
    assert sos == sos_from_bytes
