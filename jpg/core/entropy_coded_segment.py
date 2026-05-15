from collections.abc import Iterator

import numpy as np

from jpg.core.helper import JpegBitWriter, decode_encval, zigzag_scan_inv
from jpg.core.huffman_table import HuffmanTable, LookupTable
from jpg.core.quantized_blocks import QuantizedBlocks
from jpg.core.start_of_frame import StartOfFrame
from jpg.core.start_of_scan import StartOfScan


def to_entropy_coded_segment(
    dct_components: list[QuantizedBlocks], component_huffman_tables
) -> bytes:
    bit_writer = JpegBitWriter()

    for mcu_bits_list in zip(
        *[
            comp.mcu_encoder(ht["dc_huffman_table"], ht["ac_huffman_table"])
            for comp, ht in zip(dct_components, component_huffman_tables)
        ]
    ):
        for mcu_bits in mcu_bits_list:
            bit_writer.write_bits(*mcu_bits)

    bit_writer.finalize()

    return bytes(bit_writer.output)


def _bit_reader(ecs_segment: bytes) -> Iterator[int]:
    # destuffing
    clean_segment = ecs_segment.replace(b"\xff\x00", b"\xff")
    for byte in clean_segment:
        for i in range(7, -1, -1):  # 7 to 0
            yield (byte >> i) & 1


def _read_jpeg_mcu_data(
    ecs_segment: bytes,
    mcu_num_hw: tuple[int, int],
    mcu_size_hw_list: list[tuple[int, int]],
    component_lookup_tables: list[dict[str, LookupTable]],
) -> list[list[np.ndarray]]:
    gen = _bit_reader(ecs_segment)

    def decode_symbol(target_table: LookupTable) -> int:
        code = 0
        code_len = 0
        while True:
            bit = next(gen)
            code = (code << 1) | bit
            code_len += 1

            if code_len in target_table.max_codes and code <= target_table.max_codes[code_len]:
                return target_table.get_symbol(code, code_len)

    def read_val(cat: int) -> int:
        enc_val = 0
        for _ in range(cat):
            enc_val = (enc_val << 1) | next(gen)

        return decode_encval(enc_val, cat)

    def decode_block(
        dc_base: int, dc_lookup_table: LookupTable, ac_lookup_table: LookupTable
    ) -> tuple[np.ndarray, int]:
        block_buffer = np.zeros(64)

        cat = decode_symbol(dc_lookup_table)
        dc_diff = read_val(cat)
        dc_val = dc_base + dc_diff
        block_buffer[0] = dc_val

        block_ptr = 1
        while block_ptr < 64:
            symbol = decode_symbol(ac_lookup_table)
            runlen = symbol >> 4
            cat = symbol & 0x0F

            if runlen == 0 and cat == 0:
                break

            if runlen == 15 and cat == 0:
                block_ptr += 16
            else:
                val = read_val(cat)
                block_buffer[block_ptr : block_ptr + runlen] = 0
                block_buffer[block_ptr + runlen] = val
                block_ptr += runlen + 1

        return zigzag_scan_inv(block_buffer), dc_val

    # [(mcu_size_h, mcu_size_w, 8, 8),...]のテンソルのリストを作る
    n_components = len(component_lookup_tables)
    component_mcu_list = [[] for i in range(n_components)]
    dc_bases = [0] * n_components
    for _ in range(mcu_num_hw[0] * mcu_num_hw[1]):
        for i, mcu_size_hw in enumerate(mcu_size_hw_list):
            h_size, w_size = mcu_size_hw
            mcu_blocks = []
            for _ in range(h_size * w_size):
                try:
                    block_data, dc_bases[i] = decode_block(
                        dc_bases[i],
                        component_lookup_tables[i]["dc_lookup_table"],
                        component_lookup_tables[i]["ac_lookup_table"],
                    )
                except StopIteration:
                    block_data = np.zeros((8, 8))
                    dc_bases[i] = 0

                mcu_blocks.append(block_data)

            component_mcu_list[i].append(np.stack(mcu_blocks).reshape(h_size, w_size, 8, 8))

    return component_mcu_list


def from_entropy_coded_segment(
    ecs_segment: bytes,
    start_of_frame: StartOfFrame,
    start_of_scan: StartOfScan,
    component_huffman_tables: dict[tuple[int, int], HuffmanTable],
) -> list[QuantizedBlocks]:
    mcu_size_hw_list = start_of_frame.get_mcu_size_hw_list()
    component_lookup_tables = [
        {
            "dc_lookup_table": component_huffman_tables[
                (0, ht_id.dc_huffman_table_id)
            ].get_lookup_table(),
            "ac_lookup_table": component_huffman_tables[
                (1, ht_id.ac_huffman_table_id)
            ].get_lookup_table(),
        }
        for ht_id in start_of_scan.huffman_table_ids
    ]
    mcu_num_hw = start_of_frame.get_mcu_num_hw()
    component_mcu_list = _read_jpeg_mcu_data(
        ecs_segment, mcu_num_hw, mcu_size_hw_list, component_lookup_tables
    )

    return [QuantizedBlocks.create(comp_blocks, mcu_num_hw) for comp_blocks in component_mcu_list]
