from pathlib import Path

import numpy as np

from jpg.dct_component import DctComponent
from jpg.frame_information import FrameInformation
from jpg.helper import (
    END_OF_IMAGE,
    START_OF_IMAGE,
    JpegBitWriter,
    to_ycbcr,
)
from jpg.huffman_table import HuffmanTable
from jpg.quantization_table import QuantizationTable
from jpg.start_of_scan import StartOfScan


def _encode_dct_blocks(
    dct_component_list: list[DctComponent],
) -> bytes:
    bit_writer = JpegBitWriter()

    for mcu_bits_list in zip(*[x.mcu_encoder() for x in dct_component_list]):
        for mcu_bits in mcu_bits_list:
            bit_writer.write_bits(*mcu_bits)

    bit_writer.finalize()

    return bytes(bit_writer.output)


def jpg_encode(
    img: np.ndarray,
    mcu_size_hw_list: list[tuple[int, int]],
    sample_step_hw_list: list[tuple[int, int]],
    quality: int,
) -> bytes:
    frame_information = FrameInformation.create(img.shape, mcu_size_hw_list)
    start_of_scan = StartOfScan.create(img.shape)

    q_table_y = QuantizationTable.create(0, 0, True, quality)
    q_table_c = QuantizationTable.create(0, 1, False, quality)

    ydc_ht = HuffmanTable.from_file(Path("./huffman_code/ydc_hc.csv"), 0, 0)
    yac_ht = HuffmanTable.from_file(Path("./huffman_code/yac_hc.csv"), 1, 0)
    uvdc_ht = HuffmanTable.from_file(Path("./huffman_code/uvdc_hc.csv"), 0, 1)
    uvac_ht = HuffmanTable.from_file(Path("./huffman_code/uvac_hc.csv"), 1, 1)

    img = img.astype(np.float64)
    if img.ndim == 3:
        # RGB
        img_ycbcr = to_ycbcr(img) + np.array([-128, 0, 0])
        components = [
            DctComponent.create(
                img_ycbcr[:, :, 0],
                q_table_y,
                ydc_ht,
                yac_ht,
                mcu_size_hw_list[0],
                sample_step_hw_list[0],
            ),
            DctComponent.create(
                img_ycbcr[:, :, 1],
                q_table_c,
                uvdc_ht,
                uvac_ht,
                mcu_size_hw_list[1],
                sample_step_hw_list[1],
            ),
            DctComponent.create(
                img_ycbcr[:, :, 2],
                q_table_c,
                uvdc_ht,
                uvac_ht,
                mcu_size_hw_list[2],
                sample_step_hw_list[2],
            ),
        ]

        headers = [
            START_OF_IMAGE.to_bytes(2, "big"),
            q_table_y.to_bytes(),
            q_table_c.to_bytes(),
            frame_information.to_bytes(),
            ydc_ht.to_bytes(),
            yac_ht.to_bytes(),
            uvdc_ht.to_bytes(),
            uvac_ht.to_bytes(),
            start_of_scan.to_bytes(),
        ]
    else:
        # Grayscale
        img_gray = img - 128.0
        components = [
            DctComponent.create(
                img_gray, q_table_y, ydc_ht, yac_ht, mcu_size_hw_list[0], sample_step_hw_list[0]
            )
        ]

        headers = [
            START_OF_IMAGE.to_bytes(2, "big"),
            q_table_y.to_bytes(),
            frame_information.to_bytes(),
            ydc_ht.to_bytes(),
            yac_ht.to_bytes(),
            start_of_scan.to_bytes(),
        ]

    encoded_dct_blocks = _encode_dct_blocks(components)

    return b"".join(headers + [encoded_dct_blocks, END_OF_IMAGE.to_bytes(2, "big")])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
