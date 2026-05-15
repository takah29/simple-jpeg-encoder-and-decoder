from pathlib import Path

import numpy as np

from jpg.core.entropy_coded_segment import to_entropy_coded_segment
from jpg.core.helper import END_OF_IMAGE, START_OF_IMAGE, to_ycbcr
from jpg.core.huffman_table import HuffmanTable
from jpg.core.quantization_table import QuantizationTable
from jpg.core.quantized_blocks import QuantizedBlocks
from jpg.core.start_of_frame import StartOfFrame
from jpg.core.start_of_scan import StartOfScan


def jpg_encode(
    img: np.ndarray,
    subsampling_type: str,
    quality: int,
) -> bytes:
    start_of_frame = StartOfFrame.create(img.shape, subsampling_type)
    start_of_scan = StartOfScan.create(img.shape)

    q_table_y = QuantizationTable.create(0, 0, True, quality)
    q_table_c = QuantizationTable.create(0, 1, False, quality)

    huffman_code_dir = Path(__file__).parents[1] / "huffman_code"
    ydc_ht = HuffmanTable.from_file(huffman_code_dir / "ydc_hc.csv", 0, 0)
    yac_ht = HuffmanTable.from_file(huffman_code_dir / "yac_hc.csv", 1, 0)
    uvdc_ht = HuffmanTable.from_file(huffman_code_dir / "uvdc_hc.csv", 0, 1)
    uvac_ht = HuffmanTable.from_file(huffman_code_dir / "uvac_hc.csv", 1, 1)

    img = img.astype(np.float64)
    mcu_size_hw_list = start_of_frame.get_mcu_size_hw_list()
    sample_step_hw_list = start_of_frame.get_sample_step_hw_list()
    if img.ndim == 3:
        # RGB
        img_ycbcr = to_ycbcr(img) - np.array([128, 0, 0])

        q_tables = [q_table_y, q_table_c, q_table_c]
        dc_huffman_tables = [ydc_ht, uvdc_ht, uvdc_ht]
        ac_huffman_tables = [yac_ht, uvac_ht, uvac_ht]

        components = []
        component_huffman_tables = []
        for img_comp, mcu_size_hw, sample_step_hw, q_table, dc_ht, ac_ht in zip(
            img_ycbcr.transpose(2, 0, 1),
            mcu_size_hw_list,
            sample_step_hw_list,
            q_tables,
            dc_huffman_tables,
            ac_huffman_tables,
        ):
            components.append(
                QuantizedBlocks.from_image_component(img_comp, q_table, mcu_size_hw, sample_step_hw)
            )
            component_huffman_tables.append(
                {
                    "dc_huffman_table": dc_ht,
                    "ac_huffman_table": ac_ht,
                }
            )

        headers = [
            START_OF_IMAGE.to_bytes(2, "big"),
            q_table_y.to_bytes(),
            q_table_c.to_bytes(),
            start_of_frame.to_bytes(),
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
            QuantizedBlocks.from_image_component(
                img_gray, q_table_y, mcu_size_hw_list[0], sample_step_hw_list[0]
            )
        ]
        component_huffman_tables = [
            {
                "dc_huffman_table": ydc_ht,
                "ac_huffman_table": yac_ht,
            }
        ]

        headers = [
            START_OF_IMAGE.to_bytes(2, "big"),
            q_table_y.to_bytes(),
            start_of_frame.to_bytes(),
            ydc_ht.to_bytes(),
            yac_ht.to_bytes(),
            start_of_scan.to_bytes(),
        ]

    entropy_coded_segment = to_entropy_coded_segment(components, component_huffman_tables)

    return b"".join(headers + [entropy_coded_segment, END_OF_IMAGE.to_bytes(2, "big")])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
