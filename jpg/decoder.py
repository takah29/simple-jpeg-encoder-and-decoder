import numpy as np

from jpg.entropy_coded_segment import from_entropy_coded_segment
from jpg.frame_information import FrameInformation
from jpg.helper import END_OF_IMAGE, START_OF_IMAGE, to_rgb
from jpg.huffman_table import HuffmanTable, LookupTable
from jpg.quantization_table import QuantizationTable
from jpg.start_of_scan import StartOfScan


def _get_segment(data: bytes, current_idx: int) -> bytes:
    marker = data[current_idx : current_idx + 2]
    segment_length = data[current_idx + 2 : current_idx + 4]
    data_length = int.from_bytes(segment_length, "big") - 2
    data = data[current_idx + 4 : current_idx + 4 + data_length]

    return marker + segment_length + data


def _build_lookup_tables(
    start_of_scan: StartOfScan, huffman_tables: dict[tuple[int, int], HuffmanTable]
) -> dict[tuple[int, int], LookupTable]:
    lookup_tables = {}
    for i in range(1, start_of_scan.num_components + 1):
        dc_huffman_table = huffman_tables[
            (0, start_of_scan.huffman_table_ids[i].dc_huffman_table_id)
        ]
        ac_huffman_table = huffman_tables[
            (1, start_of_scan.huffman_table_ids[i].ac_huffman_table_id)
        ]

        lookup_tables[(i, 0)] = dc_huffman_table.get_lookup_table()
        lookup_tables[(i, 1)] = ac_huffman_table.get_lookup_table()

    return lookup_tables


def jpg_decode(jpg_bytes: bytes) -> np.ndarray:
    if START_OF_IMAGE.to_bytes(2, "big") not in jpg_bytes:
        raise ValueError("Invalid JPEG file. Start of image not found.")

    current_idx = jpg_bytes.find(START_OF_IMAGE.to_bytes(2, "big")) + 2

    quantization_tables = {}
    frame_information = None
    huffman_tables = {}
    start_of_scan = None
    while True:
        if QuantizationTable.MARKER.to_bytes(2, "big") == jpg_bytes[current_idx : current_idx + 2]:
            segment = _get_segment(jpg_bytes, current_idx)
            quantization_table = QuantizationTable.from_bytes(segment)
            quantization_tables[quantization_table.table_id] = quantization_table
            current_idx += len(segment)
        elif FrameInformation.MARKER.to_bytes(2, "big") == jpg_bytes[current_idx : current_idx + 2]:
            segment = _get_segment(jpg_bytes, current_idx)
            frame_information = FrameInformation.from_bytes(segment)
            current_idx += len(segment)
        elif HuffmanTable.MARKER.to_bytes(2, "big") == jpg_bytes[current_idx : current_idx + 2]:
            segment = _get_segment(jpg_bytes, current_idx)
            huffman_table = HuffmanTable.from_bytes(segment)
            huffman_tables[(huffman_table.table_class, huffman_table.table_id)] = huffman_table
            current_idx += len(segment)
        elif StartOfScan.MARKER.to_bytes(2, "big") == jpg_bytes[current_idx : current_idx + 2]:
            segment = _get_segment(jpg_bytes, current_idx)
            start_of_scan = StartOfScan.from_bytes(segment)
            current_idx += len(segment)
            break
        else:
            raise ValueError("Invalid JPEG file. Invalid marker.")

    if frame_information is None:
        msg = "Invalid JPEG file. Frame information not found."
        raise ValueError(msg)

    # print("Start of Scan: ", start_of_scan)
    print("Frame Information: ", frame_information)
    # print("Quantization Table: ", quantization_tables)
    # print("Huffman Table: ", huffman_tables)

    # decoding
    end_of_image_idx = jpg_bytes.find(END_OF_IMAGE.to_bytes(2, "big"))
    components = from_entropy_coded_segment(
        jpg_bytes[current_idx:end_of_image_idx],
        frame_information,  # pyrefly: ignore [bad-argument-type]
        start_of_scan,
        huffman_tables,  # pyrefly: ignore [bad-argument-type]
    )
    img_shape = (frame_information.image_height, frame_information.image_width)
    image_list = [
        component.to_image_component(
            img_shape,
            sample_step_hw,
            quantization_tables[si.quantization_table_id],
        )
        for component, sample_step_hw, si in zip(
            components,
            frame_information.get_sample_step_hw_list(),
            frame_information.sampling_info_list,
        )
    ]

    if len(image_list) == 1:
        return image_list[0] + 128
    elif len(image_list) == 3:
        img = np.stack(image_list, axis=2)
        img[:, :, 0] += 128
        return np.clip(to_rgb(img), 0, 255).astype(np.uint8)
    else:
        raise ValueError("Invalid JPEG file. Invalid number of components.")
