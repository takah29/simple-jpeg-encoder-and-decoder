import numpy as np

from jpg.core.entropy_coded_segment import from_entropy_coded_segment
from jpg.core.helper import END_OF_IMAGE, START_OF_IMAGE, to_rgb
from jpg.core.huffman_table import HuffmanTable
from jpg.core.quantization_table import QuantizationTable
from jpg.core.start_of_frame import StartOfFrame
from jpg.core.start_of_scan import StartOfScan


def _get_segment(data: bytes, current_idx: int) -> bytes:
    marker = data[current_idx : current_idx + 2]
    segment_length = data[current_idx + 2 : current_idx + 4]
    data_length = int.from_bytes(segment_length, "big") - 2
    data = data[current_idx + 4 : current_idx + 4 + data_length]

    return marker + segment_length + data


def jpg_decode(jpg_bytes: bytes) -> np.ndarray:
    if START_OF_IMAGE.to_bytes(2, "big") not in jpg_bytes:
        raise ValueError("Invalid JPEG file. Start of image not found.")

    current_idx = jpg_bytes.find(START_OF_IMAGE.to_bytes(2, "big")) + 2

    quantization_table_map = {}
    start_of_frame = None
    huffman_table_map = {}
    start_of_scan = None
    while True:
        current_marker = int.from_bytes(jpg_bytes[current_idx : current_idx + 2])

        match current_marker:
            case QuantizationTable.MARKER:
                segment = _get_segment(jpg_bytes, current_idx)
                q_tables = QuantizationTable.from_bytes(segment)
                for qt in q_tables:
                    quantization_table_map[qt.table_id] = qt
                current_idx += len(segment)

            case marker if marker in (StartOfFrame.MARKER, StartOfFrame.MARKER + 1):  # SOF0 or SOF1
                segment = _get_segment(jpg_bytes, current_idx)
                start_of_frame = StartOfFrame.from_bytes(segment)
                current_idx += len(segment)

            case marker if 0xFFC2 <= marker <= 0xFFCF and marker not in (
                HuffmanTable.MARKER,
                0xFFCC,
            ):
                raise ValueError(
                    f"Unsupported SOF marker: 0x{marker:02X} (Progressive or other modes not supported)"
                )

            case HuffmanTable.MARKER:
                segment = _get_segment(jpg_bytes, current_idx)
                huffman_tables = HuffmanTable.from_bytes(segment)
                for ht in huffman_tables:
                    huffman_table_map[(ht.table_class, ht.table_id)] = ht
                current_idx += len(segment)

            case StartOfScan.MARKER:
                segment = _get_segment(jpg_bytes, current_idx)
                start_of_scan = StartOfScan.from_bytes(segment)
                current_idx += len(segment)
                break

            case _:
                segment = _get_segment(jpg_bytes, current_idx)
                current_idx += len(segment)

    if start_of_frame is None:
        msg = "Invalid JPEG file. Start of frame not found."
        raise ValueError(msg)

    # decoding
    end_of_image_idx = jpg_bytes.find(END_OF_IMAGE.to_bytes(2, "big"))
    components = from_entropy_coded_segment(
        jpg_bytes[current_idx:end_of_image_idx],
        start_of_frame,
        start_of_scan,
        huffman_table_map,
    )
    img_shape = (start_of_frame.image_height, start_of_frame.image_width)
    img_comp_list = [
        component.to_image_component(
            img_shape,
            sample_step_hw,
            quantization_table_map[si.quantization_table_id],
        )
        for component, sample_step_hw, si in zip(
            components,
            start_of_frame.get_sample_step_hw_list(),
            start_of_frame.sampling_info_list,
        )
    ]

    if len(img_comp_list) == 1:
        return img_comp_list[0] + 128
    elif len(img_comp_list) == 3:
        img = np.stack(img_comp_list, axis=2)
        img[:, :, 0] += 128
        return np.clip(to_rgb(img), 0, 255).astype(np.uint8)
    else:
        raise ValueError("Invalid number of components.")
