from dataclasses import dataclass
from typing import Self


@dataclass
class SamplingInfo:
    component_id: int
    sampling_width: int
    sampling_height: int
    quantization_table_id: int

    def to_bytes(self) -> bytes:
        return (
            self.component_id.to_bytes()
            + ((self.sampling_width << 4) | self.sampling_height).to_bytes()
            + self.quantization_table_id.to_bytes()
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) != 3:
            raise ValueError("data must be 3 bytes long")

        component_id = data[0]
        sampling_width = (data[1] >> 4) & 0x0F
        sampling_height = data[1] & 0x0F
        quantization_table_id = data[2]

        return cls(component_id, sampling_width, sampling_height, quantization_table_id)


@dataclass
class FrameInformation:
    MARKER = 0xFFC0

    sample_precision: int
    image_height: int
    image_width: int
    num_components: int
    sampling_info_list: list[SamplingInfo]

    @classmethod
    def create(cls, image_shape: tuple[int, ...], sample_precision: int = 8) -> Self:
        sample_precision = 8

        if len(image_shape) == 3:
            image_height, image_width, num_components = image_shape
        elif len(image_shape) == 2:
            image_height, image_width = image_shape
            num_components = 1
        else:
            msg = f"Invalid image shape: {image_shape}. Expected GrayScale or RGB image."
            raise ValueError(msg)

        if num_components == 1:
            sampling_info_list = [SamplingInfo(1, 1, 1, 0)]
        elif num_components == 3:
            sampling_info_list = [
                SamplingInfo(1, 2, 2, 0),
                SamplingInfo(2, 1, 1, 1),
                SamplingInfo(3, 1, 1, 1),
            ]
        else:
            msg = f"Invalid image shape: {image_shape}. Expected GrayScale or RGB image."
            raise ValueError(msg)

        return cls(sample_precision, image_height, image_width, num_components, sampling_info_list)

    def to_bytes(self) -> bytes:
        marker_bytes = FrameInformation.MARKER.to_bytes(2, "big")
        segment_length = (8 + self.num_components * 3).to_bytes(2, "big")

        info = (
            self.sample_precision.to_bytes()
            + self.image_height.to_bytes(2, "big")
            + self.image_width.to_bytes(2, "big")
            + self.num_components.to_bytes()
        )

        component_bytes = b"".join(component.to_bytes() for component in self.sampling_info_list)

        return marker_bytes + segment_length + info + component_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) < 8:
            raise ValueError("data is too short")

        marker = int.from_bytes(data[0:2], "big")
        if marker != cls.MARKER:
            raise ValueError("invalid marker")

        segment_length = int.from_bytes(data[2:4], "big")
        if len(data) != segment_length + 2:
            raise ValueError("invalid segment length")

        sample_precision = data[4]
        image_height = int.from_bytes(data[5:7], "big")
        image_width = int.from_bytes(data[7:9], "big")
        num_components = data[9]

        sampling_info_list = []
        for i in range(num_components):
            start_idx = 10 + i * 3
            end_idx = start_idx + 3
            sampling_info_bytes = data[start_idx:end_idx]
            sampling_info = SamplingInfo.from_bytes(sampling_info_bytes)
            sampling_info_list.append(sampling_info)

        return cls(sample_precision, image_height, image_width, num_components, sampling_info_list)


if __name__ == "__main__":
    img_shape = (256, 512, 3)
    frame_info = FrameInformation.create(img_shape)
    print(frame_info)

    frame_info_bytes = frame_info.to_bytes()
    print(frame_info_bytes.hex(" "))

    parsed_frame_info = FrameInformation.from_bytes(frame_info_bytes)
    print(parsed_frame_info)

    assert frame_info == parsed_frame_info
