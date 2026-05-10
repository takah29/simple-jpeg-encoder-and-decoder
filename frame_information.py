import numpy as np
from typing import Self


class FrameInformation:
    class Component:
        component_id: int
        sampling_factor: tuple[int, int]
        quantization_table_id: int

    sample_precision: int
    image_height: int
    image_width: int
    num_components: int
    components: list[Component]

    @classmethod
    def create(cls, image: np.ndarray, sample_precision: int = 8) -> Self:
        sample_precision = 8
        image_height, image_width, num_components = image.shape

        if num_components not in (1, 3):
            msg = f"Number of components must be 1 or 3, but got {num_components}"
            raise ValueError(msg)

        components = [
            cls.Component(i, (2, 2) if i != 0 else (1, 1), 1 if i == 0 else 2)
            for i in range(num_components)
        ]
        return cls(
            sample_precision, image_height, image_width, num_components, components
        )

    def to_bytes(self) -> bytes:
        pass
