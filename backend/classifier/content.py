
import io
import numpy as np

from common.logger import *
from PIL import Image
import uuid

target_size = (16, 16)


class ImageContent:
    _img_array: np.ndarray
    _id: str

    def __init__(self, image_bytes):
        self._id = str(uuid.uuid4())
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.resize(target_size)
            img_array = np.array(img)

            # Ensure the image has 3 channels (RGB)
            if len(img_array.shape) == 2:  # grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]  # Remove alpha channel

            # Normalize and expand dimensions for batch
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            self._img_array = img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")

    def get_id(self):
        return self._id

    def get_data(self) -> np.ndarray:
        return self._img_array

    def get_witness_input(self):
        flattened = self._img_array.flatten().tolist()

        # Create the input JSON structure required by EZKL
        input_data = {
            "input_data": [flattened]
        }
        return input_data


