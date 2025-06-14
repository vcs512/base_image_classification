from pydantic import BaseModel


class InferencePayload(BaseModel):
    """Expected payload for inference"""

    model_dir_path: str
    input_images_dir: str
