from pydantic import BaseModel


class InferencePayload(BaseModel):
    model_dir_path: str
    input_images_dir: str
