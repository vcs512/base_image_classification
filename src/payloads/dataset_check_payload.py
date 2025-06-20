from pydantic import BaseModel


class DatasetCheckPayload(BaseModel):
    """Expected payload for dataset check"""

    model_dir_path: str
    data_root_dir: str
    transformed_output_dir: str
    train_samples_to_save: int
