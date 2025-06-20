import json
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from PIL import Image

from src.classifier import ClassifierController
from src.dataset import DatasetController
from src.payloads import DatasetCheckPayload

DATASET_CHECK_JSON_PATH = "DATASET_CHECK_JSON_PATH"


if __name__ == "__main__":
    json_path = os.environ.get(key=DATASET_CHECK_JSON_PATH)
    with open(file=json_path, mode="r") as fp:
        json_dict = json.load(fp=fp)
    payload = DatasetCheckPayload(**json_dict)
    pprint(payload)

    classifier_controller = ClassifierController(
        model_dir_path=Path(payload.model_dir_path),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        eval_only_flag=True,
    )
    dataset_controller = DatasetController(
        data_root_dir=Path(payload.data_root_dir),
        image_processor=classifier_controller.get_processor(),
    )

    dataset = dataset_controller.get_dataset()
    pprint(dataset)

    for sample_idx in range(payload.train_samples_to_save):
        sample = dataset["train"][sample_idx]
        image = sample["pixel_values"]
        image = image.permute(dims=(1, 2, 0)).cpu().numpy()
        image = np.uint8(255 * image)
        image = Image.fromarray(obj=image)
        save_path = Path(payload.transformed_output_dir, f"{sample_idx}.jpg")
        image.save(fp=save_path)
        print(f"Saved sample in {save_path}")
