import cv2
import json
import natsort
import numpy as np
import os
import torch
from pathlib import Path
from pprint import pprint
from typing import List

from src.classifier.classifier import Classifier
from src.payloads import InferencePayload


ENV_VAR_INFERENCE_JSON_PATH = "INFERENCE_JSON_PATH"
BATCH_SIZE = 64


class ClassifierController:
    """Classification model controller"""

    def __init__(
        self,
        model_dir_path: Path,
        device: torch.device,
        eval_only_flag: bool
    ) -> None:
        self.classifier = Classifier(
            model_dir_path=model_dir_path,
            device=device,
            eval_only_flag=eval_only_flag
        )

    def get_probabilities(self, input_batch: List[np.ndarray]) -> any:
        number_mini_batches = (
            len(input_batch) // BATCH_SIZE + len(input_batch) % BATCH_SIZE
        )
        mini_batches = np.array_split(
            ary=input_batch,
            indices_or_sections=number_mini_batches
        )
        probabilities_list = list()
        for mini_batch in mini_batches:
            inference_output = self.classifier.infer(input_batch=mini_batch)
            probabilities_list.extend(
                self.classifier.get_probabilities(
                    inference_output=inference_output
                )
            )
        return probabilities_list


if __name__ == "__main__":
    inference_json_path = os.environ.get(key=ENV_VAR_INFERENCE_JSON_PATH)
    with open(file=inference_json_path, mode="r") as fp:
        inference_json_dict = json.load(fp=fp)
    inference_payload = InferencePayload(**inference_json_dict)
    pprint(inference_payload)

    classifier_controller = ClassifierController(
        model_dir_path=Path(inference_payload.model_dir_path),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        eval_only_flag=True
    )

    images_path_list = natsort.natsorted(
        seq=Path(inference_payload.input_images_dir).iterdir()
    )
    input_batch = [
        cv2.imread(filename=filename, flags=cv2.IMREAD_COLOR)
        for filename in images_path_list
    ]
    probabilities = classifier_controller.get_probabilities(
        input_batch=input_batch
    )
    pprint(probabilities)
