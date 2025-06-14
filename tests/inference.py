import cv2
import json
import natsort
import os
import torch
from pathlib import Path
from pprint import pprint

from src.classifier import ClassifierController
from src.payloads import InferencePayload


ENV_VAR_INFERENCE_JSON_PATH = "INFERENCE_JSON_PATH"


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
    batch_probabilities = classifier_controller.get_probabilities(
        input_batch=input_batch
    )
    for idx, image_probabilities in enumerate(batch_probabilities):
        most_probable = image_probabilities[0]
        print(f"Image {idx} - {most_probable.label} - {most_probable.score}")
