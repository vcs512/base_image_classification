from pathlib import Path
from typing import List

import numpy as np
import torch

from .classifier import Classifier
from .classifier_interface import ClassificationPipelineOutputInterface

BATCH_SIZE = 64


class ClassifierController:
    """Classification model controller"""

    def __init__(
        self, model_dir_path: Path, device: torch.device, eval_only_flag: bool
    ) -> None:
        """Construct and initialize classification model

        Args:
            model_dir_path (Path): Dir path to a compatible model (ViT processor)
            device (torch.device): Device to use in the inference
            eval_only_flag (bool): If 'True', disable gradient calculus
        """
        self.classifier = Classifier(
            model_dir_path=model_dir_path,
            device=device,
            eval_only_flag=eval_only_flag,
        )

    def get_probabilities(
        self, input_batch: List[np.ndarray]
    ) -> List[List[ClassificationPipelineOutputInterface]]:
        """Classify the input batch of the model.
            Divides the input in mini batches to avoid memory overflow

        Args:
            input_batch (List[np.ndarray]): List of images (RGB, (h, w, c))

        Returns:
            List[ClassificationPipelineOutputInterface]:
                List of labels and scores inferred for each image in input
        """
        probabilities_list = list()
        for start_idx in range(0, len(input_batch), BATCH_SIZE):
            end_idx = start_idx + BATCH_SIZE
            if end_idx > len(input_batch):
                end_idx = len(input_batch)
            mini_batch = input_batch[start_idx:end_idx]
            inference_output = self.classifier.infer(input_batch=mini_batch)
            probabilities_list.extend(
                self.classifier.get_probabilities(
                    inference_output=inference_output
                )
            )
        return probabilities_list
