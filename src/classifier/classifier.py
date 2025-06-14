import numpy as np
import torch
import transformers.modeling_outputs
from pathlib import Path
from transformers import AutoModelForImageClassification, ViTImageProcessor
from typing import List

from .classifier_interface import ClassificationPipelineOutputInterface


class Classifier:
    """Image Classification model"""

    def __init__(
        self,
        model_dir_path: Path,
        device: torch.device,
        eval_only_flag: bool
    ) -> None:
        """Construct and initiate model and processor

        Args:
            model_dir_path (Path): Dir path to compatible model (ViT processor based)
            device (torch.device): Device to load the model
            eval_only_flag (bool): If 'True', disable gradient calculus
        """
        self.device = device
        self.eval_only_flag = eval_only_flag
        self.processor = ViTImageProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir_path,
            use_fast=True
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_dir_path
        )
        self.model.to(device)
        if self.eval_only_flag:
            self._set_model_to_eval()

    def _set_model_to_eval(self) -> None:
        """Disable gradient calculus globally"""
        self.model.eval()
        torch.set_grad_enabled(mode=False)

    def infer(
        self,
        input_batch: List[np.ndarray]
    ) -> transformers.modeling_outputs.ImageClassifierOutput:
        """Processor and model forward pass

        Args:
            input_batch (List[np.ndarray]): List of images to be classified
                (RGB, (h, w, c))

        Returns:
            transformers.modeling_outputs.ImageClassifierOutput:
                Classifier model output (raw)
        """
        inputs = self.processor(images=input_batch, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs

    def get_probabilities(
        self,
        inference_output: transformers.modeling_outputs.ImageClassifierOutput
    ) -> List[List[ClassificationPipelineOutputInterface]]:
        """Return formatted probabilities of the classification model

        Args:
            inference_output (transformers.modeling_outputs.ImageClassifierOutput):
                Raw inference output of the model

        Returns:
            List[List[ClassificationPipelineOutputInterface]]: List containing
                a list of labels and respective scores inferred (sorted from
                most to least probable)
        """
        output_logits = inference_output.logits
        batch_probabilities = torch.softmax(input=output_logits, dim=-1)
        batch_probabilities = batch_probabilities.cpu().numpy()
        batch_labels_scores_list = list()

        for probabilities in batch_probabilities:
            sorted_idx_list = np.argsort(a=-probabilities)
            image_labels_scores_list = list()
            for class_idx in sorted_idx_list:
                output_interface = ClassificationPipelineOutputInterface(
                    label=self.model.config.id2label[class_idx],
                    score=float(probabilities[class_idx])
                )
                image_labels_scores_list.append(output_interface)
            batch_labels_scores_list.append(image_labels_scores_list)

        return batch_labels_scores_list
