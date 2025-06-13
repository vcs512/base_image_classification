import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForImageClassification, ViTImageProcessor
from typing import List


class Classifier:
    """Image Classification model"""

    def __init__(
        self,
        model_dir_path: Path,
        device: torch.device,
        eval_only_flag: bool
    ) -> None:
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
        self.model.eval()
        torch.set_grad_enabled(mode=False)

    def infer(self, input_batch: List[np.ndarray]) -> any:
        inputs = self.processor(images=input_batch, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs

    def get_probabilities(self, inference_output: any) -> any:
        output_logits = inference_output.logits
        batch_probabilities = torch.softmax(input=output_logits, dim=-1)
        batch_probabilities = batch_probabilities.cpu().numpy()
        class_probabilities_dict_list = list()

        for probabilities in batch_probabilities:
            sorted_idx_list = np.argsort(a=-probabilities)
            class_probabilities_dict = dict()
            for class_idx in sorted_idx_list:
                class_name = self.model.config.id2label[class_idx]
                class_probabilities_dict[class_name] = float(probabilities[class_idx])
                break
            class_probabilities_dict_list.append(class_probabilities_dict)

        return class_probabilities_dict_list
