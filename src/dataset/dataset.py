from pathlib import Path
from typing import Dict

import torch
from datasets import DatasetDict, load_dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
from transformers import ViTImageProcessor

AUGMENTATION_RANGE_BRIGHTNESS = (0.80, 1.20)
AUGMENTATION_RANGE_CONTRAST = (0.80, 1.20)
AUGMENTATION_RANGE_SATURATION = (0.80, 1.20)
AUGMENTATION_RANGE_HUE = (-0.05, 0.05)
AUGMENTATION_FLIP_PROBABILITY = 0.50


class ClassificationDataset:
    """Image classification dataset handler"""

    def __init__(
        self, data_root_dir: Path, image_processor: ViTImageProcessor
    ) -> None:
        """Construct 'imagefolder' classification dataset

        Args:
            data_root_dir (Path): Root dir of dataset
            image_processor (ViTImageProcessor): ViT processor to be used
        """
        self.dataset = load_dataset(
            path="imagefolder",
            data_dir=data_root_dir,
        )
        self.image_processor = image_processor

        self.label2id = dict()
        self.id2label = dict()
        self._map_id_labels()

        self.train_transform = None
        self.validation_transforms = None
        self._create_transforms()
        self._apply_transforms()

    def _map_id_labels(self) -> None:
        """Map class ids to labels and vice-versa"""
        labels = self.dataset["train"].features["label"].names
        for i, label in enumerate(labels):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label

    def _create_transforms(self) -> None:
        """Create image transformations to be used in different phases"""
        color_jitter = ColorJitter(
            brightness=AUGMENTATION_RANGE_BRIGHTNESS,
            contrast=AUGMENTATION_RANGE_CONTRAST,
            saturation=AUGMENTATION_RANGE_SATURATION,
            hue=AUGMENTATION_RANGE_HUE,
        )

        self.training_transforms = Compose(
            transforms=[
                RandomHorizontalFlip(p=AUGMENTATION_FLIP_PROBABILITY),
                RandomVerticalFlip(p=AUGMENTATION_FLIP_PROBABILITY),
                color_jitter,
                ToTensor(),
            ]
        )
        self.validation_transforms = Compose(transforms=[ToTensor()])

    def _apply_training_transforms(
        self, sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Callback function to apply training transforms

        Args:
            sample (Dict[str, torch.Tensor]): Dataset sample to apply transforms

        Returns:
            Dict[str, torch.Tensor]: Dataset sample transformed
        """
        sample["pixel_values"] = [
            self.training_transforms(img=img.convert("RGB"))
            for img in sample["image"]
        ]
        del sample["image"]
        return sample

    def _apply_validation_transforms(
        self, sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Callback function to apply validation/test transforms

        Args:
            sample (Dict[str, torch.Tensor]): Dataset sample to apply transforms

        Returns:
            Dict[str, torch.Tensor]: Dataset sample transformed
        """
        sample["pixel_values"] = [
            self.validation_transforms(img=img.convert("RGB"))
            for img in sample["image"]
        ]
        del sample["image"]
        return sample

    def _apply_transforms(self) -> None:
        """Apply transforms to different phases/subsets"""
        self.dataset["train"] = self.dataset["train"].with_transform(
            transform=self._apply_training_transforms
        )
        self.dataset["validation"] = self.dataset["validation"].with_transform(
            transform=self._apply_validation_transforms
        )
        self.dataset["test"] = self.dataset["test"].with_transform(
            transform=self._apply_validation_transforms
        )

    def get_dataset(self) -> DatasetDict:
        """Return the internal dataset

        Returns:
            DatasetDict: Internal dataset
        """
        return self.dataset
