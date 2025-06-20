from pathlib import Path

from datasets import DatasetDict
from transformers import ViTImageProcessor

from .dataset import ClassificationDataset


class DatasetController:
    """Image classification dataset controller"""

    def __init__(
        self, data_root_dir: Path, image_processor: ViTImageProcessor
    ) -> None:
        """Initialize the classification dataset

        Args:
            data_root_dir (Path): Root dir of dataset
            image_processor (ViTImageProcessor): ViT processor to be used
        """
        self.classification_dataset = ClassificationDataset(
            data_root_dir=data_root_dir, image_processor=image_processor
        )

    def get_dataset(self) -> DatasetDict:
        """Return the internal dataset

        Returns:
            DatasetDict: Internal dataset
        """
        return self.classification_dataset.get_dataset()
