import cv2
import torch

from .classifier import Classifier

if __name__ == "__main__":
    MODEL_PATH = "./models/swinv2-small-patch4-window8-256"
    # MODEL_PATH = "./models/vit-small-patch16-224"
    # MODEL_PATH = "./models/deit-small-patch16-224"

    print(f"MODEL_PATH = {MODEL_PATH}")
    classifier = Classifier(
        model_dir_path=MODEL_PATH,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        eval_only_flag=True
    )

    image_path = "./data/sports100/train/air hockey/001.jpg"
    image = cv2.imread(filename=image_path, flags=cv2.IMREAD_COLOR)
    image2_path = "./data/sports100/train/volleyball/001.jpg"
    image2 = cv2.imread(filename=image2_path, flags=cv2.IMREAD_COLOR)

    result = classifier.infer(input_batch=[image, image2])
    probabilities = classifier.get_probabilities(inference_output=result)
    print(probabilities)
