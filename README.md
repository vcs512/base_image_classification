# Base Image Classification

Architecture and code to be used for Computer Vision Image Classification task.

## Base References

- [Hugging Face Community Computer Vision Course](https://huggingface.co/docs/transformers/tasks/image_classification)

## Environment Setup

### Docker

Execution code:

- [Install](https://docs.docker.com/engine/install/)
- [Remove sudo](https://docs.docker.com/engine/install/linux-postinstall/)

### NVIDIA GPU

Enable GPU usage:

- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### uv

Python dependency manager:

- [Install](https://docs.astral.sh/uv/getting-started/installation/)

## File Structure

```bash
.
├── data/
├── docker/
├── experiments/
├── models/
│
├── src/
│   ├── classifier/
│   ├── dataset/
│   ├── evaluator/
│   ├── trainer/
│   └── main.py
│
├── pyproject.toml
├── README.md
└── uv.lock
```

Intended usage:

- `data/`: Store datasets and input data
- `docker/`: Store dockerfiles
- `experiments/`: Store artifacts of training and evaluation (models, logs, etc)
- `models/`: Store models and configurations for inference (deploy)
- `src/`: Project code
    1. `classifier/`: Classifier model (inference, pre/post-processing)
    1. `dataset/`: Dataset dealer (dataloader)
    1. `evaluator/`: Evaluation metrics dealer
    1. `trainer/`: Orchestrator for training procedure (configurations applier)
    1. `main.py`: Function selector (API routes)
