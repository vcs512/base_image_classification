from pydantic import BaseModel


class ClassificationPipelineOutputInterface(BaseModel):
    """Image classification output pipeline mimic"""
    label: str
    score: float
