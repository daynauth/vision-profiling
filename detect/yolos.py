from transformers import YolosConfig, YolosModel
import torch

__all__ = ['YoloS']

def YoloS():
    configuration = YolosConfig()
    model = YolosModel(configuration)

    return model


