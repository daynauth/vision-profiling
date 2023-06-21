from transformers import YolosModel

__all__ = ['YoloS']

def YoloS():
    model = YolosModel.from_pretrained("hustvl/yolos-base")
    return model


