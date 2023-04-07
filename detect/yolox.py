import sys
import os
workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolox_path = os.path.join(workdir, 'detect', 'yolox')
sys.path.insert(0,yolox_path)


import torch
from exps.default.yolox_x import Exp

ckpt_file = os.path.join(workdir, 'detect/weights/yolox_x.pth')

__all__ = ['YoloX']


def YoloX():
    exp = Exp()  # load model
    model = exp.get_model()

    ckpt = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    return model