import sys
import os
import torch



workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolor_path = os.path.join(workdir, 'detect', 'yolor')
sys.path.insert(0,yolor_path)

from models.models import Darknet

cfg = os.path.join(yolor_path, 'cfg/yolor_p6.cfg')  # model.yaml path
img_size = 640
ckpt_file = os.path.join(workdir, 'detect/weights/yolor_p6.pt')

__all__ = ['YoloR']

def YoloR():
    model = Darknet(cfg, img_size)  # create model
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu')['model'])  # load weights
    return model
