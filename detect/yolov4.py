import sys
import os
workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolov4_path = os.path.join(workdir, 'detect', 'yolov4')
sys.path.insert(0,yolov4_path)

from tool.darknet2pytorch import Darknet

__all__ = ['YoloV4']

def YoloV4():
    cfg = os.path.join(yolov4_path, 'cfg/yolov4.cfg')
    model = Darknet(cfg)  # load darknet model
    model.load_weights(os.path.join(workdir, 'detect/weights/yolov4.pth'))  # load weights
    return model