import torch
import pandas as pd

from detect import YoloV4, YoloX, FasterRCNN, YoloR, YoloS
from profiler import Profiler
from hook import TraceHook, ProfHook


from torchviz import make_dot

model = YoloS().eval()


def generate_dep(model):
    hook = TraceHook()
    profiler = Profiler(model, "yolos", hook)

    image = torch.rand(1, 3, 416, 416)
    image.__tag__ = "input"

    record = profiler.run(image)
    df = pd.DataFrame(record)
    df.to_csv("dep.csv", index=False)


def generate_prof(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image = torch.rand(1, 3, 640, 640).to(device)


    hook = ProfHook()
    hook.add_record("input")

    profiler = Profiler(model, "yolos", hook)
    record = profiler.run(image)
    df = pd.DataFrame(record)
    df.to_csv("prof.csv", index=False)


def yolos_profiler(hook, filename="prof.csv", tag = False):
    hooks = []
    model = YoloS().eval()
    #warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image = torch.rand(1, 3, 640, 640).to(device)
    
    if tag:
        image.__tag__ = "input"

    model(image)

    image_size = image.numel() * 4 / 1024 / 1024
    hook._record.append({"layer_name" : "input", "time" : 0, "cpu_mem" : 0, "cuda_mem": 0, "size" : image_size,  "MACs": 0})


    hooks.append(model.embeddings.register_forward_pre_hook(hook.pre(f"embeddings")))
    hooks.append(model.embeddings.register_forward_hook(hook.post(f"embeddings")))


    for i, layer in enumerate(model.encoder.layer):
        hooks.append(layer.register_forward_pre_hook(hook.pre(f"encoder_{i}")))
        hooks.append(layer.register_forward_hook(hook.post(f"encoder_{i}")))


    hooks.append(model.encoder.interpolation.register_forward_pre_hook(hook.pre(f"encoder_interpolation")))
    hooks.append(model.encoder.interpolation.register_forward_hook(hook.post(f"encoder_interpolation")))


    hooks.append(model.layernorm.register_forward_pre_hook(hook.pre(f"layer_norm")))
    hooks.append(model.layernorm.register_forward_hook(hook.post(f"layer_norm")))

    hooks.append(model.pooler.register_forward_pre_hook(hook.pre(f"pooler")))
    hooks.append(model.pooler.register_forward_hook(hook.post(f"pooler")))

    


    with torch.no_grad():
        model(image)

    hook._record.append({"layer_name" : "output", "time" : 0, "cpu_mem" : 0, "cuda_mem": 0, "size" : 0,  "MACs": 0})

    for h in hooks:
        h.remove()


    df = pd.DataFrame(hook.record)
    df.to_csv(filename, index=False)

    print(model)

yolos_profiler(ProfHook(), "prof.csv")
