import torch
import pandas as pd

from hook import *

class Profiler:
    def __init__(self, model: torch.nn.Module, name: str, hook: Hook, layers:tuple = None) -> None:
        self.model = model
        self.name = name
        self.layers = layers
        self.all_layers = False

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        if self.layers is None:
            self.all_layers = True

        self.hook = hook
        self.pre_hooks, self.post_hooks = [], []

    def attach_hooks(self, model: torch.nn.Module, name:str) -> None:
        if len(list(model.named_children())) == 0:
            if self.all_layers:
                self._attach_hooks(model, name)
            else:
                if isinstance(model, self.layers):
                    self._attach_hooks(model, name)
            return
        
        for n, m in model.named_children():
            self.attach_hooks(m, name + "." + n)

    def _attach_hooks(self, model: torch.nn.Module, name:str) -> None:
        self.pre_hooks.append(model.register_forward_pre_hook(self.hook.pre(name)))
        self.post_hooks.append(model.register_forward_hook(self.hook.post(name)))
        
    def detach_hooks(self) -> None:
        for hook in self.pre_hooks:
            hook.remove()
        for hook in self.post_hooks:
            hook.remove()

    def run(self, image: torch.Tensor) -> Record:
        #warm up
        self.model(image)

        self.attach_hooks(self.model, self.name)

        with torch.no_grad():
            self.model(image)

        self.detach_hooks()

        return self.hook.record




