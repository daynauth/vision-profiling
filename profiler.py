import torch

from hook import *

def walk_layers(model: torch.nn.Module, name:str, action: callable) -> None:
    if len(list(model.named_children())) == 0:
        action(model, name)
        return
    
    for n, m in model.named_children():
        walk_layers(m, name + "." + n, action)

class Profiler:
    def __init__(self, model: torch.nn.Module, name: str, hook: Hook, layers:tuple = None) -> None:
        self.model = model
        self.name = name
        self.layers = layers
        self.all_layers = False
        
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
        map(lambda hook: hook.remove(), self.pre_hooks)
        map(lambda hook: hook.remove(), self.post_hooks)

    def run(self, image: torch.Tensor, warmup_steps:int = 1) -> Record:
        #warm up
        self._warmup(image, warmup_steps)
        self.attach_hooks(self.model, self.name)
        self._infer(image)
        self.detach_hooks()

        return self.hook.record
    
    def _warmup(self,image: torch.Tensor, warmup_steps:int = 1) -> None:
        for _ in range(warmup_steps):
            self.model(image)
    
    def _infer(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): 
            self.model(image)




