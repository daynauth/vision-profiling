import torch
import pandas as pd
import time

class Record:
    def __init__(self) -> None:
        self.data = {}
        self.total_time = 0

    def insert(self, key1: str, key2: str, value: float) -> None:
        self.data[key1] = {"layer_type" : key2, "time" : value}

    def get(self, key1: str, key2: str = None) -> dict:
        return self.data[key1]
    
    def __str__(self) -> str:
        output = ''
        for layer_type in self.data:
            output += f'{layer_type}: {self.data[layer_type]["time"]} ms' + "\n"

        return output
    
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.data, orient='index', )
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'layer_name'}, inplace=True)
        return df


class Profiler:
    def __init__(self, model: torch.nn.Module, name: str, layers:tuple = None) -> None:
        self.model = model
        self.name = name
        self.layers = layers
        self.all_layers = False
        
        if self.layers is None:
            self.all_layers = True

        
        self.record = Record()
        self.pre_hooks, self.post_hooks = [], []

    def attach_hooks(self, model: torch.nn.Module, name:str) -> None:
        if len(list(model.named_children())) == 0:
            if self.all_layers:
                self.pre_hooks.append(model.register_forward_pre_hook(self.pre_time_hook(name)))
                self.post_hooks.append(model.register_forward_hook(self.post_time_hook(name)))
            else:
                if isinstance(model, self.layers):
                    self.pre_hooks.append(model.register_forward_pre_hook(self.pre_time_hook(name)))
                    self.post_hooks.append(model.register_forward_hook(self.post_time_hook(name)))
            return
        
        for n, m in model.named_children():
            self.attach_hooks(m, name + "." + n)
    
    def pre_time_hook(self, name: str) -> callable:
        def hook(module, input):
            module_type = type(module).__name__
            start_time = time.time()
            self.record.insert(name, module_type, start_time)
        return hook
    
    def post_time_hook(self, name: str) -> callable:
        def hook(module, input, output):
            end_time = time.time()
            module_type = type(module).__name__
            self.record.insert(name, module_type,(end_time - self.record.get(name)['time']) * 1000)
        return hook
    
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
            #get total time for inference
            start_time = time.time()
            self.model(image)
            end_time = time.time()
            self.record.total_time = end_time - start_time

        self.detach_hooks()

        return self.record




