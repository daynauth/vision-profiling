import torch
import pandas as pd

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
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'layer_name'}, inplace=True)
        return df


class Profiler:
    def __init__(self, model: torch.nn.Module, name: str, layers:tuple = None) -> None:
        self.model = model
        self.name = name
        self.layers = layers
        self.all_layers = False

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        if self.layers is None:
            self.all_layers = True

        
        self.record = Record()
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
        self.pre_hooks.append(model.register_forward_pre_hook(self.pre_time_hook(name)))
        self.post_hooks.append(model.register_forward_hook(self.post_time_hook(name)))
    
    def pre_time_hook(self, name: str) -> callable:
        def hook(module, input):
            self.starter.record()
        return hook
    
    def post_time_hook(self, name: str) -> callable:
        def hook(module, input, output):
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)

            module_type = type(module).__name__
            self.record.insert(name, module_type, end_time)
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
            self.model(image)

        self.detach_hooks()

        return self.record




