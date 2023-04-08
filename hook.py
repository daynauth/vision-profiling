import torch
import pandas as pd
from abc import ABC, abstractmethod

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

class Hook(ABC):
    @abstractmethod
    def pre(self, name: str) -> callable:
        pass

    @abstractmethod
    def post(self, name: str) -> callable:
        pass

class TimeHook(Hook):
    def __init__(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.record = Record()

    def pre(self, name: str) -> callable:
        def hook(module, input):
            self.starter.record()
        return hook

    def post(self, name: str) -> callable:
        def hook(module, input, output):
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)

            module_type = type(module).__name__
            self.record.insert(name, module_type, end_time)
        return hook