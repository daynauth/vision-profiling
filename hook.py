import torch
import pandas as pd
from abc import ABC, abstractmethod
from record import *
import torchvision

from torch.profiler import profile, record_function, ProfilerActivity

class Hook(ABC):
    @property
    def record(self) -> Record:
        pass

    @abstractmethod
    def pre(self, name: str) -> callable:
        pass

    @abstractmethod
    def post(self, name: str) -> callable:
        pass

class TimeHook(Hook):
    def __init__(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self._record = TimeRecord()

    @property
    def record(self):
        return self._record

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
            self._record.insert(name, module_type, end_time)
        return hook
    
class MemoryHook(Hook):
    def __init__(self):
        self._record = []
        self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True)
        self.ref = None

    @property
    def record(self):
        return self._record
    
    def pre(self, name: str) -> callable:
        def hook(module, input):
            self.prof.__enter__()
            self.ref = record_function(name)
            self.ref.__enter__()
        return hook
    
    def post(self, name: str) -> callable:
        def hook(module, input, output):
            self.ref.__exit__(None, None, None)
            self.prof.__exit__(None, None, None)
            self.prof.events()
            self._record.append(self.prof.key_averages().table(sort_by="cuda_memory_usage"))
        return hook