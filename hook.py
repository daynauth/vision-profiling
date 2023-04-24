import torch
import pandas as pd
from abc import ABC, abstractmethod
from record import *


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
            #self.prof.events()
            if(isinstance(module, torch.nn.Conv2d)):
                events = self.prof.key_averages()
                for event in events:
                    if event.key == name:
                        self._record.append(event.cuda_memory_usage)
                        break
                    
                    

        return hook

class TraceHook(Hook):
    def __init__(self):
        self._layers = []

    @property
    def record(self):
        return self._layers

    def pre(self, name: str) -> callable:
        def hook(module, input):
            
            pass
            #print(type(input))
        return hook

    def post(self, name: str) -> callable:
        def hook(module, input, output):
            if len(input) == 1:
                if hasattr(input[0], '__tag__'):
                    self._layers.append({"src" : input[0].__tag__, "dst" : name})
                else:
                    pass
                    

            else:
                for i in range(len(input)):
                    if isinstance(input[i], torch.Tensor):
                        if hasattr(input[0], '__tag__'):
                            print(input[i].__tag__)
                        print(input[i].shape)
                # print(type(input[0]))
                # print(type(input[1]))
                # print(input[1])
                print(name)
                    
            if isinstance(output, torch.Tensor):
                output.__tag__ = name
            elif isinstance(output, tuple):
                if len(output) == 1:
                    output[0].__tag__ = name

        return hook
    
class ProfHook(Hook):
    def __init__(self):
        self._record = []
        self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.ref = None


    @property
    def record(self):
        return self._record
    

    def pre(self, name: str) -> callable:
        def hook(module, input):
            self.starter.record()
            self.prof.__enter__()
            self.ref = record_function(name)
            self.ref.__enter__()
        return hook
    
    def add_record(self, name: str):
        self._record.append({"layer_name" : name, "time" : 0, "cpu_mem" : 0, "cuda_mem": 0, "size" : 0,  "MACs": 0})

    def post(self, name: str) -> callable:
        def hook(module, input, output):
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)

            self.ref.__exit__(None, None, None)
            self.prof.__exit__(None, None, None)
 
            events = self.prof.key_averages()

            cuda_mem = 0
            cpu_mem = 0
            for event in events:
                if event.key == name:
                    cuda_mem = event.cuda_memory_usage * 1.0 / 1024 / 1024
                    cpu_mem = event.cpu_memory_usage * 1.0 / 1024 / 1024
                    break

            size = 0
            #calculate size of output tensor

            
            
                
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size() / 1024 / 1024
            elif isinstance(output, tuple):
                if len(output) == 1:
                    size = output[0].numel() * output[0].element_size() / 1024 / 1024

            self._record.append({"layer_name" : name, "time" : end_time, "cpu_mem" : cpu_mem, "cuda_mem": cuda_mem, "size" :  size, "MACs": 0})

        return hook