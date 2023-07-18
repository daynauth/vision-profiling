from dataclasses import dataclass
import csv
import os 


@dataclass
class Layer:
    name: str
    time: float
    cpu_mem: float
    cuda_mem: float
    size: float
    macs: float


@dataclass
class Partition:
    name: str
    device: int

class ModelData:
    def __init__(self, fine = True, split = True, size = 2):
        self.fine = fine
        self.split = split
        self.size = size
        self.path = f'../optimizer/testcases/yolos-agx/1/'
        self.filename = self._get_filename()
        self.profile_path = self._get_profile_path()
        self.layers = []
        self.result_dir = self._get_results()
        self.part_file = os.path.join(self.result_dir, f'yolos-agx_{self.size}gb', 'part.csv')
        self.partitions = []

        self._get_layers()
        self._get_partitions()




    def _get_filename(self):
        filename = 'fine' if self.fine else 'coarse'
        filename += '_split' if self.split else '_no_split'
        return filename
    
    def _get_profile_path(self):
        return os.path.join(self.path, f"prof_{self.filename}.csv")
    
    def _get_layers(self):
        with open(self.profile_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            self.layers = [Layer(*row) for row in reader]

    def _get_partitions(self):
        with open(self.part_file) as f:
            reader = csv.reader(f)
            next(reader, None)
            self.partitions = [Partition(*row) for row in reader]

    def _get_results(self):
        results_dir = "../results"
        return os.path.join(results_dir, self.filename)

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None
    
    def get_partition(self, name):
        for partition in self.partitions:
            if partition.name == name:
                return partition
        return None
    
    def get_encoder_layers(self, id = 0):
        return [layer for layer in self.layers if layer.name.startswith(f'layer_{id}')]