import pandas as pd
from abc import ABC, abstractmethod

class Record(ABC):
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        pass

class TimeRecord(ABC):
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