
import torch as tt
from torch.utils.data import Dataset, DataLoader

__all__ = [

    'Task', 

]

class Task(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        #assert(len(self.x)==len(self.y))
        self.x = x
        self.y = y
        self.count = len(self.x)
    
    def __len__(self): return self.count

    def __getitem__(self, index): return self.x[index], self.y[index]

