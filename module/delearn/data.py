__doc__=r"""
:py:mod:`delearn/data.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

    'Task', 
]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#import torch as tt
from torch.utils.data import Dataset, DataLoader
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class Task(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        #assert(len(self.x)==len(self.y))
        self.x = x
        self.y = y
        self.count = len(self.x)
    
    def __len__(self): return self.count

    def __getitem__(self, index): return self.x[index], self.y[index]

    def __call__(self, batch_size, shuffle, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)