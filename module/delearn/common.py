__doc__=r"""
:py:mod:`delearn/common.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    
    'default_rng', 'Identity', 'getF', 'numel', 'arange', 'shares_memory', 
    'Task', 'Modular',

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from io import BytesIO
import numpy as np


import torch as tt
from torch.utils.data import Dataset #, DataLoader
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

default_rng = np.random.default_rng

def Identity(input): return input

def getF(F:tuple, *A): 
    r""" NOTE: 
    #   F-tuple is a 2-tuple like (function:Callable, args:Dict) 
    #   getF is shorhand for getting F-tuples with additional arguments *A passed to the callable
    """
    return F[0](*A, **F[1])  

def numel(shape)->int: 
    r""" Counts total element in a given shape """
    return tt.prod(tt.tensor(shape)).item()

def arange(shape, start:int=0, step:int=1, dtype=None): 
    r""" Similar to ``torch.arange`` but reshapes the tensor to given shape """
    return tt.arange(start=start, end=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

def shares_memory(a, b) -> bool: 
    r""" Checks if two tensors share same underlying storage, in which case, changing values of one will change values in the other as well.

    .. note:: This is different from ``Tensor.is_set_to(Tensor)`` function which checks the shape as well.
    """
    return (a.storage().data_ptr() == b.storage().data_ptr())


class Task(Dataset):
    r"""
    Defines a Task i.e., a collection of (x, y) pairs of (inputs, labels)
    """
    def __init__(self, x, y, seed=None) -> None:
        super().__init__() #assert(len(self.x)==len(self.y))
        # x and y have first dimension as batch
        assert(tt.is_tensor(x))
        assert(tt.is_tensor(y))
        assert x.ndim>1, f'expecting (batch, ... )'
        assert y.ndim>1, f'expecting (batch, ... )'
        assert x.shape[0]==y.shape[0], f'expecting equal count'
        
        self.x, self.y = x, y
        self.xshape, self.yshape = self.x.shape[1:], self.y.shape[1:]
        self.n = x.shape[0]
        self.rng = default_rng(seed)
        self.mask = None

    def __len__(self): return self.n

    def __getitem__(self, i): return self.x[i], self.y[i]

    def __call__(self, batch_size=None, shuffle=False, drop_last=False): 
        # check if batch_size if larger than total len
        if batch_size is None:  batch_size=self.n
        elif batch_size>self.n:
            print(f'[{__class__}]:: batch-size is larger than task-size, setting to max size {self.n}')
            self.batch_size=self.n
        else: self.batch_size = batch_size
        self.b = int(self.n/self.batch_size) + int((self.n%self.batch_size)>0)
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.mask = np.ones(shape=(self.n,), dtype=np.bool8)
        self.ranger = np.arange(self.n)
        return self
    
    def __iter__(self):
        if self.mask is None: raise ValueError(f'first call the task object to set batch_size, shuffle, drop_last')
        self.mask[:] = True
        return self

    def __next__(self):
        sampler = self.ranger[self.mask]
        nsamples = len(sampler)
        if nsamples<self.batch_size:
            if nsamples==0 or self.drop_last: raise StopIteration
            batch_indices = sampler
        else: # NOTE: we can change self.shuffle mid training to shuffle randomly from remaining
            batch_indices = self.rng.choice(sampler, size=self.batch_size, replace=False) if self.shuffle else sampler[0:self.batch_size]
        self.mask[batch_indices]=False
        return self.x[batch_indices], self.y[batch_indices]

class Modular:
    r"""
    Defines operations on collection of named parameters stored in a dict ```module:dict[str, Tensor]```
    """

    @staticmethod
    def count(module:dict, requires_grad=None): 
        r""" Counts the total number of parameters (numel) in a module
        
        :param requires_grad: 
            if None, counts all parameters
            if True, counts trainable parameters
            if False, counts non-trainiable (frozen) parameters
        """
        return sum( ([ p.numel() for p in module.values() ]) if requires_grad is None else \
                    ([ p.numel() for p in module.values()    if p.requires_grad is requires_grad ]) )

    @staticmethod
    def show(module:dict, values:bool=False):
        r""" Prints the parameters of a module
        
        :param values: if True, prints the full tensors otherwise prints only shape
        """
        nos_trainable, nos_frozen = 0, 0
        print('=====================================')
        for i,(n,p) in enumerate(module.items()):
            iparam = p.numel()
            if p.requires_grad:
                nos_trainable += iparam
            else:
                nos_frozen += iparam
            print(f'#[{i} :: {n}]\tShape[{p.shape}]\tParams: {iparam}\tTrainable: {p.requires_grad}')
            if values: 
                print('=====================================')
                print(f'{p}')
                print('=====================================')
        print(f'\nTotal Parameters: {nos_trainable+nos_frozen}\tTrainable: {nos_trainable}\tFrozen: {nos_frozen}')
        print('=====================================')
        return 

    @staticmethod
    @tt.no_grad()
    def copy(module_from:dict, module_to:dict) -> None:
        r""" Copies the parameters of a module to another - both modules are supposed to be identical"""
        for pt,pf in zip(module_to.values(), module_from.values()): pt.copy_(pf)

    @staticmethod
    @tt.no_grad()
    def diff(module1:dict, module2:dict, do_abs:bool=True, do_sum:bool=True):
        r""" Checks the difference between the parameters of two modules.
         This can be used to check if two models have exactly the same parameters.

        :param do_abs: if True, finds the absolute difference
        :param do_sum: if True, finds the sum of difference

        :returns: a list of differences in each parameter or their sum if ``do_sum`` is True.
        """
        d = [ (abs(p1 - p2) if do_abs else (p1 - p2)) for p1,p2 in zip(module1.values(), module2.values()) ]
        if do_sum: d = [ tt.sum(p) for p in d  ]
        return d

    @staticmethod
    @tt.no_grad()
    def rand(module:dict, lb=-0.1, ub=0.1, generator=None):
        r""" Randomizes the parameters with given distribution

        param:tt.Tensor

        param.bernoulli_            (p=0.5)                             # p can be tensor a well
        param.cauchy_               (median=0.0, sigma=1.0)
        param.exponential_          (lambd=1.0)
        param.geometric_            (p=0.5)
        param.log_normal_           (mean=1.0, std=2.0)
        param.normal_               (mean=0.0, std=1.0)
        param.random_               (from_=0, to=10)                    # int arguments
        param.uniform_              (from_=0.0, to=1.0)
        """
        for p in module.values():   p.uniform_(lb, ub, generator=generator)

    @staticmethod
    def clone(module:dict):
        r""" Clones the parametrers of a module"""
        return {n:p.clone().detach().requires_grad_() for n,p in module.values()}

    @staticmethod
    def duplicate(module:dict, n_copies:int=1):
        r""" Duplicates a module by storing it in a buffer and retriving many copies"""
        buffer = BytesIO()
        tt.save(module, buffer)
        model_copies = []
        for _ in range(n_copies):
            buffer.seek(0)
            model_copy = tt.load(buffer)
            model_copies.append(model_copy)
        buffer.close()
        del buffer
        return model_copy if n_copies==1 else model_copies

    @staticmethod
    def save(module:dict, path:str): tt.save(module, path)

    @staticmethod
    def load(path:str): return tt.load(path)

    @staticmethod
    def requires_grad(module, requires_grad: bool = True):
        r""" Sets requires_grad attribute on all tensors in a module"""
        for p in module.values(): p.requires_grad_(requires_grad)

    @staticmethod
    def from_size(module_size, dtype=None, device=None, requires_grad=True):
        return { name:tt.zeros(size=size, dtype=dtype, device=device, requires_grad=requires_grad)  for name,size in module_size.items() }
    
    @staticmethod
    def from_like(module_like, requires_grad=True):
        return { name:tt.zeros_like(param, requires_grad=requires_grad)  for name,param in module_like.items() }


