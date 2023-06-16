__doc__=r"""
:py:mod:`delearn/module.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    
    'Module', 'Modular',

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from typing import Any
import torch as tt
from io import BytesIO
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class Module:
    r"""A dict of parameters containing named tensors that require grad. 
    
    # Implements __iter__ :: To iter over names only, (parameters can be accessed by name) 
        ~ Iterator over names```for name in module: parameter = module[name]```

    # Implements __call__ :: To iter over names and parameters, 
        ~ Iterator over parameters only:: ```for parameter in module()```, similar to torch ```for parameter in module.parameters()```
        ~ Iterator over name-parameter pair :: ```for name, parameter in module(True)```

    # The only member defined in the class is ```self._parameters``` which is a dict
    """

    def __init__(self, parameters=None) -> None: self._parameters = {} if parameters is None else parameters

    def __len__(self): return self._parameters.__len__()

    def __getitem__(self, i): return self._parameters[i]

    def __iter__(self): return self._parameters.__iter__() 

    def __call__(self, names=False): return  self._parameters.items() if names else self._parameters.values()  
    
class Modular:

    @staticmethod
    def count_parameters(module, requires_grad=None): 
        if requires_grad is None: # count all parameters
            return sum([p.numel() for p in module()]) 
        else:
            return sum([p.numel() for p in  module() if p.requires_grad is requires_grad ])

    @staticmethod
    def show_parameters(module, values:bool=False):
        nos_trainable, nos_frozen = 0, 0
        print('=====================================')
        for i,(n,p) in enumerate(module(True)):
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
    def copy_parameters(module_from, module_to) -> None:
        r""" Copies module parameters, both modules are supposed to be identical """
        for pt,pf in zip(module_to(), module_from()): pt.copy_(pf)

    @staticmethod
    @tt.no_grad()
    def diff_parameters(module1, module2, do_abs:bool=True, do_sum:bool=True):
        r""" Checks the difference between the parameters of two modules. This can be used to check if two models have exactly the same parameters.

        :param module1: an instance of ``nn.Module``
        :param module: an instance of ``nn.Module``
        :param do_abs: if True, finds the absolute difference
        :param do_sum: if True, finds the sum of difference

        :returns: a list of differences in each parameter or their sum if ``do_sum`` is True.
        """
        d = [ (abs(p1 - p2) if do_abs else (p1 - p2)) for p1,p2 in zip(module1(), module2()) ]
        if do_sum: d = [ tt.sum(p) for p in d  ]
        return d

    @staticmethod
    @tt.no_grad()
    def rand_parameters(module, lb=-0.1, ub=0.1):
        for p in module(): p.data.copy_(tt.rand_like(p.data) * (ub-lb) + lb)

    @staticmethod
    def clone_parameters(module, n_copies:int=1, detach:bool=False):

        buffer = BytesIO()
        
        tt.save(module, buffer)
        model_copies = []
        for _ in range(n_copies):
            buffer.seek(0)
            model_copy = tt.load(buffer)
            if detach:  
                for p in model_copy(): p.requires_grad_(False)
            model_copies.append(model_copy)
        buffer.close()
        del buffer
        return model_copy if n_copies==1 else model_copies

    @staticmethod
    def requires_grad(module, requires_grad: bool = True):
        for p in module(): p.requires_grad_(requires_grad)

    @staticmethod
    def zero_grad(module, set_to_none: bool = False) -> None:
        for p in module():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()


    """ NOTE: Module Creators: 
    ```creatorF``` accepts a ```dict[str, Tensor]``` and return an object (usually a ```Module``` object )
    ```creatorF``` is (usually) the ```Module``` class and calls the ```__init__`` method 
    """

    @staticmethod
    def create_from_parameter_size(creatorF, parameters_size, dtype=None, device=None, requires_grad=True):
        return creatorF({ 
            name:tt.zeros(size=size, dtype=dtype, device=device, requires_grad=requires_grad)  for name,size in parameters_size.items() 
        })
    
    @staticmethod
    def create_from_parameter_like(creatorF, parameters_like, requires_grad=True):
        return creatorF({
            name:tt.zeros_like(param, requires_grad=requires_grad)  for name,param in parameters_like.items()
        })
    