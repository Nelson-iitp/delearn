__doc__=r"""
:py:mod:`delearn/module.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    
    'count_parameters', 'count_trainiable_parameters', 'count_non_trainiable_parameters',
    'copy_parameters', 'show_parameters', 'diff_parameters', 'rand_parameters', 'clone_parameters',
    'requires_grad', 'zero_grad', 'Module',

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from typing import Any
import torch as tt
from torch.nn import ParameterDict, Parameter
from io import BytesIO
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



def count_parameters(module): return sum([p.numel() for p in module.parameters()]) 

def count_trainiable_parameters(module):  return sum([p.numel() for p in  module.parameters() if p.requires_grad ])

def count_non_trainiable_parameters(module):  return sum([p.numel() for p in  module.parameters() if not p.requires_grad ])

    
def show_parameters(module, values:bool=False) -> int:
    r""" Prints the parameters using ``nn.Module.parameters()``

    :param module: an instance of ``nn.Module``
    :param values: If `True`, prints the full parameter tensor 

    :returns:   total number of parameters in the module
    """
    nparam=0
    for i,p in enumerate(module.parameters()):
        iparam = p.numel()
        nparam += iparam
        print(f'#[{i+1}]\tShape[{p.shape}]\tParams: {iparam}')
        if values: print(f'{p}')
    print(f'Total Parameters: {nparam}')
    return 

@tt.no_grad()
def copy_parameters(module_from, module_to) -> None:
    r""" Copies module parameters, both modules are supposed to be identical """
    for pt,pf in zip(module_to.parameters(), module_from.parameters()): pt.copy_(pf)

@tt.no_grad()
def diff_parameters(module1, module2, do_abs:bool=True, do_sum:bool=True):
    r""" Checks the difference between the parameters of two modules. This can be used to check if two models have exactly the same parameters.

    :param module1: an instance of ``nn.Module``
    :param module: an instance of ``nn.Module``
    :param do_abs: if True, finds the absolute difference
    :param do_sum: if True, finds the sum of difference

    :returns: a list of differences in each parameter or their sum if ``do_sum`` is True.
    """
    d = [ (abs(p1 - p2) if do_abs else (p1 - p2)) for p1,p2 in zip(module1.parameters(), module2.parameters()) ]
    if do_sum: d = [ tt.sum(p) for p in d  ]
    return d

@tt.no_grad()
def rand_parameters(module, lb=-0.1, ub=0.1):
    for p in module.parameters(): p.data.copy_(tt.rand_like(p.data) * (ub-lb) + lb)


def requires_grad(module, requires_grad: bool = True):
    for p in module.parameters(): p.requires_grad_(requires_grad)

def zero_grad(module, set_to_none: bool = False) -> None:
    for p in module.parameters():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

def clone_parameters(module, n_copies:int=1, detach:bool=False):

    buffer = BytesIO()
    
    tt.save(module, buffer)
    model_copies = []
    for _ in range(n_copies):
        buffer.seek(0)
        model_copy = tt.load(buffer)
        if detach:  requires_grad(model_copy, False)
        model_copies.append(model_copy)
    buffer.close()
    del buffer
    return model_copy if n_copies==1 else model_copies

class Module:
    r"""An ordered collection of named parameters that require grad. Parameters can be accessed by both key and index"""

    def __init__(self, parameters) -> None:
        # self.forward(self._parameters, *args, **kwargs)
        if isinstance(parameters, dict):
            iters = parameters.items()
        elif isinstance(parameters, list):
            iters = enumerate(parameters, 0)
        else:
            raise ValueError(f'parameters must be Dict or List')
        
        _parameters = {}
        _names = []
        for key, param in iters:
            name = f'{key}'
            assert isinstance(name, tt._six.string_classes), f"param name should be a string."
            assert name, "param name can't be empty string \"\""
            assert '.' not in name, "param name can't contain \".\""        

            assert isinstance(param, Parameter)
            assert not(param.grad_fn), "Cannot assign non-leaf Tensor to parameter"

            _names.append(name)
            _parameters[name] = param

        self._parameters = ParameterDict(_parameters)
        self._names = _names

    def __getitem__(self, key):
        return self._parameters[ self._names[key] if isinstance(key, int)  else key ]

    def parameters(self):
        for name in self._names: yield self._parameters[name]

    def names(self):
        for name in self._names: yield name

    def __iter__(self): return enumerate(zip(self.names(), self.parameters()))

    def requires_grad(self, requires_grad: bool = True): 
        requires_grad(self, requires_grad)
        return self

