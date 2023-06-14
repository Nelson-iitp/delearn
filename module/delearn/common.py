__doc__=r"""
:py:mod:`delearn/common.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    
    'getF', 'numel', 'arange', 'shares_memory', 

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# F tuple is a 2-tuple like (Callable, Dict)
# using getF - Callable is a function which is called with kwargs provided in dict (passed as **Dict)
# can pass additional arguments to *A to args
def getF(F, *A): return F[0](*A, **F[1])  # shorhand for getting from tuples like (function, args_dict)

def numel(shape)->int: 
    r""" Returns the number of elements in an array of given shape

    .. note:: for ``torch.Tensor`` use ``Tensor.numel()``
    """
    return tt.prod(tt.tensor(shape)).item()

def arange(shape, start:int=0, step:int=1, dtype=None): 
    r""" Similar to ``torch.arange`` but reshapes the tensor to given shape """
    return tt.arange(start=start, end=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

def shares_memory(a, b) -> bool: 
    r""" Checks if two tensors share same underlying storage, in which case, changing values of one will change values in the other as well.

    .. note:: This is different from ``Tensor.is_set_to(Tensor)`` function which checks the shape as well.
    """
    return (a.storage().data_ptr() == b.storage().data_ptr())

