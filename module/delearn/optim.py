
import torch as tt
#import torch.autograd as ag
from .module import Module

__all__ = ['SGD']
class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def __call__(self, module, grads, create_graph=False):
        if create_graph:
            res_params = {}
            for n,p,g in zip(module.names(), module.parameters(), grads):
                res_params[n]=p-self.lr*g
            return Module(res_params)
        else:
            with tt.no_grad():
                for p,g in zip(module.parameters(), grads):
                    p-=self.lr*g
            return module




