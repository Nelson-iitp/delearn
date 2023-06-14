
__doc__=r"""
:py:mod:`delearn/optim.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

    'GradientOptimizer', 'SGD'
]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
#import torch.autograd as ag
from .module import Module
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class GradientOptimizer:
    def __init__(self, sign) -> None:
        self.sign = sign# -1 for gradient descent = minimize loss
    
    def __call__(self, module, grads, create_graph):
        return self.call_with_grad(module, grads) if create_graph else self.call_no_grad(module, grads)

class SGD(GradientOptimizer):
    def __init__(self, maximize=False, lr=0.01, weight_decay=0.0, momentum=0.0, dampening=0.0, nesterov=False) -> None:
        super().__init__(1 if maximize else -1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum=momentum
        self.dampening=dampening
        self.nesterov = nesterov
        # local buffers
        self.bt = None

    def call_step(self, module, grads):
        if self.weight_decay:
            grads = [ g + self.weight_decay*p for g,p in zip(grads,module.parameters()) ]
        if self.momentum:
            self.bt = grads if self.bt is None else ([self.momentum*b+(1-self.dampening)*g for b,g in zip(self.bt,grads)])
            if self.nesterov:
                grads = [ g + self.momentum*b for g,b in zip(grads,self.bt) ]
            else:
                grads = self.bt
        return grads
    
    def call_with_grad(self, module, grads):
        grads = self.call_step(module, grads)
        res = {n:p+self.sign*self.lr*g for g,(n,p) in zip(grads, module.items())}
        return Module(res)
    
    @tt.no_grad()
    def call_no_grad(self, module, grads):
        grads = self.call_step(module, grads)
        for g,p in zip(grads, module.parameters()):
            p+=self.sign*self.lr*g
        return module
