
__doc__=r"""
:py:mod:`delearn/optim/sgd.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

    'SGD', 
]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
from . import GradientOptimizer
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class SGD(GradientOptimizer):
    def __init__(self, maximize=False, lr=0.01, weight_decay=0.0, momentum=0.0, dampening=0.0, nesterov=False) -> None:
        super().__init__(1 if maximize else -1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum=momentum
        self.dampening=dampening
        self.nesterov = nesterov
        # self.build() #<---- build must be called

    def build(self):
        # local buffers
        self.last_grads = None
        self.time = 0 # should be last
        return self

    def call_step(self, params, grads):
        self.time += 1 # should be first
        if self.weight_decay: grads = [ g + self.weight_decay*p for g,p in zip(grads, params) ]
        if self.momentum:
            self.last_grads = grads if self.time==1 else ([self.momentum*b+(1-self.dampening)*g for b,g in zip(self.last_grads,grads)])
            grads = [ g + self.momentum*b for g,b in zip(grads,self.last_grads) ] if self.nesterov else self.last_grads
        return grads
    
    def call_with_grad(self, module, grads):
        step_grads = self.call_step(module.values(), grads)
        return {n:p+self.sign*self.lr*g for g,(n,p) in zip(step_grads, module.items())}
    
    @tt.no_grad()
    def call_no_grad(self, module, grads):
        step_grads = self.call_step(module.values(), grads)
        for g,p in zip(step_grads, module.values()):
            p+=self.sign*self.lr*g
        return module


