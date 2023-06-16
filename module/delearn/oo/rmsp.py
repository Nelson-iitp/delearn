
__doc__=r"""
:py:mod:`delearn/optim/rmsp.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

   'RMSprop',
]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
from . import GradientOptimizer
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class RMSprop(GradientOptimizer):
    def __init__(self, maximize=False, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) -> None:
        super().__init__(1 if maximize else -1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha=alpha
        self.eps=eps
        self.momentum = momentum
        self.centered = centered
        # self.build()

    def build(self, params_like):
        # local buffers
        self.b =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ] # buffer 
        self.sv =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ] # square avg
        self.gv =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ] #grad avg
        self.time = 0 # should be last
        return self

    def call_step(self, params, grads):
        self.time += 1 # should be first


        if self.weight_decay: grads = [ g + self.weight_decay*p for g,p in zip(grads, params) ]

        for i,g in enumerate(grads):
            self.sv[i] = self.alpha*self.sv[i] + (1-self.alpha) * (g**2)
        
        if self.centered:
            for i,g in enumerate(grads):
                self.gv[i] =  self.alpha*self.gv[i] + (1-self.alpha) * g
                self.sv[i] = self.sv[i] - self.gv[i]**2
        
        if self.momentum > 0:
            for i,g in enumerate(grads):
                self.b[i] = self.momentum*self.b[i] + g / (tt.sqrt(self.sv[i])+self.eps)

            grads = self.b
        else:
            grads = [  g/(tt.sqrt(self.sv[i])+self.eps) for i,g in enumerate(grads) ]

        return grads

    def call_with_grad(self, module, grads):
        step_grads = self.call_step(module.values(), grads)
        return {n:p-self.lr*g for g,(n,p) in zip(step_grads, module.items())}
    
    @tt.no_grad()
    def call_no_grad(self, module, grads):
        step_grads = self.call_step(module.values(), grads)
        for g,p in zip(step_grads, module.values()):
            p-=self.lr*g
        return module

