
__doc__=r"""
:py:mod:`delearn/optim/adam.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

   'Adam',
]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
from . import GradientOptimizer
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Adam(GradientOptimizer):
    def __init__(self, maximize=False, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False) -> None:
        super().__init__(1 if maximize else -1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas=betas
        self.eps=eps
        self.amsgrad = amsgrad
        # self.build()

    def build(self, params_like):
        # local buffers
        self.m =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ]
        self.v =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ]
        self.mh =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ]
        self.vh =  [ tt.zeros_like(p, requires_grad=False) for p in params_like ]
        self.vmax = [ tt.zeros_like(p, requires_grad=False) for p in params_like ]
        self.b1, self.b2 = self.betas
        self.time = 0 # should be last
        return self

    def call_step(self, params, grads):
        self.time += 1 # should be first
        if self.sign>0: grads = [ -g for g in grads ]


        if self.weight_decay: grads = [ g + self.weight_decay*p for g,p in zip(grads, params) ]

        for i,g in enumerate(grads):
            self.m[i] = self.b1*self.m[i] + (1-self.b1) * g
            self.v[i] = self.b2*self.v[i] + (1-self.b2) * (g**2)
            self.mh[i] = self.m[i]/(1-self.b1**self.time)
            self.vh[i] = self.v[i]/(1-self.b2**self.time)

        if self.amsgrad:
            grads = [self.mh[i]/(tt.sqrt(self.vmax[i]) + self.eps)  for i,g in enumerate(grads)]
        else:
            grads = [self.mh[i]/(tt.sqrt(self.vh[i]) + self.eps)  for i,g in enumerate(grads)]

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

