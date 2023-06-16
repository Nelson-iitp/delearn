__doc__=r"""
:py:mod:`delearn/layer/__init__.py`
"""

class GradientOptimizer:
    def __init__(self, sign) -> None:
        self.sign = sign# -1 for gradient descent = minimize loss
    
    def __call__(self, module, grads, create_graph):
        return self.call_with_grad(module, grads) if create_graph else self.call_no_grad(module, grads)
    



from .sgd import *

from .rmsp import *

from .adam import *