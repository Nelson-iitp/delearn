__doc__=r"""
:py:mod:`delearn/layer/linear.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    
    'Linear', 'MLP', 
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Linear:

    @staticmethod
    def parameters(input_size, output_size, dtype=None, device=None, requires_grad=True ): 
        return \
        {
            f'w':tt.zeros(size=(input_size, output_size),  dtype=dtype, device=device, requires_grad=requires_grad),  # weight  @ 0   shape = (in, out)
            f'b':tt.zeros(size=(1,          output_size),  dtype=dtype, device=device, requires_grad=requires_grad),  # bias    @ 1   shape = (1, out)
        }

    @staticmethod
    def forward(module, input, actf):  return actf(tt.matmul(input, module['w']) + module['b'] )



class MLP:

    @staticmethod
    def parameters(input_size, layer_sizes, output_size, dtype=None, device=None, requires_grad=True ): 
        params = {}
        layer_sizes = [input_size] + layer_sizes + [output_size]
        for i in range(len(layer_sizes)-1):
            for k,v in Linear.parameters(layer_sizes[i], layer_sizes[i+1], dtype, device, requires_grad).items(): params[f'{k}.{i}'] = v
        return params
    
    @staticmethod
    def forward(module, input, actf):  
        output = input
        for i in range(0, int(len(module)/2)): # since module is an MLP it has even number of parameters (w,b) for each layer
            output = actf[i](tt.matmul(output, module[f'w.{i}']) + module[f'b.{i}'])
        return output