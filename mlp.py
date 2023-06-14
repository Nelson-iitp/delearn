


import torch as tt
from torch.nn import Parameter



def MLP_PARAMS_DICT(input_size, layer_sizes, output_size, dtype=None, device=None, requires_grad=True ): 
    r"""returns a list of parameters for a Multi-Layer Perceptron Model"""
    if layer_sizes:
        params = \
            {
                # weight  @ 0   shape = (in, out)
                f'w0':Parameter(tt.zeros(size=(input_size, layer_sizes[0]), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b0':Parameter(tt.zeros(size=(1,          layer_sizes[0]), dtype=dtype, device=device), requires_grad=requires_grad), 
            }
        for i in range(len(layer_sizes)-1):
            params.update(
            {
                # weight  @ 0   shape = (in, out)
                f'w{i+1}':Parameter(tt.zeros(size=(layer_sizes[i], layer_sizes[i+1]), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b{i+1}':Parameter(tt.zeros(size=(1,          layer_sizes[i+1]), dtype=dtype, device=device), requires_grad=requires_grad), 
            })
        params.update(
            {
                # weight  @ 0   shape = (in, out)
                f'w{len(layer_sizes)}':Parameter(tt.zeros(size=(layer_sizes[-1], output_size), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b{len(layer_sizes)}':Parameter(tt.zeros(size=(1,          output_size), dtype=dtype, device=device), requires_grad=requires_grad), 
            })
    else:
        params = \
            {
                # weight  @ 0   shape = (in, out)
                f'w':Parameter(tt.zeros(size=(input_size, output_size),  dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b':Parameter(tt.zeros(size=(1,          output_size), dtype=dtype, device=device), requires_grad=requires_grad), 
            }
    return params



def MLP_PARAMS_LIST(input_size, layer_sizes, output_size, dtype=None, device=None, requires_grad=True ): 
    r"""returns a list of parameters for a Multi-Layer Perceptron Model"""
    if layer_sizes:
        params = \
            [
                # weight  @ 0   shape = (in, out)
                Parameter(tt.zeros(size=(input_size, layer_sizes[0]), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                Parameter(tt.zeros(size=(1,          layer_sizes[0]), dtype=dtype, device=device), requires_grad=requires_grad), 
            ]
        for i in range(len(layer_sizes)-1):
            params.extend(
            [
                # weight  @ 0   shape = (in, out)
                Parameter(tt.zeros(size=(layer_sizes[i], layer_sizes[i+1]), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                Parameter(tt.zeros(size=(1,          layer_sizes[i+1]), dtype=dtype, device=device), requires_grad=requires_grad), 
            ])
        params.extend(
            [
                # weight  @ 0   shape = (in, out)
                Parameter(tt.zeros(size=(layer_sizes[-1], output_size), dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                Parameter(tt.zeros(size=(1,          output_size), dtype=dtype, device=device), requires_grad=requires_grad), 
            ])
    else:
        params = \
            [
                # weight  @ 0   shape = (in, out)
                Parameter(tt.zeros(size=(input_size, output_size),  dtype=dtype, device=device), requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                Parameter(tt.zeros(size=(1,          output_size), dtype=dtype, device=device), requires_grad=requires_grad), 
            ]
    return params
