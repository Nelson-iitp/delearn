


import delearn as dl
import torch as tt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dtype=tt.float32


def MLP_PARAMS_DICT(input_size, layer_sizes, output_size, dtype=None, device=None, requires_grad=True ): 
    r"""returns a list of parameters for a Multi-Layer Perceptron Model"""
    if layer_sizes:
        params = \
            {
                # weight  @ 0   shape = (in, out)
                f'w0':tt.zeros(size=(input_size, layer_sizes[0]), dtype=dtype, device=device, requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b0':tt.zeros(size=(1,          layer_sizes[0]), dtype=dtype, device=device, requires_grad=requires_grad), 
            }
        for i in range(len(layer_sizes)-1):
            params.update(
            {
                # weight  @ 0   shape = (in, out)
                f'w{i+1}':tt.zeros(size=(layer_sizes[i], layer_sizes[i+1]), dtype=dtype, device=device, requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b{i+1}':tt.zeros(size=(1,          layer_sizes[i+1]), dtype=dtype, device=device, requires_grad=requires_grad), 
            })
        params.update(
            {
                # weight  @ 0   shape = (in, out)
                f'w{len(layer_sizes)}':tt.zeros(size=(layer_sizes[-1], output_size), dtype=dtype, device=device, requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b{len(layer_sizes)}':tt.zeros(size=(1,          output_size), dtype=dtype, device=device, requires_grad=requires_grad), 
            })
    else:
        params = \
            {
                # weight  @ 0   shape = (in, out)
                f'w':tt.zeros(size=(input_size, output_size),  dtype=dtype, device=device, requires_grad=requires_grad), 

                # bias    @ 1   shape = (1, out)
                f'b':tt.zeros(size=(1,          output_size), dtype=dtype, device=device, requires_grad=requires_grad), 
            }
    return params

parameters_dict = MLP_PARAMS_DICT(
    input_size=1,
    layer_sizes=[64,64,64],
    output_size=1,
    dtype=dtype,
)
print(parameters_dict.keys())

module = dl.Module(parameters_dict)
dl.Modular.rand_parameters(module)
dl.Modular.show_parameters(module)

def forward(m, x): 
    return ( tt.matmul(
        tt.relu( tt.matmul(
        tt.tanh( tt.matmul(
        tt.sigmoid( tt.matmul(
        x
        , m['w0']) + m['b0'] )
        , m['w1']) + m['b1'] )
        , m['w2']) + m['b2'] )
        , m['w3']) + m['b3'] )



# dl.count_parameters(module)
# for i,(n,p) in module: print(i, n, p.shape)
# dl.show_parameters(module, values=True)

tx = tt.linspace(-5, 5, 50, dtype=dtype)
ty = tt.sin(tx)
tds = dl.Task( tx.unsqueeze(-1), ty.unsqueeze(-1), )
print('Training set:', len(tds))
txx, tyy = next(iter(DataLoader(tds, batch_size=len(tds), shuffle=False)))


vx = tt.linspace(-5, 5, 8, dtype=dtype)
vy = tt.sin(vx)
vds = dl.Task( vx.unsqueeze(-1), vy.unsqueeze(-1), )
print('Validation set:',len(vds))
vxx, vyy = next(iter(DataLoader(vds, batch_size=len(vds), shuffle=False)))

plt.figure()
plt.scatter(txx,tyy, color='blue', marker='.', label='train')
plt.scatter(vxx,vyy, color='red', marker='.', label='val')
plt.legend()
plt.show()


lossf = tt.nn.MSELoss()

def lossplot():
    tloss, tpp = dl.Trainer.predict_batch(txx, tyy, forward, module, lossf)
    vloss, vpp = dl.Trainer.predict_batch(vxx, vyy, forward, module, lossf)

    plt.figure()
    plt.scatter(txx,tyy, color='blue', marker='.', label='train')
    plt.scatter(vxx,vyy, color='red', marker='.', label='val')

    plt.scatter(txx,tpp, color='blue', marker='x')
    plt.scatter(vxx,vpp, color='red', marker='x')

    plt.show()
    print(tloss, vloss)

callback = dl.TrainingCallback(val_data=(vxx,vyy), forward=forward, lossf=lossf)

# callback.clear()

optimf = dl.SGD( lr=0.025, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False )

lossplot()
dl.Trainer.train_dataset(
    forward=forward,
    module=module,
    lossf=lossf,
    optimf=optimf,
    dataf = lambda e: DataLoader(tds, batch_size=10, shuffle=True, drop_last=False),
    n=1200,
    batch_mode=False,
    create_graph=False,
    callback=callback,
)
callback.plot_results()
lossplot()