# %%
import delearn as dl
import torch as tt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dtype=tt.float32

# %%
module = dl.nn.MLP.parameters(
    input_size=1,
    layer_sizes=[64,64,64],
    output_size=1,
    dtype=dtype, 
    device=None, 
    requires_grad=True,
)
print(f'module: {module.keys()}')

dl.Modular.rand(module)
dl.Modular.show(module, False)

activations_list = (tt.sigmoid, tt.tanh, tt.relu, dl.Identity)
def forward(m, x): return dl.nn.MLP.forward(m,x,activations_list)
dl.Modular.show(module, True)

#%%
tx = tt.linspace(-5, 5, 50, dtype=dtype)
ty = tt.sin(tx)
tds = dl.Task( tx.unsqueeze(-1), ty.unsqueeze(-1), )(batch_size=10, shuffle=True, drop_last=False) # <--- traing loader
print('Training set:', len(tds))


vx = tt.linspace(-5, 5, 8, dtype=dtype)
vy = tt.sin(vx)
vds = dl.Task( vx.unsqueeze(-1), vy.unsqueeze(-1), )
print('Validation set:', len(tds))

plt.figure()
plt.scatter(tds.x,tds.y, color='blue', marker='.', label='train')
plt.scatter(vds.x,vds.y, color='red', marker='.', label='val')
plt.legend()
plt.show()

# %%
lossf = tt.nn.MSELoss()

def valplot(m):
    txx, tyy = tds.x, tds.y
    vxx, vyy =  vds.x, vds.y
    tloss, tpp = dl.Trainer.predict_batch(txx, tyy, forward, m, lossf)
    vloss, vpp = dl.Trainer.predict_batch(vxx, vyy, forward, m, lossf)

    plt.figure()
    plt.scatter(txx,tyy, color='blue', marker='.', label='train')
    plt.scatter(vxx,vyy, color='red', marker='.', label='val')

    plt.scatter(txx,tpp, color='blue', marker='x')
    plt.scatter(vxx,vpp, color='red', marker='x')

    plt.show()
    print(tloss)
    print(vloss)

valplot(module)

callback = dl.TrainingCallback(val_data=(vds.x,vds.y), forward=forward, lossf=lossf)

# callback.clear()



# %%
#optimf = dl.optim.SGD( lr=0.025, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False ).build()

#optimf = dl.oo.Adam( ).build(module.values())

optimf = dl.oo.RMSprop(lr=0.001, alpha=0.99, weight_decay=0.0, momentum=0.12, centered=False).build(module.values())

# %%

dl.Trainer.train_dataset(
    forward=forward,
    module=module,
    lossf=lossf,
    optimf=optimf,
    dataf = lambda e: tds,
    n=150,
    batch_mode=False,
    create_graph=False,
    callback=callback,
)
callback.plot_results()
valplot(module)
valplot(callback.module)


