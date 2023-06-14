#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`delearn/trainer.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

    'Callback',  'Trainer', 

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
import torch.autograd as ag
from torch.utils.data import Dataset, DataLoader
import datetime
now = datetime.datetime.now
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# NOTE: forward signature is forward(module, inputs)

class Callback(object):
    def __init__(self) -> None: super().__init__()
    def on_train_begin(self, n, optim): pass
    def on_train_end(self, n): pass
    def on_batch_begin(self, batch, x, y): pass
    def on_batch_end(self, batch, batch_loss, module): pass
    def on_epoch_begin(self, epoch): pass
    def on_epoch_end(self, epoch): pass

class Trainer:

    @staticmethod
    @tt.no_grad()
    def predict_batch(forward, module, inputs, labels, lossf):
        preds = forward(module, inputs)
        loss = lossf(preds, labels).item()
        return loss, preds
    
    @staticmethod
    def predict_ds(forward, module, dataset, lossf):
        inputs, labels = next(iter(DataLoader(dataset, len(dataset))))
        return __class__.predict_batch(forward, module, inputs, labels, lossf)

    @staticmethod
    def train_batch(forward, module, inputs, labels, lossf, optimf, create_graph=False):
        r""" [+1] Trains a batch of data (x, y) on model given lossf and optim """
        preds = forward(module, inputs)
        #print(f'Loss:{x.shape=}, {ypred.shape=}, {y.shape=}')
        loss = lossf(preds, labels)
        grads = ag.grad(loss, module.parameters(), create_graph=create_graph)
        module = optimf(module, grads, create_graph)
        lossval = loss.item()
        return lossval, module
    
    @staticmethod
    def train_ds(forward, module, lossf, optimf, dsF, n, batch_mode, batch_size, shuffle=True, callback=None, create_graph=False):  
        r"""[+2] Trains on given ds for given number of epochs, creates a dataloader"""
        # dsF(batch_mode, n, epoch, batch) -> DataSet
        if callback is None: callback = Callback()
        #train_losses = []
        

        callback.on_train_begin(n, optimf)  

        if batch_mode:
            epoch=-1
            di = iter([])
            for batch in range(n): 
                try:
                    x,y = next(di)
                except StopIteration:
                    if epoch>-1: callback.on_epoch_end(epoch)
                    
                    di = iter(DataLoader(dsF(batch_mode, n, batch), batch_size, shuffle))
                    #train_losses.append([])
                    epoch+=1
                    callback.on_epoch_begin(epoch)
                    x,y = next(di)
                callback.on_batch_begin(batch, x, y)
                batch_loss, module = __class__.train_batch(forward, module, x, y, lossf, optimf, create_graph=create_graph)
                #train_losses[-1].append(batch_loss)
                callback.on_batch_end(batch, batch_loss, module) 
        else:
            for epoch in range(n): 
                callback.on_epoch_begin(epoch)
                #train_losses.append([])
                for batch,(x,y) in enumerate(DataLoader(dsF(batch_mode, n, epoch), batch_size, shuffle), 0):
                    callback.on_batch_begin(batch, x, y)
                    batch_loss, module = __class__.train_batch(forward, module, x, y, lossf, optimf, create_graph=create_graph)
                    #train_losses[-1].append(batch_loss)
                    callback.on_batch_end(batch, batch_loss, module) 
                    
                callback.on_epoch_end(epoch) 

        callback.on_train_end(n)  


        return #train_losses
        # ============================================================================
