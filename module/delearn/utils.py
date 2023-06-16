#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`delearn/utils.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [

    'QuantiyMonitor',

]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class QuantiyMonitor:
    """ Monitors a quantity overtime to check if it improves (decreases) after a given patience. 
    Quantity is checked on each call to :func:`~delearn.utils.QuantiyMonitor.check`. 
    The ``__call__`` methods implements the ``check`` method. Can be used to monitor loss for early stopping.
    
    :param name: name of the quantity to be monitored
    :param patience: number of calls before the monitor decides to stop
    :param delta: the amount by which the monitored quantity should decrease to consider an improvement
    """

    def __init__(self, name:str, patience:int, delta:float) -> None:
        r"""
        :param name: name of the quantity to be monitored
        :param patience: number of calls before the monitor decides to stop
        :param delta: the amount by which the monitored quantity should decrease to consider an improvement
        """
        assert(patience>0) # patience should be positive
        assert(delta>0) # delta should be positive
        self.patience, self.delta = patience, delta
        self.name = name
        self.reset()
    
    def reset(self, initial=None):
        r""" Resets the monitor's state and starts at a given `initial` value """
        self.last = (tt.inf if initial is None else initial)
        self.best = self.last
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, current, epoch=-1, verbose=False) -> bool:
        return self.check(current, epoch, verbose)
        
    def check(self, current, epoch=-1, verbose=False) -> bool:
        r""" Calls the monitor to check the current value of monitored quality
        
        :param current: the current value of quantity
        :param epoch:   optional, the current epoch (used only for verbose)
        :param verbose: if `True`, prints monitor status when it changes

        :returns: `True` if the quanity has stopped improving, `False` otherwise.
        """
        self.last = current
        if self.best == tt.inf: 
            self.best=self.last
            self.best_epoch = epoch
            if verbose: print(f'|~|\t{self.name} Set to [{self.best}] on epoch {epoch}')
        else:
            delta = self.best - current # `loss` has decreased if `delta` is positive
            if delta > self.delta:
                # loss decresed more than self.delta
                if verbose: print(f'|~|\t{self.name} Decreased by [{delta}] on epoch {epoch}') # {self.best} --> {current}, 
                self.best = current
                self.best_epoch = epoch
                self.counter = 0
            else:
                # loss didnt decresed more than self.delta
                self.counter += 1
                if self.counter >= self.patience: 
                    if verbose: print(f'|~| Stopping on {self.name} = [{current}] @ epoch {epoch} | best value = [{self.best}] @ epoch {self.best_epoch}')
                    return True # end of patience
        return False


