import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.005, typ='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.type = typ
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if typ == 'loss':
            self.val_score = np.Inf
        elif typ == 'accuracy':
            self.val_score = 0
        self.delta = delta

    def __call__(self, val_loss, model, pth_to_write):

        if self.type == 'loss':
            score = -val_loss
        elif self.type == 'accuracy':
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth_to_write)
        elif score <= self.best_score + self.delta and self.type == 'loss':
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score <= self.best_score - self.delta and self.type == 'accuracy':
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth_to_write)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, pth_to_write):
        '''Saves model when validation loss decrease.'''
        if self.verbose and self.type == 'loss':
            print(f'Validation loss decreased ({self.val_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        elif self.verbose and self.type == 'accuracy':
            print(f'Accuracy increased ({self.val_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), pth_to_write)
        self.val_score = val_loss
