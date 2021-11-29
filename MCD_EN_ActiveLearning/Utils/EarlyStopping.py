import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, round, selected_data, patience=7, verbose=False,
                 delta=0.005, typ='loss', write_all=False, is_ensemble=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            typ (str): can be 'loss', 'accuracy' or 'none', if none, all epocs are written
        """
        self.type = typ
        self.round = round
        self.selected_data = selected_data
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.write_all = write_all
        self.early_stop = False
        self.is_ensemble = is_ensemble
        if typ == 'loss':
            self.best_score = np.Inf
        elif typ == 'accuracy':
            self.best_score = 0
        if typ == 'loss':
         self.delta = 0.005

        elif typ == 'accuracy':
            self.delta = 0.2

    def __call__(self, metric, model, optimizer, pth_to_write):

        score = metric
        '''
        For ensemble models, there is no early stopping.
        '''
        if not self.is_ensemble:
            if self.best_score == 0 or self.best_score == np.Inf:
                self.save_checkpoint(metric, model, optimizer, pth_to_write)
            elif (abs(score-self.best_score) <= self.delta or score > self.best_score) and self.type == 'loss':
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            elif score <= self.best_score - self.delta and self.type == 'accuracy' and not self.write_all:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(metric, model, optimizer, pth_to_write)
                self.counter = 0
        else:
            self.save_checkpoint(metric, model, optimizer, pth_to_write)

    def save_checkpoint(self, metric, model, optimizer, pth_to_write):

        '''
        TODO: need to re-write the following in a better way
        '''
        if not self.is_ensemble:
            '''Saves model when validation loss decrease.'''
            if self.verbose and self.type == 'loss':
                print(f'Validation loss decreased ({self.best_score:.6f} --> {metric:.6f}).  Saving model ...')
            elif self.verbose and self.type == 'accuracy':
                print(f'Accuracy increased ({self.best_score:.6f} --> {metric:.6f}).  Saving model ...')
            state = {'round': self.round, 'metric':metric, 'weights': model.state_dict(), \
                'selected_data':self.selected_data, 'optimizer': optimizer.state_dict()}
            torch.save(state, pth_to_write)
            # update best score
            self.best_score = metric
        else:
            for i,m in enumerate(model):
                    state = {'round': self.round, 'metric':metric, 'weights': m.state_dict(), \
                        'selected_data':self.selected_data, 'optimizer': optimizer[i].state_dict()}
                    torch.save(state, pth_to_write[i])
