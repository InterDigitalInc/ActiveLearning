import torch
import GPUtil

if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    custom_select_gpu = input("Custom select GPU number? 0 for 'no' and 1 for 'yes': ")
    if int(custom_select_gpu):
        AVAILABLE_GPU = int(input("Enter gpu number:"))
    else:
        # check which gpu is free and assign that gpu
        AVAILABLE_GPU = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, \
                                            maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])[0]

    torch.cuda.set_device(AVAILABLE_GPU)
    print('Program will be executed on GPU:{}'.format(AVAILABLE_GPU))
    DEVICE = torch.device('cuda:' + str(AVAILABLE_GPU))

import argparse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from Models import LenetDropout300_100_Regress
import os
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from os.path import expanduser
from torch.distributions import Categorical
import shutil
import sys
import torch.nn.functional as F
from torchsummary import summary
from Utils import Config
from Utils import EarlyStopping
import time
from Utils import LogSummary
from sklearn.metrics import r2_score, mean_squared_error

config = Config()

HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'
torch.manual_seed(33)
np.random.seed(33)

if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')
if not os.path.isdir('results'):
    os.makedirs('results')

class RunModel:

    def __init__(self, m_name,d_nam, train_batch_size, test_batch_size, active_batch_size, nn_instances,
                 epoch, rounds,seed_sample, mode,active_learn,topk, retrain, acquisition,
                 optimizer, lr, resume_round):

        self.m_name = m_name
        self.topk = topk
        self.activ_learn = active_learn
        self.d_name = d_nam
        self.epochs = epoch
        self.rounds = rounds
        # initial sample size to train NN
        self.isample = seed_sample
        self.criterion = nn.MSELoss()
        self.tr_b_sz = train_batch_size
        self.tst_b_sz = test_batch_size
        self.activ_lrn_b_sz = active_batch_size
        self.nn_instances = nn_instances
        self.n_samples = 20
        self.d_rate = 0.2
        self.l_rate = lr
        self.acq = acquisition
        self.retrain = retrain
        self.optim = optimizer

        self.results_dir = ''
        if self.activ_learn:
            self.acc_fil = open('tmp_active_accuracy.txt','w')
        else:
            self.acc_fil = open('tmp_random_accuracy.txt','w')
        self.n_classes, self.i_channel, self.i_dim, self.train_d, self.test_d = config.get_data(d_nam)

        self.test_loader = DataLoader(self.test_d, batch_size=self.tst_b_sz, shuffle=True, num_workers=2)
        self.selected_data = set([])
        self.unexplored = set(range(len(self.train_d)))
        self.test_len = len(self.test_d)
        self.train_len = len(self.train_d)

        if resume_round:
            self.InitModel(load_weights=True, round=resume_round)
        else:
            self.InitModel()
            self.init_optimizer(self.l_rate)
        t_param = sum(p.numel() for p in self.model.parameters())

    def init_optimizer(self,l_rate):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=l_rate, amsgrad=True)
    def __load_pre_train_model(self, round):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel(round)
        # load the weights
        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
            # update selected data
            self.selected_data = state['selected_data']
            self.model.load_state_dict(state['weights'])
            self.init_optimizer(self.l_rate)
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            state = torch.load(self.train_weight_path)
            self.selected_data = state['selected_data']
            self.model.load_state_dict(state['weights'])
            self.init_optimizer(self.l_rate)
            self.optimizer.load_state_dict(state['optimizer'])

    def InitModel(self, load_weights=False, round=None):
        if self.m_name == 'lenet300-100':
            self.model = LenetDropout300_100_Regress(self.i_dim, self.n_classes).to(DEVICE)

        print(summary(self.model, (self.i_channel,self.i_dim,self.i_dim)))
        if load_weights and round:
            self.__load_pre_train_model(round)



    def __getVariationRatio(self, outputs):
        standard_dev = torch.var(outputs, dim=0).view(-1)
        return standard_dev

    def __getEntropy(self, outputs):
        predicts = []
        ensemble_prob = outputs.mean(0)
        entropy = Categorical(probs=ensemble_prob).entropy()
        return entropy

    def ActiveSubSelectData(self, round):

        if round == 1:
            # add the new data points to the selected list of data
            self.selected_data = set(range(self.isample))
            self.unexplored = self.unexplored.difference(self.selected_data)
        else:
            all_data = DataLoader(self.train_d, batch_size=self.activ_lrn_b_sz, num_workers=2)
            # number of NN  instantiations
            T = self.nn_instances
            correct = 0
            metrics = []
            self.model.eval()
            self.model.drop_train = True
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(all_data)):
                    # replicate the samples to create T instantiations of NN
                    b_sz = inputs.shape[0]
                    inputs = inputs.repeat(self.nn_instances, 1)
                    targets = targets.repeat(self.nn_instances)
                    # note: we don't use targets variable for active learning
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    ensemble_outputs = self.model(inputs)
                    ensemble_outputs = ensemble_outputs.reshape(self.nn_instances, b_sz, self.n_classes)
                    if self.acq == 'entropy':
                        metric = self.__getEntropy(ensemble_outputs)
                        metrics.append(metric)
                    if self.acq == 'variation-ratio':
                        metric = self.__getVariationRatio(ensemble_outputs)
                        metrics.append(metric)
                metrics = torch.cat(metrics)
                new_indices = metrics.argsort(descending=True).tolist()
                # remove indices that have already been explored
                new_indices = [n for n in new_indices if n not in self.selected_data]
                '''
                get the top_k data points with most uncertainty from the new list of data points and add
                this to the explored list.
                '''
                self.selected_data = self.selected_data.union(set(new_indices[:self.topk]))
                self.unexplored = self.unexplored.difference(self.selected_data)
                # add the newly selected data points to the explored list

    # this method simply performs random sampling of data instead of active learning-based sampling
    def RandomSubSelectData(self, round):

        if round == 1:
            # add the new data points to the selected list of data
            self.selected_data = set(range(self.isample))
            self.unexplored = self.unexplored.difference(self.selected_data)
        else:
            min_ind = np.random.choice(list(self.unexplored), self.topk)
            self.selected_data = self.selected_data.union(min_ind)
            self.unexplored = self.unexplored.difference(self.selected_data)

    '''
    This method simply retrieves the data points that are already selected by the model during a training phase.
    This method is only called when we have to retrain a model from scratch with pre-selected data points.
    '''
    def get_preselected_points(self, round):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel(round)
        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)
        # get the selected data
        self.selected_data = state['selected_data']

    def get_validation_data(self, is_valid):

        if not is_valid:
            train_sampler = SubsetRandomSampler(list(self.selected_data))
            self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=1)
            return

        indices = list(self.unexplored)
        np.random.shuffle(indices)
        split = int(np.floor(0.1 * len(indices)))
        valid_indx = np.random.choice(indices,split)
        train_sampler = SubsetRandomSampler(list(self.selected_data))
        valid_sampler = SubsetRandomSampler(valid_indx)
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=5)
        self.valid_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=valid_sampler, num_workers=5)

    # training using the proposed active learning approach
    def Train(self, round, epoch, is_valid):

        t_correct, v_correct = 0, 0
        t_total, v_total = 0, 0
        t_r2_scores = []
        # use 10% of training data as validation every round by randomly shuffling the unexplored indices of training data
        if epoch == 0:
            self.get_validation_data(is_valid)
        self.model.train()
        train_loss, valid_loss = [], []

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).view(-1)
            loss = self.criterion(outputs,targets)
            t_total += targets.size(0)
            t_r2_scores.append(r2_score(outputs.detach().cpu().numpy(), targets.cpu().numpy()))
            train_loss.append(loss.item())
            loss.backward()
            # parameter update
            self.optimizer.step()

        if is_valid:
            # override the results with the validation dataset
            self.model.eval()
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs,targets)
                _, predicted = F.softmax(outputs,dim=1).max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()
                valid_loss.append(loss.item())

            avg_valid_loss = np.average(valid_loss)
            avg_train_loss = np.average(train_loss)
            accuracy = (100. * v_correct / v_total)
            print('Epoch:{}, TrainLoss:{}, ValidationLoss:{}, ValidationAccuracy:{}'. \
                format(epoch, avg_train_loss, avg_valid_loss, accuracy))
            return avg_valid_loss,  accuracy
        else:

            avg_train_loss = np.average(train_loss)
            avg_r2_score = np.average(t_r2_scores)
            print('Epoch:{}, TrainLoss:{}, TrainAvgR2Score:{:.3f}'.format(epoch, avg_train_loss, avg_r2_score))

            return avg_train_loss, avg_r2_score


    def Test(self, round):

        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)
        self.model.load_state_dict(state['weights'])
        self.model.eval()
        predictions = []
        actual = []
        mse_scores = []
        self.model.drop_train = False
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                X, Y = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(X)
                predictions.append(outputs.detach().cpu()[:,0].numpy())
                actual.append(targets.numpy())
                mse_scores.append(self.criterion(outputs.view(-1), Y).item())
        predictions = np.concatenate(predictions, axis=0)
        actual = np.concatenate(actual, axis=0)
        df = pd.DataFrame(data = {"actual":actual, "prediction":predictions})
        df.loc["r2"] = r2_score(df.prediction, df.actual)
        df.loc["mse"] = mean_squared_error(df.prediction, df.actual)
        print('Non-Ensemble Test MSE:{:.3f}, TestR2:{:.3f}'.format(df.loc["mse"][0], df.loc["r2"][0]))
        if round == 1: self.__createFolder()
        if self.activ_learn:
            df.to_csv(self.results_dir + '/accuracy_active_round' + str(round) + '.csv', index=True)
        else:
            df.to_csv(self.results_dir + '/accuracy_random_round' + str(round) + '.csv', index=True)
        return df.loc["r2"][0]

    def __createFolder(self):
        pth = 'results/McDropout/'
        if self.activ_learn:
            name = self.m_name+'_'+self.d_name+'_isample'+str(self.isample)+'_e'+str(self.epochs)+ "_activInstances" + str(self.nn_instances) \
                   +'_r'+ str(self.rounds)+'_ac'+str(self.activ_learn) + '_optim-' + self.optim +'_top-k'+ str(self.topk) +\
                   '_b'+str(self.tr_b_sz) +'_rtAfter'+str(self.retrain)+'_'+self.acq
        else:
            name = self.m_name+'_'+self.d_name+'_isample'+str(self.isample)+'_e'+ \
                   str(self.epochs)+'_r'+str(self.rounds)+'_ac'+str(self.activ_learn) + \
                       '_optim-' + self.optim + '_top-k'+ str(self.topk) + '_b'+ str(self.tr_b_sz) + '_rtAfter'+ str(self.retrain)

        if not os.path.isdir(pth+name):
            os.mkdir(pth+name)
        else:
            print('folder:{} already exists. Want to override?***NOTE: all contents will be overwritten****'.format(pth+name))
            inp = input('Enter y/n')
            if inp == 'y' or inp == 'Y':
                shutil.rmtree(pth+name)
                print('done overwriting the files')
                os.mkdir(pth+name)
            else:
                sys.exit('quitting')
        self.results_dir = pth+name


    def getTrainedmodel(self,rounds):
        # path to write trained weights
        if self.activ_learn:
            self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name + \
                '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k'+ str(self.topk) +'_rtAfter'+str(self.retrain) + \
                '-' + 'b' + str(self.tr_b_sz) +'_optim-'+ self.optim +'_active'+'_'+self.acq+'.pkl'
        else:
            self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name + '-isample'+ str(self.isample) + \
                '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k'+ str(self.topk) +'_rtAfter'+str(self.retrain) + \
                '-' + 'b' + str(self.tr_b_sz) +'_optim-'+ self.optim + '_random'+'_'+self.acq+'.pkl'
        return (self.model, self.train_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP models that use regression datasets')
    parser.add_argument('-d', '--dataset', help='dataset type, options 1.boston_housing 2. cali_housing',
                        default='boston_housing')
    parser.add_argument('-m', '--model', help='model name 1.lenet300-100', default='lenet300-100')
    parser.add_argument('-mo', '--mode', help='1.train or 2.test', default='test')
    parser.add_argument('-al', '--activelearn', help='1, 2 or 3', default=1, type=int)
    parser.add_argument('-trb', '--train_batch_size', help='batch size', default=8, type=int)
    parser.add_argument('-tsb', '--test_batch_size', help='batch size', default=512, type=int)
    parser.add_argument('-alb', '--active_batch_size', help='batch size', default=512, type=int)
    parser.add_argument('-nni', '--nn_instances', help='number of neural network instances to perform active learning', default=25, type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('-r', '--rounds', help='number of rounds', default=41, type=int)
    parser.add_argument('-rt', '--retrain', help='integer value between 1-# rounds. If  1 -> at each round we retrain from scratch',\
         default=0, type=int)
    parser.add_argument('-ss', '--seedsample', help='number of seed samples to begin training', default=50, type=int)
    parser.add_argument('-tk', '--topk', help='number of samples to add during each round', default=5, type=int)
    parser.add_argument('-af', '--acquisitionFunction', help='1. entropy, 2. variation-ratio', default='variation-ratio')
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-v', '--is_valid', help='whether to use validation or not', default=0, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate to start with', default=0.001, type=float)
    parser.add_argument('-ra', '--resume_after', help='resume after a round number, default is 0', \
         default=0, type=int)
    parser.add_argument('-twp', '--trained_weight_path', help='location of the trained weights (applied only for testing)', \
         default='trained_weights')
    args = parser.parse_args()
    start_epoch, subseq_epoch, patience = config.get_init_params_v2_beta(args.dataset, args.model, args.retrain)
    run_model = RunModel(args.model, args.dataset, args.train_batch_size, args.test_batch_size, args.active_batch_size, args.nn_instances,
                         args.epochs, args.rounds, args.seedsample, args.mode, args.activelearn, args.topk, args.retrain,
                         args.acquisitionFunction, args.optimizer, args.lr, args.resume_after)

    lr = args.lr
    start = 1
    epoch = args.epochs
    if args.resume_after and args.mode == 'train':
        start = args.resume_after + 1
    if start == 1:
        epoch = start_epoch
    else:
        epoch = subseq_epoch
    if args.mode == 'train':
        write_summary = LogSummary(
            name=args.model + '_' + args.dataset + '_al' + str(args.activelearn) + "_nni" +
                 str(args.nn_instances) + '_af-' + args.acquisitionFunction + '_retrain' + str(args.retrain) + '_nni-' + str(args.nn_instances))

        for r in range(start, args.rounds):

            if args.activelearn == 1:
                run_model.ActiveSubSelectData(r)
            elif args.activelearn == 2:
                run_model.get_preselected_points(r)
            else:
                run_model.RandomSubSelectData(r)
            # initialize the early stopping criteria for each round
            early_stopping = EarlyStopping(round=r, selected_data=run_model.selected_data, \
                                           patience=patience, verbose=True, typ='loss')
            if r != 1 and args.retrain:
                run_model.InitModel(args.model)
                run_model.init_optimizer(lr)

            for e in range(epoch):
                start = time.time()
                mse_loss, r2 = run_model.Train(r, e, is_valid=args.is_valid)
                print('round:{}, lr:{} time-taken :{:.2f} seconds'.format(r, lr, time.time() - start))
                model, path_to_write = run_model.getTrainedmodel(r)
                early_stopping(mse_loss, model, run_model.optimizer, path_to_write)
                if early_stopping.early_stop:
                    break

            test_r2 = run_model.Test(r)
            # acc = run_model.Test()
            write_summary.write_final_r2(test_r2, r)
            epoch = subseq_epoch

    else:
        for r in range(start, args.rounds):
            print('performing ensemble testing on dataset:{}, model:{}, batch size:{}, epochs:{}, round:{}'
                  .format(args.dataset, args.model, args.test_batch_size, args.epochs, r))
            run_model.getTrainedmodel(r)
            run_model.Test(r)
