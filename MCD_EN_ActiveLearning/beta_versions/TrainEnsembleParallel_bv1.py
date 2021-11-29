'''
This is a work in progress
'''
import torch
import GPUtil

if torch.cuda.is_available():
    # check which gpu is free and assign that gpu
    AVAILABLE_GPU = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, \
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])[0]
    torch.cuda.set_device(AVAILABLE_GPU)
    print('Program will be executed on GPU:{}'.format(AVAILABLE_GPU))
    DEVICE = torch.device('cuda:' + str(AVAILABLE_GPU))
else:
    DEVICE = torch.device('cpu')
import argparse

from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from Models import Lenet300_100
from Models import Lenet5, Alexnet, AlexnetLight, VGG
from Models import Densenet_100, DenseNet121
import os
import numpy as np
import copy
from Utils import LogSummary
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from os.path import expanduser
from torch.distributions import Categorical
import shutil
import sys
import torch.nn.functional as F
from torchsummary import summary
from Utils import Config, ModelParallel
from Utils import EarlyStopping
import time
if not os.path.isdir("trained_weights/ensemble"):
    os.mkdir("trained_weights/ensemble")
config = Config()
HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'
np.random.seed(33)

if not os.path.isdir('results'):
    os.makedirs('results')
if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')
if not os.path.isdir('results/McDropout/ensemble'):
    os.makedirs('results/McDropout/ensemble')


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.m = len(models)
        self.module = nn.ModuleList(models)

    def forward(self, input):
        return [self.module[i](input) for i in range(self.m)]

class RunModel:

    def __init__(self, m_name, d_nam, train_batch_size, test_batch_size, active_batch_size, nn_instances,
                 epoch, rounds, seed_sample, mode, active_learn, topk, retrain, acquisition,
                 optimizer, lr, resume_round):

        self.m_name = m_name
        self.topk = topk
        self.activ_learn = active_learn
        self.d_name = d_nam
        self.epochs = epoch
        self.rounds = rounds
        # initial sample size to train NN
        self.isample = seed_sample
        self.tr_b_sz = train_batch_size
        self.tst_b_sz = test_batch_size
        self.activ_lrn_b_sz = active_batch_size
        self.nn_instances = nn_instances
        self.n_samples = 20
        self.d_rate = 0.2
        self.l_rate = lr
        self.acq = acquisition
        self.retrain = retrain
        self.opt = optimizer

        self.results_dir = ''
        if self.activ_learn:
            self.acc_fil = open('tmp_active_accuracy.txt', 'w')
        else:
            self.acc_fil = open('tmp_random_accuracy.txt', 'w')
        self.n_classes, self.i_channel, self.i_dim, self.train_d, self.test_d = config.get_data(d_nam)

        self.test_loader = DataLoader(self.test_d, batch_size=self.tst_b_sz, shuffle=True, num_workers=2)
        self.selected_data = set([])
        self.unexplored = set(range(len(self.train_d)))
        self.test_len = len(self.test_d)
        self.train_len = len(self.train_d)
        # list to hold the ensemble of models
        self.models = []
        self.optimizers = []
        # seed value to replicate
        self.seeds = [8,21,33,145,1002]
        # self.seeds = [8]
        # not implemented

        for s in self.seeds:
            torch.manual_seed(s)
            m = self.InitModel()
            op = self.init_optimizer(self.l_rate, m)
            self.models.append(m)
            self.optimizers.append(op)
        if resume_round:
            self.__load_pre_train_model(resume_round)
        # parallelize the ensemble models
        self.model = Ensemble(self.models)
        gpu_list = [AVAILABLE_GPU] * len(self.models)
        self.model = ModelParallel(self.model, device_ids=gpu_list, output_device=0)
        t_param = sum(p.numel() for p in m.parameters())
        torch.manual_seed(33)

    def init_optimizer(self, l_rate, model):
        if self.opt == 'SGD':
            optim = torch.optim.SGD(model.parameters(), lr=l_rate, momentum=0.9)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=l_rate, amsgrad=True)
        return optim

    def __load_pre_train_model(self, round):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel(round)
        for i,model in enumerate(self.models):
            # load the weights
            if DEVICE.type == 'cpu':
                state = torch.load(self.train_weight_path[i], map_location=torch.device('cpu'))
                # update selected data
                self.selected_data = state['selected_data']
                self.model.load_state_dict(state['weights'])
                self.optimizers[i].load_state_dict(state['optimizer'])
            else:
                state = torch.load(self.train_weight_path[i])
                self.selected_data = state['selected_data']
                model.load_state_dict(state['weights'])
                self.optimizers[i].load_state_dict(state['optimizer'])

    def InitModel(self, load_weights=False, round=None):
        if self.m_name == 'lenet300-100':
            model = Lenet300_100(self.i_dim * self.i_dim, self.n_classes).to(DEVICE)
        elif self.m_name == 'lenet5':
            model = Lenet5(self.n_classes, self.i_channel).to(DEVICE)
        elif self.m_name == 'alexnet':
            model = Alexnet(self.n_classes, self.i_channel).to(DEVICE)
        elif self.m_name == 'alexnet-light' and self.d_name == 'cifar10':
            model = AlexnetLight(self.n_classes, self.i_channel).to(DEVICE)
        elif self.m_name == 'VGG' and self.d_name == 'cifar10':
            model = VGG(self.n_classes, self.i_channel, 'VGG19').to(DEVICE)
        elif self.m_name == 'VGG' and self.d_name == 'cifar100':
            model = VGG(self.n_classes, self.i_channel, 'VGG19').to(DEVICE)
        elif self.m_name == 'densenet':
            model = DenseNet121(self.n_classes).to(DEVICE)
        print(summary(model, (self.i_channel, self.i_dim, self.i_dim)))

        return model

    def __getVariationRatio(self, outputs):
        predicts = []
        n_instances = len(outputs)
        # ToDO: should avoid iteration and need to peform these operations directly on the tensor
        for out in outputs:
            out = out.squeeze(dim=0)
            _, predicted = out.max(1)
            predicts.append(predicted.unsqueeze(dim=0))
        predicts = torch.cat(predicts, dim=0)
        mods, locs = torch.mode(predicts, dim=0)
        mode_count = predicts.eq(mods).sum(0)
        variation_ratio = 1 - torch.div(mode_count.double(), n_instances)
        return variation_ratio

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
            T = len(self.models)
            correct = 0
            m_num = 0
            metrics = []
            ensemble_outputs = []
            # iterate thru every model and get the predicted outputs
            for model in self.models:
                model.eval()
                model.drop_train = True
                outputs = []
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(tqdm(all_data, desc="ActiveLearn Selection for Model:{}".format(m_num))):
                        #
                        b_sz = inputs.shape[0]
                        inputs = inputs.view(-1, self.i_channel, self.i_dim, self.i_dim)
                        # note: we don't use targets variable for active learning
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        # op = torch.unsqueeze(F.softmax(model(inputs), dim=1), 0)
                        op = F.softmax(model(inputs), dim=1)
                        outputs.append(op)
                # add new dimension for every model
                outputs = torch.unsqueeze(torch.cat(outputs),0)
                ensemble_outputs.append(outputs)
                m_num += 1
            ensemble_outputs = torch.cat(ensemble_outputs)
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
        valid_indx = np.random.choice(indices, split)
        train_sampler = SubsetRandomSampler(list(self.selected_data))
        valid_sampler = SubsetRandomSampler(valid_indx)
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=5)
        self.valid_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=valid_sampler, num_workers=5)

    # training using the proposed active learning approach
    def Train(self, epoch, is_valid):

        criterion = nn.CrossEntropyLoss()
        # list to hold loss and accuracies for each model
        m_train_loss = []
        m_valid_loss = []
        m_accuracies = []
        # use 10% of training data as validation every round by randomly shuffling the unexplored indices of training data
        if epoch == 0:
            self.get_validation_data(is_valid)

        train_loss, valid_loss = [], []
        t_correct, v_correct = 0, 0
        t_total, v_total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            for optim in self.optimizers:
                optim.zero_grad()
            outputs = self.model(inputs)
            losses = [criterion(o, targets) for o in outputs]
            # loss = criterion(outputs, targets)
            ensemble_predictions = [F.softmax(o, dim=1).max(1) for o in outputs]
            for i,vals in enumerate(ensemble_predictions):
                optim = self.optimizers[i]
                loss = losses[i]
                _, predicted = vals
                t_total += targets.size(0)
                t_correct += predicted.eq(targets).sum().item()
                train_loss.append(loss.item())
                loss.backward()
                # parameter update
                optim.step()

        if is_valid:
            # override the results with the validation dataset
            self.model.eval()
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                losses = [criterion(o, targets) for o in outputs]
                ensemble_predictions = [F.softmax(o, dim=1).max(1) for o in outputs]
                for i,vals in enumerate(ensemble_predictions):
                    loss = losses[i]
                    _, predicted = vals
                    v_total += targets.size(0)
                    v_correct += predicted.eq(targets).sum().item()
                    valid_loss.append(loss.item())

            avg_valid_loss = np.average(valid_loss)
            avg_train_loss = np.average(train_loss)
            accuracy = (100. * v_correct / v_total)
            m_train_loss.append(avg_train_loss)
            m_valid_loss.append(avg_valid_loss)
            m_accuracies.append(accuracy)
            print('Epoch:{}, TrainLoss:{}, ValidationLoss:{}, ValidationAccuracy:{}'. \
                  format(epoch, avg_train_loss, avg_valid_loss, accuracy))
        else:

            avg_train_loss = np.average(train_loss)
            accuracy = (100. * t_correct / t_total)
            m_train_loss.append(avg_train_loss)
            m_accuracies.append(accuracy)
            print('Epoch:{}, TrainLoss:{}, Train Accuracy:{}'. \
                  format(epoch, avg_train_loss, accuracy))

        if is_valid:
            return m_valid_loss, m_accuracies
        else:
            return m_train_loss, m_accuracies

    def Test(self, is_eval=False):

        if is_eval:
            for i,m in enumerate(self.models):
                if DEVICE.type == 'cpu':
                    state = torch.load(self.train_weight_path[i], map_location=torch.device('cpu'))
                else:
                    state = torch.load(self.train_weight_path[i])
                m.load_state_dict(state['weights'])

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                X, Y = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(X)
                ensemble_predictions = [F.softmax(o, dim=1).max(1) for o in outputs]
                for i,vals in enumerate(ensemble_predictions):
                    _, predicted = vals
                    total += targets.size(0)
                    correct += predicted.eq(Y).sum().item()

        acc = (100. * correct / total)
        print('Ensemble Test Accuracy:{}'.format(acc))
        return acc


    def __createFolder(self):
        pth = 'results/McDropout/ensemble/'
        if self.activ_learn:
            name = self.m_name + '_' + self.d_name + '_isample' + str(self.isample) + '_e' + str(
                self.epochs) + "_activInstances" + str(self.nn_instances) \
                   + '_r' + str(self.rounds) + '_ac' + str(self.activ_learn) + '_optim-' + self.opt + '_top-k' + str(
                self.topk) + \
                   '_b' + str(self.tr_b_sz) + '_rtAfter' + str(self.retrain) + '_' + self.acq
        else:
            name = self.m_name + '_' + self.d_name + '_isample' + str(self.isample) + '_e' + \
                   str(self.epochs) + '_r' + str(self.rounds) + '_ac' + str(self.activ_learn) + \
                   '_optim-' + self.opt + '_top-k' + str(self.topk) + '_b' + str(self.tr_b_sz) + '_rtAfter' + str(
                self.retrain)

        if not os.path.isdir(pth + name):
            os.mkdir(pth + name)
        else:
            print('folder:{} already exists. Want to override?***NOTE: all contents will be overwritten****'.format(
                pth + name))
            inp = input('Enter y/n')
            if inp == 'y' or inp == 'Y':
                shutil.rmtree(pth + name)
                print('done overwriting the files')
                os.mkdir(pth + name)
            else:
                sys.exit('quitting')
        self.results_dir = pth + name

    # if argument is_sample is False, then it simply acts as a non-ensemble testing with same weights (mean) sampled each time
    def Test_ensemble(self, round):



        for i,m in enumerate(self.models):
            if DEVICE.type == 'cpu':
                state = torch.load(self.train_weight_path[i], map_location=torch.device('cpu'))
            else:
                state = torch.load(self.train_weight_path[i])
            m.load_state_dict(state['weights'])

        predictions = []
        targets = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                for data, target in tqdm(self.test_loader):
                    targets.append(target.cpu().numpy())
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    outputs = F.softmax(model(data), dim=1)
                    preds = outputs.max(1, keepdim=True)[1]
                    predictions.append(preds.cpu().numpy().flatten())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        report = classification_report(predictions, targets, output_dict=True)
        df = pd.DataFrame(report).transpose()
        if round == 1: self.__createFolder()

        if self.activ_learn:
            df.to_csv(self.results_dir + '/accuracy_active_round' + str(round) + '.csv')
        else:
            df.to_csv(self.results_dir + '/accuracy_random_round' + str(round) + '.csv')
        print(df)

    def getTrainedmodel(self, rounds):

        self.train_weight_path = []
        # path to write trained weights
        for i in range(len(self.models)):
            if self.activ_learn:
                train_weight_path = 'trained_weights/ensemble/' + self.m_name + ':' + str(i) + '-' + self.d_name + \
                                         '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k' + str(
                    self.topk) + '_rtAfter' + str(self.retrain) + \
                                         '-' + 'b' + str(
                    self.tr_b_sz) + '_optim-' + self.opt + '_active' + '_' + self.acq + '.pkl'
            else:
                train_weight_path = 'trained_weights/ensemble/' + self.m_name + ':' + str(i) + '-' + self.d_name + '-isample' + str(
                    self.isample) + \
                                         '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k' + str(
                    self.topk) + '_rtAfter' + str(self.retrain) + \
                                         '-' + 'b' + str(
                    self.tr_b_sz) + '_optim-' + self.opt + '_random' + '_' + self.acq + '.pkl'

            self.train_weight_path.append(train_weight_path)

        return (self.models, self.train_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP models that use MNIST and FashionMNIST')
    parser.add_argument('-d', '--dataset', help='dataset type, options 1. mnist, 2. fmnist 3. cifar10 4. cifar100',
                        default='mnist')
    parser.add_argument('-m', '--model',
                        help='model name 1. lenet300-100, 2.lenet512-512, 3.lenet5 4. alexnet 5. VGG 6.densenet',
                        default='lenet300-100')
    parser.add_argument('-mo', '--mode', help='1.train or 2.test', default='test')
    parser.add_argument('-al', '--activelearn', help='1, 2 or 3', default=1, type=int)
    parser.add_argument('-trb', '--train_batch_size', help='batch size', default=100, type=int)
    parser.add_argument('-tsb', '--test_batch_size', help='batch size', default=500, type=int)
    parser.add_argument('-alb', '--active_batch_size', help='batch size', default=512, type=int)
    parser.add_argument('-nni', '--nn_instances', help='number of neural network instances to perform active learning',
                        default=5, type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('-r', '--rounds', help='number of rounds', default=41, type=int)
    parser.add_argument('-rt', '--retrain',
                        help='integer value between 1-# rounds. If  1 -> at each round we retrain from scratch', \
                        default=100, type=int)
    parser.add_argument('-ss', '--seedsample', help='number of seed samples to begin training', default=1000, type=int)
    parser.add_argument('-tk', '--topk', help='number of samples to add during each round', default=100, type=int)
    parser.add_argument('-af', '--acquisitionFunction', help='1. entropy, 2. variation-ratio',
                        default='variation-ratio')
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-v', '--is_valid', help='whether to use validation or not', default=1, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate to start with', default=0.001, type=float)
    parser.add_argument('-ra', '--resume_after', help='resume after a round number, default is 0', \
                        default=0, type=int)
    args = parser.parse_args()
    run_model = RunModel(args.model, args.dataset, args.train_batch_size, args.test_batch_size, args.active_batch_size,
                         args.nn_instances,
                         args.epochs, args.rounds, args.seedsample, args.mode, args.activelearn, args.topk,
                         args.retrain,
                         args.acquisitionFunction, args.optimizer, args.lr, args.resume_after)


    lr = args.lr
    epoch = args.epochs
    patience = 15
    start = 1


    if args.resume_after and args.mode == 'train':
        start = args.resume_after + 1
    write_summary = LogSummary(
        name=args.model + '_' + args.dataset + '_al' + str(args.activelearn) + '_af-' + args.acquisitionFunction)
    if args.mode == 'train':
        for r in range(start, args.rounds):
            if r == 1:
                # for fmnist set epoch = 50 to match bayesian
                epoch, patience = config.get_init_params(args.dataset, args.model)
            if args.activelearn == 1:
                run_model.ActiveSubSelectData(r)
            elif args.activelearn == 2:
                run_model.get_preselected_points(r)
            else:
                run_model.RandomSubSelectData(r)
            # initialize the early stopping criteria for each round
            early_stopping = EarlyStopping(round=r, selected_data=run_model.selected_data, \
                                           patience=patience, verbose=True, typ='accuracy', is_ensemble=True)
            for e in range(epoch):
                start = time.time()
                valid_loss, accuracy = run_model.Train(e, is_valid=args.is_valid)
                print('end of round: {}, lr: {}, time-taken to train: {:.2f} seconds'.format(r, lr, time.time() - start))
                start = time.time()
                model, path_to_write = run_model.getTrainedmodel(r)
                '''
                early stopping for ensemble-based models is very tricky since one of the models stop early, while the rest continue.
                So, as of now, I am not implementing this.
                '''
                early_stopping(accuracy, model, run_model.optimizers, path_to_write)
                print('time-taken to write trained models: {:.2f} seconds'.format(r, lr, time.time() - start))

            acc = run_model.Test()
            # we visualize the average of the accuracies obtained from the ensembles
            write_summary.write_final_accuracy(acc, r)
            epoch = args.epochs
            # reset patience to a lower value for future epochs
            patience = 15

    else:
        for r in range(start, args.rounds):
            print('performing ensemble testing on dataset:{}, model:{}, batch size:{}, epochs:{}, round:{}'
                  .format(args.dataset, args.model, args.train_batch_size, args.epochs, r))
            run_model.getTrainedmodel(r)
            run_model.Test_ensemble(r)
