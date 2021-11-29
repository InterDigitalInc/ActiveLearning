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

from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from Models import LenetDropout300_100
from Models import Lenet5Dropout, AlexnetDropout, AlexnetLightDropout, VGGDropout
from Models import Densenet100, DenseNet121
import pkbar
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
from Utils import Config
from Utils import EarlyStopping
import time
# do pip install netcal, src: https://github.com/fabiankueppers/calibration-framework#id59
from netcal.metrics import ECE

if not os.path.isdir('results'):
    os.makedirs('results')
if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')
if not os.path.isdir('results/McDropout/ensemble'):
    os.makedirs('results/McDropout/ensemble')
if not os.path.isdir("trained_weights/ensemble"):
    os.mkdir("trained_weights/ensemble")
config = Config()
HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'
torch.manual_seed(33)
np.random.seed(33)
torch.backends.cudnn.deterministic = True




class RunModel:

    def __init__(self, m_name,d_nam, train_batch_size, test_batch_size, active_batch_size, nn_instances,
                 epoch, rounds,seed_sample, mode,active_learn,topk, retrain, acquisition,
                 optimizer, lr, resume_round, trained_weight_path):

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
        self.twp = trained_weight_path
        self.results_dir = ''
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
        if self.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=l_rate, amsgrad=True)
    def __load_pre_train_model(self, round):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel(round, self.twp)
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
            self.model = LenetDropout300_100(self.i_dim * self.i_dim, self.n_classes).to(DEVICE)
        if self.m_name == 'lenet5':
            self.model = Lenet5Dropout(self.n_classes, self.i_channel, d_rate=[0.1,0.2]).to(DEVICE)
        if self.m_name == 'alexnet':
            self.model = AlexnetDropout(self.n_classes, self.i_channel, d_rate=[0.1,0.2]).to(DEVICE)
        if self.m_name == 'alexnetlight' and self.d_name == 'cifar10':
            self.model = AlexnetLightDropout(self.n_classes, self.i_channel).to(DEVICE)
        if self.m_name == 'vgg' and self.d_name == 'cifar10':
            self.model = VGGDropout(self.n_classes, self.i_channel , 'VGG19').to(DEVICE)
        if self.m_name == 'vgg' and self.d_name == 'cifar100':
            self.model = VGGDropout(self.n_classes, self.i_channel , 'VGG19').to(DEVICE)
        elif self.m_name == 'densenet':
            self.model = DenseNet121(self.n_classes).to(DEVICE)
        print(summary(self.model, (self.i_channel,self.i_dim,self.i_dim)))
        if load_weights and round:
            self.__load_pre_train_model(round)



    def __getVariationRatio(self, outputs):
        predicts = []
        n_instances = len(outputs)
        for out in outputs:
            out = out.squeeze(dim=0)
            _, predicted = out.max(1)
            predicts.append(predicted.unsqueeze(dim=0))
        predicts = torch.cat(predicts, dim=0)
        mods, locs = torch.mode(predicts, dim=0)
        mode_count = predicts.eq(mods).sum(0)
        variation_ratio = 1-torch.div(mode_count.double(), n_instances)
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
            T = self.nn_instances
            correct = 0
            metrics = []
            self.model.eval()
            # set dropouts to true
            if hasattr(self.model, 'drop_train'):
                self.model.drop_train = True
            else:
                for each_module in self.model.modules():
                    if each_module.__class__.__name__.startswith('Dropout'):
                        each_module.train()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(all_data)):
                    # replicate the samples to create T instantiations of NN
                    b_sz = inputs.shape[0]
                    inputs = inputs.view(-1, self.i_channel, self.i_dim, self.i_dim).repeat(T, 1, 1, 1)
                    targets = targets.repeat(T)
                    # note: we don't use targets variable for active learning
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    ensemble_outputs = torch.unsqueeze(F.softmax(self.model(inputs),dim=1), 0)
                    ensemble_outputs = ensemble_outputs.reshape(T, b_sz, self.n_classes)
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
        valid_indx = np.random.choice(indices,split)
        train_sampler = SubsetRandomSampler(list(self.selected_data))
        valid_sampler = SubsetRandomSampler(valid_indx)
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=5)
        self.valid_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=valid_sampler, num_workers=5)

    # training using the proposed active learning approach
    def Train(self, epoch, is_valid):

        t_correct, v_correct = 0, 0
        t_total, v_total = 0, 0
        if hasattr(self.model, 'drop_train'):
            self.model.drop_train = True
        criterion = nn.CrossEntropyLoss()
        # use X% of training data as validation every round by randomly shuffling the unexplored indices of training data
        if epoch == 0:
            self.get_validation_data(is_valid)
        self.model.train()
        train_loss, valid_loss = [], []

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs,targets)
            _, predicted = F.softmax(outputs,dim=1).max(1)
            t_total += targets.size(0)
            t_correct += predicted.eq(targets).sum().item()
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
                loss = criterion(outputs,targets)
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
            accuracy = (100. * t_correct / t_total)
            print('Epoch:{}, TrainLoss:{}, Train Accuracy:{}'. \
                format(epoch, avg_train_loss, accuracy))

            return avg_train_loss,  accuracy

    def Test(self, is_eval=False):

        if is_eval:
            if DEVICE.type == 'cpu':
                state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
            else:
                state = torch.load(self.train_weight_path)
            self.model.load_state_dict(state['weights'])
        correct = 0
        total = 0
        n_bins = 10
        ece = ECE(n_bins)
        confidence = []
        targets = []
        self.model.eval()
        if hasattr(self.model, 'drop_train'):
            self.model.drop_train = False
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tqdm(self.test_loader)):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                outputs = F.softmax(self.model(X),dim=1)
                confidence.append(outputs)
                targets.append(Y.cpu().numpy())
                _, predicted = outputs.max(1)
                total += Y.size(0)
                correct += predicted.eq(Y).sum().item()

        confidence = torch.cat(confidence).cpu().numpy()
        targets = np.array(targets).flatten()
        uncaliberated_error = ece.measure(confidence,targets)
        acc = (100. * correct / total)
        print('Non-Ensemble Metrics-- TestAccuracy:{} ECE:{}, '.format(acc,uncaliberated_error))
        return acc, uncaliberated_error

    def __createFolder(self):
        pth = 'results/McDropout/'
        if self.activ_learn:
            name = self.m_name+'_'+self.d_name+'_isample'+str(self.isample)+'_e'+str(self.epochs)+ "_activInstances" + str(self.nn_instances) \
                   +'_r'+ str(self.rounds)+'_ac'+str(self.activ_learn) + '_optim-' + self.opt +'_top-k'+ str(self.topk) +\
                   '_b'+str(self.tr_b_sz) +'_rtAfter'+str(self.retrain)+'_'+self.acq
        else:
            name = self.m_name+'_'+self.d_name+'_isample'+str(self.isample)+'_e'+ \
                   str(self.epochs)+'_r'+str(self.rounds)+'_ac'+str(self.activ_learn) + \
                       '_optim-' + self.opt + '_top-k'+ str(self.topk) + '_b'+ str(self.tr_b_sz) + '_rtAfter'+ str(self.retrain)

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
    # if argument is_sample is False, then it simply acts as a non-ensemble testing with same weights (mean) sampled each time
    def Test_ensemble(self,round):

        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.model.load_state_dict(state['weights'])
        self.model.eval()
        if hasattr(self.model, 'drop_train'):
            self.model.drop_train = False
        correct = 0
        # number of ensemble samples for testing
        test_samples = 1
        corrects = np.zeros(test_samples, dtype=int)
        predictions = []
        targets = []
        # list to hold expected calibration error
        n_bins = 10
        ece = ECE(n_bins)
        confidence = []
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                targets.append(target.cpu().numpy())
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = torch.zeros(test_samples, self.tst_b_sz, self.n_classes).to(DEVICE)
                for i in range(test_samples):
                    outputs[i] = F.softmax(self.model(data),dim=1)

                # given a batch and the predicted probabilities for C classes, get the mean probability across K test_samples for each class
                output_probs = outputs.mean(0)
                confidence.append(output_probs)
                # predict the output class with highest probability across K test_samples
                preds = outputs.max(2, keepdim=True)[1]
                # From the mean probability, get class with max probability for each d \in test batch
                pred = output_probs.max(1, keepdim=True)[1]  # index of max log-probability
                predictions.append(pred.cpu().numpy().flatten())
                # view as basically reshapes the tensor into the shape of a target tensor, here target is a 1-D tensor, while pred is a 5*1 tensor
                corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
                correct += pred.eq(target.view_as(pred)).sum().item()

        print('Ensemble Accuracy: {}/{}'.format(correct, self.test_len))
        confidence = torch.cat(confidence).cpu().numpy()
        targets = np.array(targets).flatten()
        uncaliberated_error = ece.measure(confidence,targets)
        predictions = np.array(predictions).flatten()
        report = classification_report(predictions, targets, output_dict=True)
        df = pd.DataFrame(report).transpose()
        # add the expected calibration error to the dataframe
        df.loc["ece"] = [uncaliberated_error]*len(df.columns)
        if round == 1: self.__createFolder()

        if self.activ_learn:
            df.to_csv(self.results_dir + '/accuracy_active_round' + str(round) + '.csv')
        else:
            df.to_csv(self.results_dir + '/accuracy_random_round' + str(round) + '.csv')
        print(df)

    def getTrainedmodel(self,rounds, root=''):

        if not root:
            root = 'trained_weights'
        # path to write trained weights
        if self.activ_learn:
            self.train_weight_path = root + '/' + self.m_name + '-' + self.d_name + \
                '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k'+ str(self.topk) +'_rtAfter'+str(self.retrain) + \
                '-' + 'b' + str(self.tr_b_sz) +'_optim-'+ self.opt +'_active'+'_'+self.acq+'.pkl'
        else:
            self.train_weight_path = root + '/' + self.m_name + '-' + self.d_name + '-isample'+ str(self.isample) + \
                '-' + 'e' + str(self.epochs) + '-r' + str(rounds) + '_top-k'+ str(self.topk) +'_rtAfter'+str(self.retrain) + \
                '-' + 'b' + str(self.tr_b_sz) +'_optim-'+ self.opt + '_random'+'_'+self.acq+'.pkl'
        return (self.model, self.train_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP models that use MNIST and FashionMNIST')
    parser.add_argument('-d', '--dataset', help='dataset type, options 1. mnist, 2. fmnist 3. cifar10 4. cifar100', default='mnist')
    parser.add_argument('-m', '--model', help='model name 1. lenet300-100, 2.lenet512-512, 3.lenet5 4. \
        alexnet 5. alexnetlight 6. vgg 7.densenet', default='lenet300-100')
    parser.add_argument('-mo', '--mode', help='1.train or 2.test', default='test')
    parser.add_argument('-al', '--activelearn', help='1, 2 or 3', default=1, type=int)
    parser.add_argument('-bs', '--train_batch_size', help='batch size', default=100, type=int)
    parser.add_argument('-tbs', '--test_batch_size', help='batch size', default=500, type=int)
    parser.add_argument('-abs', '--active_batch_size', help='batch size', default=512, type=int)
    parser.add_argument('-nni', '--nn_instances', help='number of neural network instances to perform active learning', default=5, type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('-r', '--rounds', help='number of rounds', default=41, type=int)
    parser.add_argument('-rt', '--retrain', help='integer value between 1-# rounds. If  1 -> at each round we retrain from scratch',\
         default=0, type=int)
    parser.add_argument('-ss', '--seedsample', help='number of seed samples to begin training', default=1000, type=int)
    parser.add_argument('-tk', '--topk', help='number of samples to add during each round', default=100, type=int)
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
                         args.acquisitionFunction, args.optimizer, args.lr, args.resume_after, args.trained_weight_path)

    lr = args.lr
    start = 1
    if args.resume_after and args.mode == 'train':
        start = args.resume_after + 1
    if start == 1:
        epoch = start_epoch
    else:
        epoch = subseq_epoch
    if args.mode == 'train':

        write_summary = LogSummary(name= 'dropout_' + args.model + '_' + args.dataset + '_al' + \
        str(args.activelearn) + '_af-' + args.acquisitionFunction + '_retrain' + str(args.retrain) + '_nni-' + str(args.nn_instances))

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
                valid_loss, accuracy = run_model.Train(e,is_valid=args.is_valid)
                print('round:{}, lr:{} time-taken :{:.2f} seconds'.format(r, lr, time.time() - start))
                model, path_to_write = run_model.getTrainedmodel(r)
                early_stopping(valid_loss, model, run_model.optimizer, path_to_write)
                if early_stopping.early_stop:
                    break
            acc, ece = run_model.Test()
            write_summary.write_final_accuracy(acc, r)
            write_summary.write_final_ece(ece, r)
            epoch = subseq_epoch

    else:
        for r in range(start, args.rounds):
            print('performing ensemble testing on dataset:{}, model:{}, epochs:{}, round:{}'
                  .format(args.dataset, args.model, args.epochs, r))
            run_model.getTrainedmodel(r, root=args.trained_weight_path)
            # acc = run_model.Test(is_eval=True)
            run_model.Test_ensemble(r)
