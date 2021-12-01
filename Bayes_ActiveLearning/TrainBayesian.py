import torch
import GPUtil

IS_CUDA = torch.cuda.is_available()
if not IS_CUDA:
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
from Models import Lenet300_100J, Lenet300_100BBB
from Models import Lenet5Jeffrey
from Models import AlexnetJeffrey, AlexnetLightJeffrey, AlexnetBBB
from Models import AlexnetLightJeffreyLowCapacity
from Models import VGG, VggBBB
import os
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from os.path import expanduser
from torch.distributions import Categorical
import shutil
import sys
import torch.nn.functional as F
from Utils import Config
from Utils import EarlyStopping
import time
from Utils import LogSummary
# from Models import densenet121bc, densenet100bc
from Models import DenseNet201, DenseNet121, DenseNet100
# do pip install netcal, src: https://github.com/fabiankueppers/calibration-framework#id59
from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling

config = Config()
HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'
torch.manual_seed(33)
np.random.seed(33)
torch.backends.cudnn.deterministic = True


if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')
if not os.path.isdir('results'):
    os.makedirs('results')


class SaveOutput:
    def __init__(self, n_instances, b_sz, round):
        self.T = n_instances
        self.b_sz = b_sz
        self.outputs = []
        self.r = round
        self.cnt = 0

    def __call__(self, module, module_in, module_out):
        # get just one data point, write the output of  just a single datapoint
        if self.cnt < 3:
            sample_data = np.random.randint(self.b_sz)
            outs = module_out.view(self.T, self.b_sz, -1)[:, 0, :]
            l_sz = outs.shape[1]
            write_summary.per_round_layer_output(l_sz, outs, self.r)
            self.cnt += 1

    def clear(self):
        self.outputs = []


class RunModel:

    def __init__(self, m_name, m_typ, d_nam, train_batch_size, test_batch_size, active_batch_size, nn_instances,
                 epoch, round, seed_sample, mode, active_learn, topk, n_mcmc, beta_type, retrain,
                 acquisition, optimizer, lr, resume_round):

        self.m_name = m_name
        self.topk = topk
        self.m_typ = m_typ
        self.activ_learn = active_learn
        self.d_name = d_nam
        self.epochs = epoch
        self.rounds = round
        # initial sample size to train NN
        self.isample = seed_sample
        self.criterion = nn.CrossEntropyLoss()
        self.tr_b_sz = train_batch_size
        self.beta_typ = beta_type
        self.tst_b_sz = test_batch_size
        self.activ_lrn_b_sz = active_batch_size
        self.nn_instances = nn_instances
        # number of MCMC samples
        self.n_samples = n_mcmc
        if self.m_typ == 'jeffrey':
            self.n_samples = 1
        self.d_rate = 0.2
        self.l_rate = lr
        self.acq = acquisition
        self.retrain = retrain
        self.optim = optimizer
        self.results_dir = ''
        self.n_classes, self.i_channel, self.i_dim, self.train_d, self.test_d = config.get_data(d_nam)
        self.test_loader = DataLoader(self.test_d, batch_size=self.tst_b_sz, shuffle=True, num_workers=1)
        self.selected_data = set([])
        self.unexplored = set(range(len(self.train_d)))
        self.test_len = len(self.test_d)
        self.train_len = len(self.train_d)

        if resume_round:
            self.InitModel(load_weights=True, res_round=resume_round)
        else:
            self.InitModel()
            self.init_optimizer(self.l_rate)

        t_param = sum(p.numel() for p in self.model.parameters())
        print('Running Model:{}, mode:{}, #Parameters:{}, Dataset:{}, Epoch:{}, Batch Size:{}'
              .format(m_name, mode, t_param, d_nam, self.epochs, self.tr_b_sz))
        print(self.model)

    def init_optimizer(self, l_rate):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)

    def __load_pre_train_model(self, load_weights, res_round):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel(res_round)
        # load the weights
        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.selected_data = state['selected_data']
        self.init_optimizer(self.l_rate)
        # if load weights, then we load the previous weights and optimizer state
        if load_weights:
            self.model.load_state_dict(state['weights'])
            self.optimizer.load_state_dict(state['optimizer'])

    def InitModel(self, load_weights=False, res_round=None):

        if self.m_name == 'Blenet300-100' and self.m_typ == 'jeffrey':
            self.model = Lenet300_100J(self.i_dim * self.i_dim, self.n_classes, IS_CUDA).to(DEVICE)
        elif self.m_name == 'Blenet300-100' and self.m_typ == 'BBB':
            self.model = Lenet300_100BBB(self.i_dim * self.i_dim, self.n_classes).to(DEVICE)
        elif self.m_name == 'lenet5' and self.m_typ == 'jeffrey':
            self.model = Lenet5Jeffrey(self.n_classes, self.i_channel, IS_CUDA).to(DEVICE)
        elif self.m_name == 'alexnet' and self.m_typ == 'BBB':
            self.model = AlexnetBBB(self.n_classes, self.i_channel).to(DEVICE)
        elif self.m_name == 'alexnet' and self.m_typ == 'jeffrey':
            self.model = AlexnetJeffrey(self.n_classes, self.i_channel, IS_CUDA).to(DEVICE)
        elif self.m_name == 'alexnetlight' and self.m_typ == 'jeffrey':
            self.model = AlexnetLightJeffrey(self.n_classes, self.i_channel, IS_CUDA).to(DEVICE)
            # self.model = AlexnetLightJeffreyLowCapacity(self.n_classes, self.i_channel, IS_CUDA).to(DEVICE)
        elif self.m_name == 'vgg' and self.m_typ == 'jeffrey':
            self.model = VGG(self.n_classes, self.i_channel, 'VGG19', IS_CUDA).to(DEVICE)
            print(self.model)
        elif self.m_name == 'vgg' and self.m_typ == 'BBB':
            self.model = VggBBB(self.n_classes, self.i_channel, 'VGG19').to(DEVICE)
            # torch.nn.DataParallel(self.model.features)
            t_param = sum(p.numel() for p in self.model.parameters())
        elif self.m_name == 'densenet' and self.m_typ == 'jeffrey':
            self.model = DenseNet121(self.n_classes, IS_CUDA).to(DEVICE)
            print(self.model)
        if res_round:
            self.__load_pre_train_model(load_weights, res_round)


    def __getVariationRatio(self, outputs):
        predicts = []
        n_instances = len(outputs)
        # ToDO: should avoid iteration and need to perform these operations directly on the tensor
        for out in outputs:
            out = out.squeeze(dim=0)
            _, predicted = out.max(1)
            predicts.append(predicted.unsqueeze(dim=0))
        predicts = torch.cat(predicts, dim=0)
        mods, locs = torch.mode(predicts, dim=0)
        mode_count = predicts.eq(mods).sum(0)
        variation_ratio = 1 - torch.div(mode_count.double(), n_instances)
        return variation_ratio

    def __getVariationRatio_jeffry(self, outputs):
        predicts = []
        n_instances = len(outputs)
        # ToDO: should avoid iteration and need to perform these operations directly on the tensor
        for out in outputs:
            out = out.squeeze(dim=0)
            _, predicted = out.max(1)
            predicts.append(predicted.unsqueeze(dim=0))
        predicts = torch.cat(predicts, dim=0)
        variation_ratio = torch.var(predicts.double(), dim=0)
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

            all_data = DataLoader(self.train_d, batch_size=self.activ_lrn_b_sz, num_workers=1)
            correct = 0
            metrics = []
            hook_handles = []
            save_output = SaveOutput(n_instances=self.nn_instances, b_sz=self.activ_lrn_b_sz, round=round)
            self.model.eval()
            for layer in self.model.kl_list:
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(all_data)):
                    # replicate the samples to create T instantiations of NN
                    b_sz = inputs.shape[0]
                    save_output.b_sz = b_sz
                    inputs = inputs.view(-1, self.i_channel, self.i_dim, self.i_dim).repeat(self.nn_instances, 1, 1, 1)
                    targets = targets.repeat(self.nn_instances)
                    # note: we don't use targets variable for active learning
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    # for i in range(15):
                    if self.m_typ == 'BBB':
                        outputs = self.model(inputs)
                        ensemble_outputs = torch.unsqueeze(F.softmax(outputs, dim=1), 0)
                    elif self.m_typ == 'jeffrey':
                        ensemble_outputs = torch.unsqueeze(F.softmax(self.model(inputs), dim=1), 0)
                    ensemble_outputs = ensemble_outputs.reshape(self.nn_instances, b_sz, self.n_classes)
                    if self.acq == 'entropy':
                        metric = self.__getEntropy(ensemble_outputs)
                        metrics.append(metric)
                    if self.acq == 'variation-ratio':
                        metric = self.__getVariationRatio(ensemble_outputs)
                        metrics.append(metric)
                # clear the per-layer outputs and reset count and remove hooks
                save_output.clear()
                save_output.cnt = 0
                for h in hook_handles:
                    h.remove()
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

    def objective(self, output, target, kl_divergence, beta):
        discrimination_loss = nn.functional.cross_entropy
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + kl_divergence * beta
        return discrimination_error, variational_bound, kl_divergence

    # thiss method simply performs random sampling of data instead of active learning-based sampling
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
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=1)
        self.valid_loader = DataLoader(self.train_d, batch_size=256, sampler=valid_sampler, num_workers=1)

    # training using the proposed active learning approach
    def Train(self, round, epoch, is_valid):

        t_correct, v_correct = 0, 0
        t_total, v_total = 0, 0
        # use 10% of training data as validation every round by randomly shuffling the unexplored indices of training data
        if epoch == 0:
            self.get_validation_data(is_valid)
        self.model.train()
        t_loss, v_loss = [], []
        t_lkhood, v_lkhood = [], []
        t_kl, v_kl = [], []
        self.model.train()
        m = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):

            if self.beta_typ == 'blundell':
                beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
            elif self.beta_typ == 'soenderby':
                beta = min(epoch / (self.epochs // 4), 1)
            elif self.beta_typ == 'standard1':
                beta = 1 / m
            elif self.beta_typ == 'standard2':
                beta = 1 / len(self.selected_data)

            X = inputs.view(-1, self.i_channel, self.i_dim, self.i_dim).repeat(self.n_samples, 1, 1, 1)
            Y = targets.repeat(self.n_samples)
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            self.optimizer.zero_grad()
            if self.m_typ == 'BBB':
                outputs = self.model(X)
                loss = self.model.sample_elbo(inputs=X,
                                              labels=Y,
                                              criterion=self.criterion,
                                              sample_nbr=self.n_samples,
                                              complexity_cost_weight=beta)
                log_like, scaled_kl = 0, 0
                t_lkhood.append(log_like)
                t_kl.append(scaled_kl)
            if self.m_typ == 'jeffrey':
                outputs = self.model(X)
                loss, log_like, scaled_kl = self.objective(outputs, Y, self.model.kl_divergence(), beta)
                t_lkhood.append(log_like.item())
                t_kl.append(scaled_kl.item())

            # if you are getting the ouputs from sample_elbo function do outputs.mean(dim=0).max(1)
            _, predicted = F.softmax(outputs, dim=1).max(1)
            t_total += targets.size(0)
            t_correct += predicted.eq(Y).sum().item()
            t_loss.append(loss.item())
            loss.backward()
            # parameter update
            self.optimizer.step()
            if self.m_typ == 'jeffrey':
                for layer in self.model.kl_list:
                    layer.clip_variances()

        if is_valid:
            m = len(self.valid_loader)
            # override the results with the validation dataset
            self.model.eval()
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.valid_loader)):

                if self.beta_typ == 'blundell':
                    beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
                elif self.beta_typ == 'soenderby':
                    beta = min(epoch / (self.epochs // 4), 1)
                elif self.beta_typ == 'standard1':
                    beta = 1 / m
                elif self.beta_typ == 'standard2':
                    beta = 1 / len(self.selected_data)

                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                if self.m_typ == 'BBB':
                    outputs = self.model(inputs)
                    loss = self.model.sample_elbo(inputs=inputs,
                                                  labels=targets,
                                                  criterion=self.criterion,
                                                  sample_nbr=self.n_samples,
                                                  complexity_cost_weight=beta)
                    log_like, scaled_kl = torch.tensor(0), torch.tensor(0)
                elif self.m_typ == 'jeffrey':
                    outputs = self.model(inputs)
                    loss, log_like, scaled_kl = self.objective(outputs, targets, self.model.kl_divergence(), beta)

                _, predicted = F.softmax(outputs, dim=1).max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()
                v_loss.append(loss.item())
                v_lkhood.append(log_like.item())
                v_kl.append(scaled_kl.item())

            avg_valid_loss = np.average(v_loss)
            avg_train_loss = np.average(t_loss)
            avg_valid_lkhood = np.average(v_lkhood)
            avg_valid_kl = np.average(v_kl)
            v_accuracy = (100. * v_correct / v_total)

            avg_train_lkhood = np.average(t_lkhood)
            avg_train_kl = np.average(t_kl)
            t_accuracy = (100. * t_correct / t_total) / self.n_samples

            print(
                'Epoch:{}, AvgTrainLoss:{}, AvgTrainLog-Likelihood:{}, AvgTrainScaledKL:{}, AvgTrainAccuracy:{}'.format(
                    epoch, avg_train_loss, \
                    avg_train_lkhood, avg_train_kl, t_accuracy))
            print('Epoch:{}, AvgValidLoss:{}, AvgValidLog-Likelihood:{}, AvgValidScaledKL:{}, ValidAccuracy:{}'.format(
                epoch, avg_valid_loss, \
                avg_valid_lkhood, avg_valid_kl, v_accuracy))

            return avg_valid_loss, v_accuracy
        else:
            avg_train_loss = np.average(t_loss)
            avg_train_lkhood = np.average(t_lkhood)
            avg_train_kl = np.average(t_kl)
            t_accuracy = (100. * t_correct / t_total) / self.n_samples
            print(
                'Epoch:{}, Train AvgLoss:{}, Train AvgLog-Likelihood:{}, Train AvgScaledKL:{}, TrainAccuracy:{}'.format(
                    epoch, avg_train_loss, \
                    avg_train_lkhood, avg_train_kl, t_accuracy))
            return avg_train_loss, t_accuracy

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
        temperature = TemperatureScaling()
        ece = ECE(n_bins)
        confidence = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tqdm(self.test_loader)):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                if self.m_typ == 'BBB':
                    outputs = F.softmax(self.model(X),dim=1)
                elif self.m_typ == 'jeffrey':
                    outputs = F.softmax(self.model(X),dim=1)
                confidence.append(outputs)
                targets.append(Y.cpu().numpy())
                _, predicted = outputs.max(1)
                total += Y.size(0)
                correct += predicted.eq(Y).sum().item()

        confidence = torch.cat(confidence).cpu().numpy()
        targets = np.array(targets).flatten()
        temperature.fit(confidence, targets)
        calibrated = temperature.transform(confidence)
        uncaliberated_error = ece.measure(confidence,targets)
        calibrated_score = ece.measure(calibrated, targets)
        acc = (100. * correct / total)
        print('Non-Ensemble Metrics-- TestAccuracy:{} ECE:{}, '.format(acc,uncaliberated_error))
        return acc, calibrated_score

    def __createFolder(self):
        pth = 'results/'
        if self.activ_learn:
            name = self.m_name + '_' + self.d_name + '_isample' + str(self.isample) + '_e' + str(
                self.epochs) + '_r' + str(self.rounds) + '_ac' + str(self.activ_learn) + '_Klreg-' + self.beta_typ + \
                   '_b' + str(self.tr_b_sz) + '_topK-' + str(self.topk) + '_mcmc_' + str(
                self.n_samples) + 'activInstances'+ str(self.nn_instances) + '_netType-' + self.m_typ + \
                   'optim-' + self.optim + '_' + self.acq + '_rtAfter' + str(self.retrain)
        else:
            name = self.m_name + '_' + self.d_name + '_isample' + str(self.isample) + '_e' + str(
                self.epochs) + '_r' + str(self.rounds) + '_ac' + str(self.activ_learn) + '_Klreg-' + self.beta_typ + \
                   '_b' + str(self.tr_b_sz) + '_topK-' + str(self.topk) + '_mcmc_' + str(
                self.n_samples) + 'activInstances'+ str(self.nn_instances) + '_netType-' + self.m_typ + \
                   'optim-' + self.optim + self.acq + '_rtAfter' + str(self.retrain)
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
    def Test_ensemble(self, round, is_sample=True):

        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)
        self.model.load_state_dict(state['weights'])
        self.model.eval()

        correct = 0
        # number of ensemble samples for testing
        test_samples = 5
        corrects = np.zeros(test_samples, dtype=int)
        predictions = []
        targets = []
        n_bins = 10
        temperature = TemperatureScaling()
        ece = ECE(n_bins)
        confidence = []
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                targets.append(target.cpu().numpy())
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = torch.zeros(test_samples, self.tst_b_sz, self.n_classes).to(DEVICE)
                for i in range(test_samples):
                    # for each sample we obtain the output from our trained NN. Here we set sample=True,
                    # because we need different weights for each prediction.
                    # give self.model(data, sample=True) for ensemble testing, the default is not to sample from weight distribution
                    if self.m_typ == 'BBB':
                        op = self.model(data)
                        outputs[i] = F.softmax(op, dim=1)
                    elif self.m_typ == 'jeffrey':
                        outputs[i] = F.softmax(self.model(data), dim=1)

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
        temperature.fit(confidence, targets)
        calibrated = temperature.transform(confidence)
        uncaliberated_error = ece.measure(confidence,targets)
        calibrated_error = ece.measure(calibrated, targets)
        predictions = np.array(predictions).flatten()
        report = classification_report(predictions, targets, output_dict=True)
        df = pd.DataFrame(report).transpose()
        # add the expected calibration error to the dataframe
        df.loc["ece"] = [uncaliberated_error]*len(df.columns)
        df.loc["ece_calibrated"] = [calibrated_error]*len(df.columns)
        if round == 1: self.__createFolder()

        if self.activ_learn:
            df.to_csv(self.results_dir + '/accuracy_active_round' + str(round) + '.csv')
        else:
            df.to_csv(self.results_dir + '/accuracy_random_round' + str(round) + '.csv')
        print(df)

    def getTrainedmodel(self, rounds, root=''):
        # path to write trained weights
        # retrain = 100
        if not root:
            root = 'trained_weights'
        if self.activ_learn:
            self.train_weight_path = root + '/' + self.m_name + '-' + self.d_name + '-' + 'e' + str(
                self.epochs) + '-r' + str(rounds) + '_top-k' + str(self.topk) + '_rtAfter' + str(self.retrain) + \
                                     '-b' + str(self.tr_b_sz) + '_mcmc' + str(self.n_samples) + "_activInstances" + str(self.nn_instances) +'_seedsample' + \
                                     str(self.isample) + '_netType-' + self.m_typ + '_optim-' + self.optim + '_active' + '_' + self.acq + '.pkl'
        else:
            self.train_weight_path = root + '/' + self.m_name + '-' + self.d_name + '-' + 'e' + str(
                self.epochs) + \
                                     '-r' + str(rounds) + '_top-k' + str(self.topk) + '_rtAfter' + str(self.retrain) + \
                                     '-b' + str(self.tr_b_sz) + '_mcmc' + str(self.n_samples) + '_seedsample' + str(
                self.isample) + \
                                     '_netType-' + '_optim-' + self.optim + self.m_typ + '_random' + '_' + self.acq + '.pkl'
        return (self.model, self.train_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP models that use MNIST and FashionMNIST')
    parser.add_argument('-d', '--dataset', help='dataset type, options 1. mnist, 2. fmnist 3.cifar10 4.cifar100',
                        default='mnist')
    parser.add_argument('-m', '--model',
                        help='model name 1.FBlenet300-100 2.Lenet5Hybrid 3.lenet5 4.AlexnetHybrid 5.vgg', \
                        default='Blenet300-100')
    parser.add_argument('-mt', '--modeltype', help='type of bayesian network 1.BBB (bayes-by-backprop) 2.jeffrey', default='jeffrey')
    parser.add_argument('-mo', '--mode', help='1.train or 2.test', default='test')
    parser.add_argument('-al', '--activelearn', help='1, 2 or 3', default=1, type=int)
    parser.add_argument('-bs', '--train_batch_size', help='batch size', default=100, type=int)
    parser.add_argument('-tbs', '--test_batch_size', help='batch size', default=500, type=int)
    parser.add_argument('-abs', '--active_batch_size', help='batch size', default=128, type=int)
    parser.add_argument('-nni', '--nn_instances', help='number of neural network instances to perform active learning', default=25, type=int)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('-r', '--rounds', help='number of rounds', default=41, type=int)
    parser.add_argument('-rt', '--retrain',
                        help='integer value between 1-# rounds. If  0 -> there is no retraining at all, if.', \
                        default=0, type=int)
    parser.add_argument('-ss', '--seedsample', help='number of seed samples to begin training', default=1000, type=int)
    parser.add_argument('-tk', '--topk', help='number of samples to add during each round', default=100, type=int)
    parser.add_argument('-af', '--acquisitionFunction', help='1. entropy, 2. variation-ratio',
                        default='variation-ratio')
    parser.add_argument('-mc', '--n_mcmc', help='number of mcmc samples, defalut is 5.', default=5, type=int)
    parser.add_argument('-kls', '--beta', help='KL scaling factor 1. standard1 2. standard2 3. blundell',
                        default='standard2')
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-v', '--is_valid', help='whether to use validation or not', default=0, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate to start with', default=0.001, type=float)
    parser.add_argument('-ra', '--resume_after', help='resume after a round number, default is 0', \
                        default=0, type=int)

    args = parser.parse_args()
    start_epoch, subseq_epoch, patience = config.get_init_params_v2_beta(args.dataset, args.model, args.retrain)

    run_model = RunModel(args.model, args.modeltype, args.dataset, args.train_batch_size, args.test_batch_size, args.active_batch_size,
                         args.nn_instances, 50, args.rounds, args.seedsample, args.mode, args.activelearn, args.topk, args.n_mcmc,
                         args.beta, args.retrain, args.acquisitionFunction, args.optimizer, args.lr, args.resume_after)

    lr = args.lr
    start = 1
    if args.resume_after and args.mode == 'train':
        start = args.resume_after + 1
    if start == 1:
        epoch = start_epoch
    else:
        epoch = subseq_epoch
    if args.mode == 'train':
        write_summary = LogSummary(
            name=args.model + '_' + args.modeltype + '_' + args.dataset + '_al' + \
             str(args.activelearn) + '_af-' + args.acquisitionFunction + '_retrain' + \
             str(args.retrain) + '_nni-' + str(args.nn_instances))
    
    if args.mode == 'train':
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
                valid_loss, accuracy = run_model.Train(r, e, is_valid=args.is_valid)
                print('round:{}, lr:{} time-taken :{:.2f} seconds'.format(r, lr, time.time() - start))
                model, path_to_write = run_model.getTrainedmodel(r)
                early_stopping(valid_loss, model, run_model.optimizer, path_to_write)
                if early_stopping.early_stop:
                    break

            acc, ece = run_model.Test()
            # acc = run_model.Test()
            write_summary.write_final_accuracy(acc, r)
            write_summary.write_final_ece(ece, r)
            epoch = subseq_epoch

    else:
        root = input("Enter source location of trained files (enter nothing for default location):")
        for r in range(start, args.rounds):
            print('\n performing ensemble testing on dataset:{}, model:{}, batch size:{}, epochs:{}, round:{}'
                  .format(args.dataset, args.model, args.train_batch_size, args.epochs, r))
            run_model.getTrainedmodel(r,root)
            # acc = run_model.Test(is_eval=True)
            # write_summary.write_final_accuracy(np.average(acc), r)
            run_model.Test_ensemble(r)
