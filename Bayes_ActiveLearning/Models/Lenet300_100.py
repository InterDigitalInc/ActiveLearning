import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from NeuralLayersJeffrey import LinearGroupNJ
from Utils import FlattenLayer, ModuleWrapper
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)



class Lenet300_100J(nn.Module):

    def __init__(self,i_dim, o_dim, is_cuda=False):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # define a fully connected layer
        self.fc1 = LinearGroupNJ(i_dim,300, is_cuda)
        self.fc2 = LinearGroupNJ(300,100, is_cuda)
        # define the final prediction layer
        self.fc3 = LinearGroupNJ(100,o_dim, is_cuda)

        self.layers = [self.fc1, self.relu_1, self.fc2, self.relu_2, self.fc3]
        self.relu_list = [self.relu_1, self.relu_2]
        self.kl_list = [self.fc1, self.fc2, self.fc3]
    
    def forward(self, x):
        
        # if not self.training:
        #     for l in self.kl_list:
        #         l.deterministic = True
        # else:
        #     for l in self.kl_list:
        #         l.deterministic = False

        x = x.view(-1, 28*28)
        for layer in self.layers:
                x = layer(x)
        return x
    
    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

@variational_estimator
class Lenet300_100BBB(nn.Module):

    def __init__(self,i_dim, o_dim):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        prior_sigma_1 = 1
        prior_sigma_2 = 0.0025
        prior_pi = 0.5
        # output dimension
        self.o_dim = o_dim
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # define a fully connected layer
        self.fc1 = BayesianLinear(i_dim,300,prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.fc2 = BayesianLinear(300,100,prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # define the final prediction layer
        self.fc3 = BayesianLinear(100,o_dim,prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)

        self.layers = [self.fc1, self.relu_1, self.fc2, self.relu_2, self.fc3]
        self.relu_list = [self.relu_1, self.relu_2]
        self.kl_list = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):

        x = x.view(-1, 28*28)
        for layer in self.layers:
                x = layer(x)
        return x


""" For Regression"""

class Lenet300_100J_Regress(nn.Module):

    def __init__(self,i_dim, o_dim, is_cuda=False):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # define a fully connected layer
        self.fc1 = LinearGroupNJ(i_dim,300, is_cuda)
        self.fc2 = LinearGroupNJ(300,100, is_cuda)
        # define the final prediction layer
        self.fc3 = LinearGroupNJ(100,o_dim, is_cuda)

        self.layers = [self.fc1, self.relu_1, self.fc2, self.relu_2, self.fc3]
        self.relu_list = [self.relu_1, self.relu_2]
        self.kl_list = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):

        # if not self.training:
        #     for l in self.kl_list:
        #         l.deterministic = True
        # else:
        #     for l in self.kl_list:
        #         l.deterministic = False

        for layer in self.layers:
                x = layer(x)
        return x

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
