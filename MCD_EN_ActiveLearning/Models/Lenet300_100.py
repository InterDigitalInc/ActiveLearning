import torch
import math
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)
# indicates the number of MCMC samples we need
SAMPLES = 3

# this is the simple point estimate lenet without dropouts (used for ensemble)
class Lenet300_100(nn.Module):

    def __init__(self,i_dim, o_dim):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        # define a fully connected layer
        self.l1 = nn.Linear(i_dim,300, bias=True)
        # define a second FCN
        self.l2 = nn.Linear(300,100, bias=True)
        # define the final prediction layer
        self.ol = nn.Linear(100,o_dim, bias=True)

    def forward(self,x):

        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.ol(x)
        # not sure why we give dim=1
        return x

# this is the simple point estimate lenet with dropouts
class LenetDropout300_100(nn.Module):

    def __init__(self,i_dim, o_dim, d_rate=0.25):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        # dropout rate
        self.d_rate = d_rate
        self.drop_train = False
        # define a fully connected layer
        self.l1 = nn.Linear(i_dim,300, bias=True)
        # define the dropout layer
        # define a second FCN
        self.l2 = nn.Linear(300,100, bias=True)
        # define the final prediction layer
        self.ol = nn.Linear(100,o_dim, bias=True)

    def forward(self,x):

        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.dropout(x, training=self.drop_train)
        x = F.relu(self.l2(x))
        x = F.dropout(x, training=self.drop_train)
        x = self.ol(x)
        # not sure why we give dim=1
        return x

""" For Regression"""
class LenetDropout300_100_Regress(nn.Module):

    def __init__(self,i_dim, o_dim, d_rate=0.25, is_dropout=True):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        # dropout rate
        self.d_rate = d_rate
        self.is_dropout = is_dropout
        # define a fully connected layer
        self.l1 = nn.Linear(i_dim,300, bias=True)
        # define the dropout layer
        # define a second FCN
        self.l2 = nn.Linear(300,100, bias=True)
        # define the final prediction layer
        self.ol = nn.Linear(100,o_dim, bias=True)

    def forward(self,x):

        x = F.relu(self.l1(x))
        if self.is_dropout:
            x = F.dropout(x, training=True, p=self.d_rate)
        x = F.relu(self.l2(x))
        if self.is_dropout:
            x = F.dropout(x, training=True, p=self.d_rate)
        x = self.ol(x)
        # not sure why we give dim=1
        return x

class Lenet300_100_Regress(nn.Module):

    def __init__(self,i_dim, o_dim, is_dropout=True):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        # define a fully connected layer
        self.l1 = nn.Linear(i_dim,300, bias=True)
        # define the dropout layer
        # define a second FCN
        self.l2 = nn.Linear(300,100, bias=True)
        # define the final prediction layer
        self.ol = nn.Linear(100,o_dim, bias=True)

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.ol(x)
        # not sure why we give dim=1
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
