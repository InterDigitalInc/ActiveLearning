import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
from NeuralLayersJeffrey import LinearGroupNJ, Conv2dGroupNJ
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
torch.manual_seed(33)


# end-to-end bayesian NN using Jeffrey's layers
class AlexnetJeffrey(nn.Module):

    def __init__(self, n_classes, i_channel, is_cuda=False):
        super().__init__()

        if i_channel == 1:
            i_dense_dim = 2304
        else:
            i_dense_dim = 4096

        self.conv_1 = Conv2dGroupNJ(in_channels=i_channel, out_channels=64, kernel_size=3, stride=1, padding=2, cuda=is_cuda)
        self.conv_2 = Conv2dGroupNJ(in_channels=64, out_channels=192, kernel_size=3, padding=2, cuda=is_cuda)
        self.conv_3 = Conv2dGroupNJ(in_channels=192, out_channels=384, kernel_size=3, padding=1, cuda=is_cuda)
        self.conv_4 = Conv2dGroupNJ(in_channels=384, out_channels=256, kernel_size=3, padding=1, cuda=is_cuda)
        self.conv_5 = Conv2dGroupNJ(in_channels=256, out_channels=256, kernel_size=3, padding=1, cuda=is_cuda)

        # fully connected layers
        self.fc_1 = LinearGroupNJ(in_features=i_dense_dim, out_features=512, cuda=is_cuda)
        self.fc_2 = LinearGroupNJ(in_features=512, out_features=256, cuda=is_cuda)
        self.fc_3 = LinearGroupNJ(in_features=256, out_features=n_classes, cuda=is_cuda)

        self.kl_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.fc_1, self.fc_2, self.fc_3]

    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)
        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.relu(self.conv_5(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

# this architecture is similar to https://keras.io/examples/cifar10_cnn/
class AlexnetLightJeffrey(nn.Module):

    def __init__(self,n_classes, i_channel, is_cuda=False):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 256
        self.train_mode = False
        self.conv_1 = Conv2dGroupNJ(in_channels=i_channel, out_channels=32, kernel_size=3, stride=1, padding=1, cuda=is_cuda)
        self.conv_2 = Conv2dGroupNJ(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, cuda=is_cuda)
        self.conv_3 = Conv2dGroupNJ(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, cuda=is_cuda)
        self.conv_4 = Conv2dGroupNJ(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, cuda=is_cuda)

        # fully connected layers
        self.fc_1 = LinearGroupNJ(in_features=i_dense_dim, out_features=512, cuda=is_cuda)
        self.fc_2 = LinearGroupNJ(in_features=512, out_features=256, cuda=is_cuda)
        self.fc_3 = LinearGroupNJ(in_features=256, out_features=n_classes, cuda=is_cuda)

        self.kl_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.fc_1, self.fc_2, self.fc_3]

    def forward(self, x):

        if self.train_mode:
            for l in self.kl_list:
                l.deterministic = True
        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)

        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out
    
    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

# bayesian alexnet for CIFAR 10
@variational_estimator
class AlexnetBBB(nn.Module):

    def __init__(self, n_classes, i_channel):
        super().__init__()

        if i_channel == 1:
            i_dense_dim = 2304
        else:
            i_dense_dim = 4096

        self.conv_1 = BayesianConv2d(in_channels=i_channel, out_channels=64, kernel_size=(3,3), stride=1, padding=2)
        self.conv_2 = BayesianConv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=2)
        self.conv_3 = BayesianConv2d(in_channels=192, out_channels=384, kernel_size=(3,3), padding=1)
        self.conv_4 = BayesianConv2d(in_channels=384, out_channels=256, kernel_size=(3,3), padding=1)
        self.conv_5 = BayesianConv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1)
        self.mx_pl = nn.MaxPool2d(kernel_size=3, stride=2)

        # fully connected layers
        self.fc_1 = BayesianLinear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = BayesianLinear(in_features=512, out_features=256)
        self.fc_3 = BayesianLinear(in_features=256, out_features=n_classes)

        self.kl_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.fc_1, self.fc_2, self.fc_3]
    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out,kernel_size=2)
        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out))
        out = F.relu(self.conv_4(out))
        out = F.relu(self.conv_5(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

# this just for paper, that mimics lower capacity Mcdropout, 25% less for CNN and 50% for Dense
class AlexnetLightJeffreyLowCapacity(nn.Module):

    def __init__(self,n_classes, i_channel, is_cuda=False):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 192
        self.train_mode = False
        self.conv_1 = Conv2dGroupNJ(in_channels=i_channel, out_channels=24, kernel_size=3, stride=1, padding=1, cuda=is_cuda)
        self.conv_2 = Conv2dGroupNJ(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=0, cuda=is_cuda)
        self.conv_3 = Conv2dGroupNJ(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, cuda=is_cuda)
        self.conv_4 = Conv2dGroupNJ(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=0, cuda=is_cuda)

        # fully connected layers
        self.fc_1 = LinearGroupNJ(in_features=i_dense_dim, out_features=256, cuda=is_cuda)
        self.fc_2 = LinearGroupNJ(in_features=256, out_features=128, cuda=is_cuda)
        self.fc_3 = LinearGroupNJ(in_features=128, out_features=n_classes, cuda=is_cuda)

        self.kl_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.fc_1, self.fc_2, self.fc_3]

    def forward(self, x):

        if self.train_mode:
            for l in self.kl_list:
                l.deterministic = True
        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)

        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out
    
    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
