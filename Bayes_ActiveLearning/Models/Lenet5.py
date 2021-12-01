import torch
import torch.nn as  nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Dropout2d
from NeuralLayersJeffrey import LinearGroupNJ, Conv2dGroupNJ
torch.manual_seed(33)

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

# following link shows the Lenet-5 architecture https://engmrk.com/lenet-5-a-classic-cnn-architecture/
class Lenet5Jeffrey(nn.Module):

    def __init__(self,n_classes, i_channel, is_cuda=False):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            i_dense_dim = 400
        # resnet has 3 input channels (RGB) and 6 filters with kernel size 5
        self.conv_1 = Conv2dGroupNJ(in_channels=i_channel, out_channels=6,kernel_size=5, cuda=is_cuda)
        self.conv_2 = Conv2dGroupNJ(in_channels=6,out_channels=16,kernel_size=5, cuda=is_cuda)
        # create max pool layers
        self.p = MaxPool2d(kernel_size=2, stride=2)
        # create dropout layer, used only for convolution layers

        self.relu = ReLU()
        self.flatten = FlattenLayer(i_dense_dim)
        self.fc_1 = LinearGroupNJ(in_features=i_dense_dim, out_features=512, cuda=is_cuda)
        self.fc_2 = LinearGroupNJ(in_features=512,out_features=256, cuda=is_cuda)
        self.fc_3 = LinearGroupNJ(in_features=256,out_features=n_classes, cuda=is_cuda)

        # layers including kl_divergence
        self.layers = [self.conv_1, self.relu,  self.p, self.conv_2, self.relu, \
            self.p, self.flatten, self.fc_1, self.relu,self.fc_2, self.relu, self.fc_3]
        
        self.kl_list = [self.conv_1, self.conv_1, self.fc_1, self.fc_2, self.fc_3]

    def forward(self, x):

        for layer in self.layers:
                x = layer(x)
        return x
    
    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
