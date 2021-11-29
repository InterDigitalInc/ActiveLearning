import torch.nn as nn
from torch.nn import Conv2d, Linear, MaxPool2d
import torch.nn.functional as F

# we use a modfied version Resnet-5 architecture https://engmrk.com/lenet-5-a-classic-cnn-architecture/ with just 1-Conv layer

# class Lenet5Dropout(nn.Module):
#
#     def __init__(self,n_classes, i_channel, d_rate=[0.25,0.4]):
#         super().__init__()
#         if i_channel == 1:
#             i_dense_dim = 256
#         else:
#             # change this
#             i_dense_dim = 400
#         self.d_rate = d_rate
#         self.drop_train = False
#         # resnet has 3 input channels (RGB) and 6 filters with kernel size 5
#         self.conv_1 = Conv2d(in_channels=i_channel, out_channels=6, kernel_size=5)
#         self.conv_2 = Conv2d(in_channels=6,out_channels=16,kernel_size=5)
#         # create max pool layers
#         self.mx_pool = MaxPool2d(kernel_size=2, stride=2)
#         '''create the 3 fully connected layers, the input size at fc_1 is determined by
#         (W_1 - F)S + 1, where W_1 is the width of the channel from conv_2 layer. '''
#         self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
#         self.fc_2 = Linear(in_features=512,out_features=256)
#         self.fc_3 = Linear(in_features=256,out_features=n_classes)
#
#     def forward(self, x):
#
#         out = F.relu(self.conv_1(x))
#         out = self.mx_pool(F.dropout(out, p=self.d_rate[0], training=self.drop_train))
#         out = F.relu(self.conv_2(out))
#         out = self.mx_pool(F.dropout(out, p=self.d_rate[0], training=self.drop_train))
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc_1(out))
#         out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
#         out = F.relu(self.fc_2(out))
#         out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
#         out = self.fc_3(out)
#         return out

class Lenet5Dropout(nn.Module):

    def __init__(self,n_classes, i_channel, d_rate=[0.25,0.4]):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 400
        self.d_rate = d_rate
        self.drop_train = False
        # resnet has 3 input channels (RGB) and 6 filters with kernel size 5
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=6, kernel_size=5)
        self.conv_2 = Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        # create max pool layers
        self.mx_pool = MaxPool2d(kernel_size=2, stride=2)
        '''create the 3 fully connected layers, the input size at fc_1 is determined by 
        (W_1 - F)S + 1, where W_1 is the width of the channel from conv_2 layer. '''
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512,out_features=256)
        self.fc_3 = Linear(in_features=256,out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = self.mx_pool(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train))
        out = F.relu(self.conv_2(out))
        out = self.mx_pool(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = F.relu(self.fc_2(out))
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = self.fc_3(out)
        return out

# lenet 5 without dropouts
class Lenet5(nn.Module):

    def __init__(self,n_classes, i_channel):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 400
        self.drop_train = False
        # resnet has 3 input channels (RGB) and 6 filters with kernel size 5
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=6, kernel_size=5)
        self.conv_2 = Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        # create max pool layers
        self.mx_pool = MaxPool2d(kernel_size=2, stride=2)
        '''create the 3 fully connected layers, the input size at fc_1 is determined by
        (W_1 - F)S + 1, where W_1 is the width of the channel from conv_2 layer. '''
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512,out_features=256)
        self.fc_3 = Linear(in_features=256,out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x))
        out = self.mx_pool(out)
        out = F.relu(self.conv_2(out))
        out = self.mx_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
