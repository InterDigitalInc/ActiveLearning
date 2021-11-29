# alexnet for CIFAR 10
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Dropout2d
import torch.nn.functional as F

# class AlexnetDropout(nn.Module):
#
#     def __init__(self,n_classes, i_channel, d_rate=[0.25, 0.5]):
#         super().__init__()
#         if i_channel == 1:
#             i_dense_dim = 2304
#         else:
#             # change this
#             i_dense_dim = 4096
#         self.drop_train = False
#         self.d_rate = d_rate
#         self.conv_1 = Conv2d(in_channels=i_channel, out_channels=64, kernel_size=3, stride=1, padding=2)
#         self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=2)
#         self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
#         self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
#         self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#
#         # fully connected layers
#         self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
#         self.fc_2 = Linear(in_features=512, out_features=256)
#         self.fc_3 = Linear(in_features=256, out_features=n_classes)
#
#     def forward(self, x):
#
#         out = F.relu(self.conv_1(x), inplace=True)
#         out = F.max_pool2d(F.dropout(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)
#         out = F.relu(self.conv_2(out), inplace=True)
#         out = F.max_pool2d(F.dropout(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)
#
#         # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
#         out = F.relu(self.conv_3(out), inplace=True)
#         out = F.dropout(out,p=self.d_rate[0], training=self.drop_train)
#         out = F.relu(self.conv_4(out), inplace=True)
#         out = F.dropout(out,p=self.d_rate[0], training=self.drop_train)
#         out = F.relu(self.conv_5(out), inplace=True)
#         out = F.dropout(out,p=self.d_rate[0], training=self.drop_train)
#         out = F.max_pool2d(out, kernel_size=3, stride=2)
#         out = out.view(out.size(0),-1)
#         out = F.relu(self.fc_1(out))
#         out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
#         out = self.fc_2(out)
#         out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
#         out = self.fc_3(out)
#         return out

class AlexnetDropout(nn.Module):

    def __init__(self,n_classes, i_channel, d_rate=[0.25, 0.5]):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 2304
        else:
            # change this
            i_dense_dim = 4096
        self.drop_train = False
        self.d_rate = d_rate
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=2)
        self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # fully connected layers
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512, out_features=256)
        self.fc_3 = Linear(in_features=256, out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)

        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.dropout2d(out,p=self.d_rate[0], training=self.drop_train)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.dropout2d(out,p=self.d_rate[0], training=self.drop_train)
        out = F.relu(self.conv_5(out), inplace=True)
        out = F.dropout2d(out,p=self.d_rate[0], training=self.drop_train)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = self.fc_2(out)
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = self.fc_3(out)
        return out


class Alexnet(nn.Module):

    def __init__(self,n_classes, i_channel):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 2304
        else:
            # change this
            i_dense_dim = 4096
        self.drop_train = False
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv_2 = Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=2)
        self.conv_3 = Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv_4 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # fully connected layers
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512, out_features=256)
        self.fc_3 = Linear(in_features=256, out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out, kernel_size=2)
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

# this architecture is similar to https://keras.io/examples/cifar10_cnn/
class AlexnetLightDropout(nn.Module):

    def __init__(self,n_classes, i_channel , d_rate=[0.25, 0.5]):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 256
        self.drop_train = False
        self.d_rate = d_rate
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv_3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        # fully connected layers
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512, out_features=256)
        self.fc_3 = Linear(in_features=256, out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0]),kernel_size=2)
        out = F.relu(self.conv_2(out), inplace=True)
        out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0]),kernel_size=2)

        # out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)
        # out = F.relu(self.conv_2(out), inplace=True)
        # out = F.max_pool2d(F.dropout2d(out, p=self.d_rate[0], training=self.drop_train),kernel_size=2)

        # what does inplace do: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        out = F.relu(self.conv_3(out), inplace=True)
        out = F.dropout2d(out,p=self.d_rate[0], training=self.drop_train)
        out = F.relu(self.conv_4(out), inplace=True)
        out = F.dropout2d(out,p=self.d_rate[0], training=self.drop_train)
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc_1(out))
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = self.fc_2(out)
        out = F.dropout(out, p=self.d_rate[1], training=self.drop_train)
        out = self.fc_3(out)
        return out


# this architecture is similar to https://keras.io/examples/cifar10_cnn/
class AlexnetLight(nn.Module):

    def __init__(self,n_classes, i_channel):
        super().__init__()
        if i_channel == 1:
            i_dense_dim = 256
        else:
            # change this
            i_dense_dim = 256
        self.conv_1 = Conv2d(in_channels=i_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv_3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        # fully connected layers
        self.fc_1 = Linear(in_features=i_dense_dim, out_features=512)
        self.fc_2 = Linear(in_features=512, out_features=256)
        self.fc_3 = Linear(in_features=256, out_features=n_classes)

    def forward(self, x):

        out = F.relu(self.conv_1(x), inplace=True)
        out = F.max_pool2d(out,kernel_size=2)
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