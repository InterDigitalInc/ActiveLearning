import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from os.path import expanduser
import torch
from torch import nn
from sklearn.datasets import load_boston, fetch_california_housing
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection  import train_test_split

HOME = expanduser('~')
D_PTH = HOME + '/Google Drive/DataRepo'
# D_PTH = '/run/user/20110'


class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df,is_normalize=False):
    # self.X = torch.FloatTensor(df.loc[:, df.columns != 'label'].values)
    self.X = df.loc[:, df.columns != 'target']
    if is_normalize:
        self.X = (self.X-self.X.mean())/self.X.std()
    self.X = torch.FloatTensor(self.X.values)
    self.y = torch.FloatTensor(df.target.values)
    self.shape = self.X.shape

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)

class Config:
        
            
    def get_data(self,name):

        if name == 'mnist':
            n_classes, i_channel, i_dim, train_d, test = self.get_mnist()
            return (n_classes, i_channel, i_dim, train_d, test)
        if name == 'fmnist':
            n_classes, i_channel, i_dim, train_d, test = self.get_fashion_mnist()
            return (n_classes, i_channel, i_dim, train_d, test)
        if name == 'cifar10':
            n_classes, i_channel, i_dim, train_d, test = self.get_cifar10()
            return (n_classes, i_channel, i_dim, train_d, test)
        elif name == 'cifar100':
            n_classes, i_channel, i_dim, train_d, test = self.get_cifar100()
            return (n_classes, i_channel, i_dim, train_d, test)
        elif name == 'boston_housing' or name == "cali_housing":
            n_classes, i_channel, i_dim, train_d, test = self.get_regression_data(name)
            return (n_classes, i_channel, i_dim, train_d, test)

    # this is the new version of the method get_init_params, under progress..
    def get_init_params_v2_beta(self, d_name, m_name, is_retrain=0, typ='dropout'):
        name = "{}_{}_{}_{}".format(m_name,d_name,typ,is_retrain)
        # vgg_cifar10_ensemble_1 2 250
        param_d = {'lenet300-100_mnist_dropout_1':{'start_epoch':100, 'epoch':100, 'patience':100},
                    'lenet300-100_mnist_dropout_0':{'start_epoch':100, 'epoch':50, 'patience':100},
                   'lenet300-100_mnist_ensemble_1':{'start_epoch':100, 'epoch':100, 'patience':100},
                    'lenet300-100_mnist_ensemble_0':{'start_epoch':100, 'epoch':50, 'patience':100},
                    'lenet300-100_fmnist_dropout_1':{'start_epoch':100, 'epoch':100, 'patience':100},
                    'lenet300-100_fmnist_dropout_0':{'start_epoch':100, 'epoch':50, 'patience':50},
                    'lenet300-100_fmnist_ensemble':{'iepoch':100, 'patience':100, 'seed':10},
                    'lenet5_mnist_ensemble_0':{'start_epoch':100, 'epoch':50, 'patience':25},
                   'lenet5_mnist_dropout_0':{'start_epoch':60, 'epoch':50, 'patience':25},
                   'lenet5_fmnist_ensemble':{'iepoch':100, 'patience':25},
                   'lenet5_fmnist_dropout_1':{'iepoch':400, 'patience':25},
                   'lenet5_fmnist_dropout_0':{'start_epoch':100, 'epoch':50, 'patience':25},
                   'alexnet_cifar10_ensemble_1':{'start_epoch':70, 'epoch': 150, 'patience':50},
                   'alexnet_cifar10_ensemble_0':{'start_epoch':70, 'epoch': 50, 'patience':25},
                   'alexnetlight_cifar10_ensemble_1':{'start_epoch':25, 'epoch': 150, 'patience':50},
                   'alexnetlight_cifar10_ensemble_0':{'start_epoch':25, 'epoch': 50, 'patience':25},
                   'alexnet_cifar10_dropout_1':{'start_epoch':70, 'epoch':150, 'patience':50},
                   'alexnet_cifar10_dropout_0':{'start_epoch':70, 'epoch':50, 'patience':25},
                   'alexnetlight_cifar10_dropout_1':{'start_epoch':70, 'epoch':150, 'patience':50},
                   'alexnetlight_cifar10_dropout_0':{'start_epoch':70, 'epoch':50, 'patience':25},
                   'vgg_cifar10_ensemble_1':{'start_epoch':250,'epoch':250, 'patience':200},
                   'vgg_cifar10_ensemble_0':{'start_epoch':250,'epoch':50, 'patience':25},
                   'vgg_cifar10_dropout_1':{'start_epoch':200,'epoch':200, 'patience':50},
                   'vgg_cifar10_dropout_0':{'start_epoch':200,'epoch':50, 'patience':15},
                   'densenet_cifar100_ensemble_1':{'start_epoch':150,'epoch':200, 'patience':100},
                   'densenet_cifar100_ensemble_0':{'start_epoch':200,'epoch':50, 'patience':25},
                   'densenet_cifar100_dropout_1':{'start_epoch':150, 'epoch':200, 'patience':50},
                   'densenet_cifar100_dropout_0':{'start_epoch':150, 'epoch':50, 'patience':25},
                   'lenet300-100_boston_housing_dropout_0':{'start_epoch':20, 'epoch':100, 'patience':15},
                   'lenet300-100_boston_housing_dropout_1':{'start_epoch':20, 'epoch':100, 'patience':30},
                   'lenet300-100_boston_housing_ensemble_0':{'start_epoch':43, 'epoch':50, 'patience':15},
                   'lenet300-100_boston_housing_ensemble_1':{'start_epoch':43, 'epoch':100, 'patience':30},
                   'lenet300-100_cali_housing_dropout_0':{'start_epoch':20, 'epoch':100, 'patience':15},
                   'lenet300-100_cali_housing_dropout_1':{'start_epoch':20, 'epoch':100, 'patience':30},
                   'lenet300-100_cali_housing_ensemble_0':{'start_epoch':43, 'epoch':50, 'patience':15},
                   'lenet300-100_cali_housing_ensemble_1':{'start_epoch':43, 'epoch':100, 'patience':30}}
        param = param_d[name]
        e_a, e_b, p = param['start_epoch'], param['epoch'],  param['patience']
        return e_a, e_b, p

    # method to get starting parameters
    def get_init_params(self, d_name, m_name):
        # 'mnist':{'iepoch':100, 'patience':100, 'seed':10}
        param_d = {'fmnist_lenet5':{'iepoch':400, 'patience':25},
            'mnist':{'iepoch':100, 'patience':100, 'seed':10},
            'fmnist':{'iepoch':100, 'patience':100, 'seed':10},
            'cifar_10_vgg_adam':{'iepoch':150, 'patience':15},
            'cifar_100_vgg':{'iepoch':150, 'patience':25},
            'cifar_100_alexnet':{'iepoch':150, 'patience':50},
            'cifar_10_alexnet':{'iepoch':100, 'patience':25},
            'boston_housing':{'iepoch':150, 'patience':50},
            'cali_housing':{'iepoch':150, 'patience':50},
            'cifar_100_densenet_adam':{'iepoch':150, 'patience':25}}

        if (d_name == 'mnist' or d_name == 'fmnist') and (m_name == 'lenet5' or m_name == 'alexnet'):
            param = param_d['fmnist_lenet5']

        elif d_name == 'mnist' or d_name == 'fmnist' and m_name == 'lenet300-100':
            param = param_d['mnist']

        elif d_name == 'cifar10' and m_name == 'alexnet':
            param = param_d['cifar_10_alexnet']
            
        elif d_name == 'cifar10' and m_name == 'VGG':
            param = param_d['cifar_10_vgg_adam']

        elif d_name == 'cifar100' and m_name == 'VGG':
            param = param_d['cifar_100_vgg_adam']

        elif d_name == 'cifar100' and m_name == 'alexnet':
            param = param_d['cifar_10_vgg_adam']

        elif (d_name == 'cifar100' or d_name == 'cifar10') and m_name == 'densenet':
            param = param_d['cifar_100_densenet_adam']

        elif d_name == 'boston_housing':
            param = param_d['boston_housing']
        elif d_name == 'cali_housing':
            param = param_d['cali_housing']

        e, p = param['iepoch'], param['patience']
        return e, p

    def get_mnist(self):

        n_classes = 10
        i_channel = 1
        i_dim = 28
        transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
        train_d = MNIST(
            root=D_PTH, train=True,
            download=True, transform=transform_train)
        # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
        test_d = MNIST(
            root=D_PTH, train=False,
            download=True, transform=transform_test)
        return (n_classes, i_channel, i_dim, train_d, test_d)

    def get_fashion_mnist(self):

        n_classes = 10
        i_channel = 1
        i_dim = 28
        transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
        train_d = FashionMNIST(
            root=D_PTH, train=True,
            download=True, transform=transform_train)
        # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
        test_d = FashionMNIST(
            root=D_PTH, train=False,
            download=True, transform=transform_test)
        
        return (n_classes, i_channel, i_dim, train_d, test_d)

    # def get_cifar10(self):
    #
    #     n_classes = 10
    #     i_channel = 3
    #     i_dim = 32
    #
    #     transform_train = transforms.Compose([ transforms.ToTensor(), \
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    #
    #     transform_test = transforms.Compose([ transforms.ToTensor(), \
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    #
    #     train_d = CIFAR10(
    #         root=D_PTH, train=True,
    #         download=True, transform=transform_train)
    #     # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
    #     test_d = CIFAR10(
    #         root=D_PTH, train=False,
    #         download=True, transform=transform_test)
    #
    #     return (n_classes, i_channel, i_dim, train_d, test_d)
    
    def get_cifar10(self):

        n_classes = 10
        i_channel = 3
        i_dim = 32
        transform_train = transforms.Compose([ transforms.ToTensor(), \
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        # transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4), \
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([ transforms.ToTensor(), \
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_d = CIFAR10(
            root=D_PTH, train=True,
            download=True, transform=transform_train)
        # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
        test_d = CIFAR10(
            root=D_PTH, train=False,
            download=True, transform=transform_test)
        
        return (n_classes, i_channel, i_dim, train_d, test_d)

    
    def get_cifar100(self):

        n_classes = 100
        i_channel = 3
        i_dim = 32
        # transforms taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transform_train = transforms.Compose([ transforms.ToTensor(), \
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        # transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4), \
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        # transform_test = transforms.Compose([transforms.ToTensor(),\
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([ transforms.ToTensor(), \
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_d = CIFAR100(
            root=D_PTH, train=True,
            download=True, transform=transform_train)
        # self.train_d.data = self.train_d.data.reshape(len(self.train_d),self.i_dim)
        test_d = CIFAR100(
            root=D_PTH, train=False,
            download=True, transform=transform_test)
        
        return (n_classes, i_channel, i_dim, train_d, test_d)

    def get_regression_data(self,name):

        is_normalize = False
        if name == "boston_housing":
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df.columns = data.feature_names
            df['target'] = pd.Series(data.target)
            is_normalize = True
        elif name == "cali_housing":
            df = pd.read_csv("/home/vineeth/Documents/DataRepo/CaliforniaHousing/housing_pre_processed.csv")
            df.rename({"median_house_value":"target"}, axis='columns', inplace=True)

        i_channel = 1
        n_classes = 1
        r_state = np.random.randint(100)
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=r_state)
        train_d = custom_data_loader(df_train, is_normalize)
        test_d = custom_data_loader(df_test, is_normalize)
        i_dim = train_d.shape[1]

        return (n_classes, i_channel, i_dim, train_d, test_d)



class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
            for module in self.children():
                x = module(x)

            kl = 0.0
            for module in self.modules():
                if hasattr(module, 'kl_loss'):
                    kl = kl + module.kl_loss()

            return x, kl

class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

if __name__ == '__main__':

    Config().get_data("boston_housing")

