import torch
import torch.nn as nn
import torch.nn.functional as F
# how to apply dropout during testing using Apply method: https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/7

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_d = {'DenseDropout':['D',512,'D',256,'C'], 'Dense': [512, 256, 'C']}

class VGG(nn.Module):
    def __init__(self, n_classes, i_channel, vgg_name):

        if i_channel == 1:
            i_dim = 512
        else:
            # change this
            i_dim = 512

        super().__init__()
        self.features = self._make_layers(i_channel, cfg[vgg_name])
        self.classifier = self._make_dense_layers(i_dim, n_classes, cfg_d['Dense'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_dense_layers(self,i_dim, o_dim, cfg):
        layers = []
        for x in cfg:
            if x == 'C':
                layers += [nn.Linear(i_dim, o_dim)]
            else:
                layers += [nn.Linear(i_dim, x),
                nn.ReLU()]
                i_dim = x
        return nn.Sequential(*layers)

    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGGDropout(nn.Module):
    def __init__(self, n_classes, i_channel, vgg_name, d_rates=[0.25,0.5]):

        self.d_rates = d_rates
        if i_channel == 1:
            i_dim = 512
        else:
            # change this
            i_dim = 512

        super().__init__()
        self.features = self._make_layers(i_channel, cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        self.classifier = self._make_dense_layers(i_dim, n_classes, cfg_d['DenseDropout'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = F.dropout(self.classifier(out), p=0.5)
        out = self.classifier(out)
        return out

    def _make_dense_layers(self,i_dim, o_dim, cfg):
        layers = []
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout(p=self.d_rates[1])]
            elif x == 'C':
                layers += [nn.Linear(i_dim, o_dim)]
            else:
                layers += [nn.Linear(i_dim, x),
                nn.ReLU()]
                i_dim = x
        return nn.Sequential(*layers)

    # the right sequence of using dropout after batchnorm https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(),
                           nn.Dropout2d(p=self.d_rates[0])]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    
    i_channel = 3
    classes = 10
    net = VGG(classes, i_channel , 'VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
