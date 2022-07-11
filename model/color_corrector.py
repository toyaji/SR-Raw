from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import namedtuple
from torch.nn.modules.activation import PReLU
from torch.nn.modules.container import ModuleList, Sequential
from torch.nn.modules.dropout import Dropout


class ColorCorrector(nn.Module):
    def __init__(self,
                 n_feats,
                 reduction=2):
        
        super(ColorCorrector, self).__init__()

        # set conv layers for each steps
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.conv3 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.conv4 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(n_feats, int(n_feats/2), 6, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(n_feats, int(n_feats/2), 6, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(n_feats, int(n_feats/2), 6, 2, 2)
        self.deconv4 = nn.ConvTranspose2d(n_feats, int(n_feats/2), 6, 2, 2)

        self.merge1= nn.Conv2d(n_feats, int(n_feats/2), 3, 1, 1)
        self.merge2= nn.Conv2d(n_feats, int(n_feats/2), 3, 1, 1)
        self.merge3= nn.Conv2d(n_feats, int(n_feats/2), 3, 1, 1)
        self.merge4= nn.Conv2d(n_feats, int(n_feats/2), 3, 1, 1)

        self.relu = nn.PReLU()
        self.avg = nn.AvgPool2d(2)

        self.tail = Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.PReLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.relu(self.conv1(x))
        x2 = self.avg(x1)
        x2 = self.relu(self.conv2(x2))
        x3 = self.avg(x2)
        x3 = self.relu(self.conv2(x3))
        x4 = self.avg(x3)
        x4 = self.relu(self.conv2(x4))
        x5 = self.avg(x4)  
         
        y4 = torch.cat([self.deconv4(x5), self.merge4(x4)], axis=1)
        y4 = self.relu(y4)
        y3 = torch.cat([self.deconv3(y4), self.merge3(x3)], axis=1)
        y3 = self.relu(y3)
        y2 = torch.cat([self.deconv3(y3), self.merge3(x2)], axis=1)
        y2 = self.relu(y2)
        y1 = torch.cat([self.deconv3(y2), self.merge3(x1)], axis=1)
        y1 = self.relu(y1)

        y = self.tail(y1)

        theta = torch.matmul(x.view(b, c, h*w), y.view(b, c, h*w).transpose(1, 2)) # b x c x c
        output = torch.matmul(theta, x.reshape(b, c, h*w))
        output = output.view(b, c, h, w)
        return output







if __name__ == '__main__':

    example = torch.zeros(2, 128, 480, 480)
    net = ColorCorrector(128, 2, )
    result = net(example)
    print(result.shape)