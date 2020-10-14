import torch
import torch.nn as nn

class General(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(General, self).__init__()
        self.out_planes = 64
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1_1  = conv1x1(channel * 2, channel)
        self.fc       = nn.Sequential(
                                conv1x1(channel * 2, channel),
                                nn.Sigmoid()
                                )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.max_pool(x).view(b, c)
        o = torch.cat((y,z),1).view(b, c * 2, 1, 1)
        o = self.fc(o)
        return x * o

class Cluster(nn.Module):
    def __init__(self, groups = 64):
        super(Cluster, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.tanh     = nn.Tanh()
        self.sig      = nn.Sigmoid()
        kernel        = [[0.125,0.125,0.125],
                            [0.125,-1,0.125],
                            [0.125,0.125,0.125]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight_1 = nn.Parameter(data=kernel, requires_grad=False)
        
    def forward(self, x): 
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        b2, c2 = t.size()
        t = t.view(1, -1, b2, c2) 
        t2 = F.conv2d(t, self.weight_1, padding=1)  
        t = t + t2   
        t = t.view(b2, c2) 
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)  
        return x
