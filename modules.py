import torch.nn as nn
import torch
import math

class Double13(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double13, self).__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, input):
        x1 = self.conva(input)
        x2 = self.convb(input)
        return x1 + x2

class ADConv(nn.Module): #非对称双卷积 Asymmetric Double Convolution
    def __init__(self, in_ch, out_ch):
        super(ADConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Double13(out_ch, out_ch))
    def forward(self, input):
        return self.conv(input)

class ADCG(nn.Module):
    def __init__(self, inp, oup, ratio=2, dw_size=3, relu=True):
        super(ADCG, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # ratio = oup / intrinsic
        new_channels = init_channels*(ratio-1)

        self.primary_conv = ADConv(inp, init_channels)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding=dw_size//2, groups=init_channels, bias=False), # groups 分组卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class DE(nn.Module): #Difference enhancement module 差异增强模块
    def __init__(self, in_planes):
        super(DE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input1, input2):
        diff = torch.sub(input2, input1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(diff))))
        att = self.sigmoid(avg_out + max_out)
        feature1 = input1 * att + input1
        feature2 = input2 * att + input2
        different = torch.sub(feature2, feature1)
        return feature1, feature2, different

class SSA(nn.Module): #spatial-spectral attention module 空-谱注意力模块
    def __init__(self, in_dim):
        super(SSA, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.pool1 = nn.AvgPool2d(16)
        self.pool2 = nn.AvgPool2d(8)
        self.pool3 = nn.AvgPool2d(4)
        self.pool4 = nn.AvgPool2d(2)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q1 = self.pool1(x)
        pq1 = q1.view(q1.size(0), -1, q1.size(2) * q1.size(3))
        q2 = self.pool2(x)
        pq2 = q2.view(q2.size(0), -1, q2.size(2) * q2.size(3))
        q3 = self.pool3(x)
        pq3 = q3.view(q3.size(0), -1, q3.size(2) * q3.size(3))
        q4 = self.pool4(x)
        pq4 = q4.view(q4.size(0), -1, q4.size(2) * q4.size(3))
        proj_query = torch.cat((pq1, pq2, pq3, pq4), dim=2)

        k1 = self.pool1(x)
        pk1 = k1.view(k1.size(0), -1, k1.size(2) * k1.size(3))
        k2 = self.pool2(x)
        pk2 = k2.view(k2.size(0), -1, k2.size(2) * k2.size(3))
        k3 = self.pool3(x)
        pk3 = k3.view(k3.size(0), -1, k3.size(2) * k3.size(3))
        k4 = self.pool4(x)
        pk4 = k4.view(k4.size(0), -1, k4.size(2) * k4.size(3))
        proj_key = torch.cat((pk1, pk2, pk3, pk4), dim=2)

        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1)) # B C C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  #[0]只返回最大值的数值，不返回索引
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1) # B C N     -1表示剩下的值的个数一起构成一个维度

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out