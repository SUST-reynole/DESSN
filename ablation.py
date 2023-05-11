import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)
class Double15(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double15, self).__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, input):
        x1 = self.conva(input)
        x2 = self.convb(input)
        return x1 + x2



class MSConv(nn.Module):
    """multi-scale strip Convolution"""
    def __init__(self, in_ch, out_ch):
        super(MSConv, self).__init__()
        self.conva1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conva2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convb1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convb2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convc1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convc2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        xa1 = self.conva1(input)
        xa2 = self.conva2(input)
        xa = xa1+xa2
        xb1 = self.convb1(input)
        xb2 = self.convb2(input)
        xb = xb1 + xb2
        xc1 = self.convc1(input)
        xc2 = self.convc2(input)
        xc = xc1 + xc2
        return xa + xb + xc



class MSUnet(nn.Module): ##用1*5和5*1卷积代替后三组的2个3*3卷积
    """multi-scale strip convolution"""
    def __init__(self, in_ch, out_ch):
        super(MSUnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = MSConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = MSConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # self.ReGate1 = ReGate(64, 64, reduction=16, bn_momentum=0.0003)
        # self.ReGate2 = ReGate(128, 128, reduction=16, bn_momentum=0.0003)
        # self.ReGate3 = ReGate(256, 256, reduction=16, bn_momentum=0.0003)
        # self.ReGate4 = ReGate(512, 512, reduction=16, bn_momentum=0.0003)
        #
    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        c5 = torch.sub(c5_1, c5_2)
        d5 = self.drop5(c5)
        up_6 = self.up6(d5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = self.up9(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
        return out

class MPM(nn.Module):
    """Multi-scale pooling module"""
    def __init__(self):
        super(MPM, self).__init__()
        self.pool1 = nn.AvgPool2d(16)
        self.pool2 = nn.AvgpPool2d(8)
        self.pool3 = nn.AvgPool2d(4)
        self.pool4 = nn.AvgPool2d(2)
        # self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, input1, input2):
        # m_batchsize1, C1, width1, height1 = input1.size()
        p1 = self.pool1(input1)
        p2 = self.pool2(input1)
        p3 = self.pool3(input1)
        p4 = self.pool4(input1)
        proj_key1 = p1.view(p1.size(0), -1, p1.size(2) * p1.size(3))
        proj_key2 = p2.view(p2.size(0), -1, p2.size(2) * p2.size(3))
        proj_key3 = p3.view(p3.size(0), -1, p3.size(2) * p3.size(3))
        proj_key4 = p4.view(p4.size(0), -1, p4.size(2) * p4.size(3))
        proj_key = torch.cat((proj_key1, proj_key2, proj_key3, proj_key4), dim=2)
        pa = self.pool1(input2)
        pb = self.pool2(input2)
        pc = self.pool3(input2)
        pd = self.pool4(input2)
        proj_value1 = pa.view(pa.size(0), -1, pa.size(2) * pa.size(3))
        proj_value2 = pb.view(pb.size(0), -1, pb.size(2) * pb.size(3))
        proj_value3 = pc.view(pc.size(0), -1, pc.size(2) * pc.size(3))
        proj_value4 = pd.view(pd.size(0), -1, pd.size(2) * pd.size(3))
        proj_value = torch.cat((proj_value1, proj_value2, proj_value3, proj_value4), dim=2)
        return proj_key, proj_value

# class SPAM(nn.Module):
#     """ spatial pyramid attention module
#     """
#     def __init__(self, in_dim, ds=8, activation=nn.ReLU):
#         super(SPAM, self).__init__()
#         self.chanel_in = in_dim
#         self.key_channel = self.chanel_in //8
#         self.activation = activation
#         self.ds = ds  #
#         # self.pool = nn.AvgPool2d(self.ds)
#         # print('ds: ', ds)
#         self.mpm = MPM
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, input):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         # x = self.pool(input)
#         m_batchsize, C, width, height = input.size()
#         proj_query = self.query_conv(input).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B C (N)/(ds*ds)  .permute():转置，维度转换
#         # proj_key = self.key_conv(input).view(m_batchsize, -1, width * height)  # B C (W*H)/(ds*ds)
#         proj_key, proj_value = self.mpm(self.key_conv(input), self.value_conv(input))
#         energy = torch.bmm(proj_query, proj_key)  # transpose check 矩阵乘
#         energy = (self.key_channel**-.5) * energy
#
#         attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
#
#         # proj_value = self.value_conv(input).view(m_batchsize, -1, width * height)  # B C N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#         # out = F.interpolate(out, [width*self.ds, height*self.ds])
#         out = out + input
#         return out

# class MSPAUnet(nn.Module): ##用1*5和5*1卷积代替后三组的2个3*3卷积
#     """multi-scale strip convolution pyramid attention"""
#     def __init__(self, in_ch, out_ch):
#         super(MSPAUnet, self).__init__()
#         self.att = SPAM(1024, ds=8, activation=nn.ReLU)
#         self.conv1 = DoubleConv(in_ch, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.drop1 = nn.Dropout2d(0.1)
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.drop2 = nn.Dropout2d(0.1)
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)
#         self.drop3 = nn.Dropout2d(0.2)
#         self.conv4 = MSConv(256, 512)
#         self.pool4 = nn.MaxPool2d(2)
#         self.drop4 = nn.Dropout2d(0.2)
#         self.conv5 = MSConv(512, 1024)
#         self.drop5 = nn.Dropout2d(0.3)
#         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv6 = DoubleConv(1024, 512)
#         self.drop6 = nn.Dropout2d(0.2)
#         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv7 = DoubleConv(512, 256)
#         self.drop7 = nn.Dropout2d(0.2)
#         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv8 = DoubleConv(256, 128)
#         self.drop8 = nn.Dropout2d(0.1)
#         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv9 = DoubleConv(128, 64)
#         self.drop9 = nn.Dropout2d(0.1)
#         self.conv10 = nn.Conv2d(64, out_ch, 1)
#
#         # self.ReGate1 = ReGate(64, 64, reduction=16, bn_momentum=0.0003)
#         # self.ReGate2 = ReGate(128, 128, reduction=16, bn_momentum=0.0003)
#         # self.ReGate3 = ReGate(256, 256, reduction=16, bn_momentum=0.0003)
#         # self.ReGate4 = ReGate(512, 512, reduction=16, bn_momentum=0.0003)
#         #
#     def forward(self, x1, x2):
#         c1_1 = self.conv1(x1)
#         c1_2 = self.conv1(x2)
#         c1 = torch.sub(c1_1, c1_2)
#         p1_1 = self.pool1(c1_1)
#         p1_2 = self.pool1(c1_2)
#         d1_1 = self.drop1(p1_1)
#         d1_2 = self.drop1(p1_2)
#         c2_1 = self.conv2(d1_1)
#         c2_2 = self.conv2(d1_2)
#         c2 = torch.sub(c2_1, c2_2)
#         p2_1 = self.pool1(c2_1)
#         p2_2 = self.pool1(c2_2)
#         d2_1 = self.drop1(p2_1)
#         d2_2 = self.drop2(p2_2)
#         c3_1 = self.conv3(d2_1)
#         c3_2 = self.conv3(d2_2)
#         c3 = torch.sub(c3_1, c3_2)
#         p3_1 = self.pool3(c3_1)
#         p3_2 = self.pool3(c3_2)
#         d3_1 = self.drop3(p3_1)
#         d3_2 = self.drop3(p3_2)
#         c4_1 = self.conv4(d3_1)
#         c4_2 = self.conv4(d3_2)
#         c4 = torch.sub(c4_1, c4_2)
#         p4_1 = self.pool4(c4_1)
#         p4_2 = self.pool4(c4_2)
#         d4_1 = self.drop4(p4_1)
#         d4_2 = self.drop4(p4_2)
#         c5_1 = self.conv5(d4_1)
#         c5_2 = self.conv5(d4_2)
#         c5_1 = self.att(c5_1)
#         c5_2 = self.att(c5_2)
#         c5 = torch.sub(c5_1, c5_2)
#         # d5 = self.drop5(c5)
#         up_6 = self.up6(c5)
#         merge6 = torch.cat([up_6, c4], dim=1)
#         c6 = self.conv6(merge6)
#         d6 = self.drop6(c6)
#         up_7 = self.up7(d6)
#         merge7 = torch.cat([up_7, c3], dim=1)
#         c7 = self.conv7(merge7)
#         d7 = self.drop7(c7)
#         up_8 = self.up8(d7)
#         merge8 = torch.cat([up_8, c2], dim=1)
#         c8 = self.conv8(merge8)
#         d8 = self.drop8(c8)
#         up_9 = self.up9(d8)
#         merge9 = torch.cat([up_9, c1], dim=1)
#         c9 = self.conv9(merge9)
#         d9 = self.drop9(c9)
#         c10 = self.conv10(d9)
#         out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
#         return out

class SAM(nn.Module):
    """ spatial attention module
    """
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(SAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        # self.pool = nn.AvgPool2d(self.ds)
        # print('ds: ', ds)
        # self.mpm = MPM
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # x = self.pool(input)
        m_batchsize, C, width, height = input.size()
        proj_query = self.query_conv(input).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B C (N)/(ds*ds)  .permute():转置，维度转换
        proj_key = self.key_conv(input).view(m_batchsize, -1, width * height)  # B C (W*H)/(ds*ds)
        # proj_key, proj_value = self.mpm(self.key_conv(input), self.value_conv(input))
        energy = torch.bmm(proj_query, proj_key)  # transpose check 矩阵乘
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(input).view(m_batchsize, -1, width * height)  # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # out = F.interpolate(out, [width*self.ds, height*self.ds])
        out = out + input
        return out

class MSAUnet(nn.Module): ##用1*5和5*1卷积代替后三组的2个3*3卷积
    """multi-scale strip convolution attention"""
    def __init__(self, in_ch, out_ch):
        super(MSAUnet, self).__init__()
        self.att = SAM(1024, ds=8, activation=nn.ReLU)
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = MSConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = MSConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # self.ReGate1 = ReGate(64, 64, reduction=16, bn_momentum=0.0003)
        # self.ReGate2 = ReGate(128, 128, reduction=16, bn_momentum=0.0003)
        # self.ReGate3 = ReGate(256, 256, reduction=16, bn_momentum=0.0003)
        # self.ReGate4 = ReGate(512, 512, reduction=16, bn_momentum=0.0003)
        #
    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        c5_1 = self.att(c5_1)
        c5_2 = self.att(c5_2)
        c5 = torch.sub(c5_1, c5_2)
        # d5 = self.drop5(c5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = self.up9(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
        return out

class ChannelAttention(nn.Module):
    """channel attention module"""
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# class STAM(nn.Module):
#     """ spatial-temporal attention module
#     """
#     def __init__(self, in_dim, ds=8, activation=nn.ReLU):
#         super(STAM, self).__init__()
#         self.chanel_in = in_dim
#         self.key_channel = self.chanel_in //8
#         self.activation = activation
#         self.ds = ds  #
#         # self.pool = nn.AvgPool2d(self.ds)
#         # print('ds: ', ds)
#         # self.mpm = MPM
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, input1, input2):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         # x = self.pool(input)
#         height0 = input1.shape[3]
#         x = torch.cat((input1, input2), 3)
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B C (N)/(ds*ds)  .permute():转置，维度转换
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B C (W*H)/(ds*ds)
#         # proj_key, proj_value = self.mpm(self.key_conv(input), self.value_conv(input))
#         energy = torch.bmm(proj_query, proj_key)  # transpose check 矩阵乘
#         # energy = (self.key_channel**-.5) * energy
#
#         attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
#
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B C N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#         # out = F.interpolate(out, [width*self.ds, height*self.ds])
#         out = self.gamma*out + x
#         # return out
#         return out[:, :, :, 0:height0], out[:, :, :, height0:]

class STAM(nn.Module):
    """ spatial-temporal attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(STAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))   #nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input1, input2):
        height0 = input1.shape[3]
        x = torch.cat((input1, input2), 3)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B C (N)/(ds*ds)  .permute():转置，维度转换
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B C (W*H)/(ds*ds)
        # proj_key, proj_value = self.mpm(self.key_conv(input), self.value_conv(input))
        energy = torch.bmm(proj_query, proj_key)  # transpose check 矩阵乘

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out[:, :, :, 0:height0], out[:, :, :, height0:]

class STAUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(STAUnet, self).__init__()
        self.att = STAM(1024)
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = MSConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = MSConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        c5_1, c5_2 = self.att(c5_1, c5_2)
        c5 = torch.sub(c5_1, c5_2)
        # d5 = self.drop5(c5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = self.up9(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1) # B C N
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B N C
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  #[0]只返回最大值的数值，不返回索引
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1) # B C N     -1表示剩下的值的个数一起构成一个维度

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class ACB(nn.Module): #非对称卷积块，3*3+1*3+3*1
    def __init__(self, in_ch, out_ch):
        super(ACB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.conv1(input)
        x3 = self.conv2(input)
        return x1 + x2 +x3

class ACUnet(nn.Module): #非对称卷积Unet
    def __init__(self, in_ch, out_ch):
        super(ACUnet, self).__init__()
        # self.att = STAM(1024)
        self.conv1 = ACB(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = ACB(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = ACB(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = ACB(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = ACB(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        # c5_1, c5_2 = self.att(c5_1, c5_2)
        c5 = torch.sub(c5_1, c5_2)
        d5 = self.drop5(c5)
        up_6 = self.up6(d5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = self.up9(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
        return out

class CPAM1(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CPAM1, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.pool1 = nn.AvgPool2d(16)
        # self.pool2 = nn.AvgPool2d(8)
        # self.pool3 = nn.AvgPool2d(4)
        # self.pool4 = nn.AvgPool2d(2)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q1 = self.pool1(x)
        pq1 = q1.view(q1.size(0), -1, q1.size(2) * q1.size(3))
        # q2 = self.pool2(x)
        # pq2 = q2.view(q2.size(0), -1, q2.size(2) * q2.size(3))
        # q3 = self.pool3(x)
        # pq3 = q3.view(q3.size(0), -1, q3.size(2) * q3.size(3))
        # q4 = self.pool4(x)
        # pq4 = q4.view(q4.size(0), -1, q4.size(2) * q4.size(3))
        # proj_query = torch.cat((pq1, pq2, pq3, pq4), dim=2)
        # proj_query = x.view(m_batchsize, C, -1) # B C N
        k1 = self.pool1(x)
        pk1 = k1.view(k1.size(0), -1, k1.size(2) * k1.size(3))
        # k2 = self.pool2(x)
        # pk2 = k2.view(k2.size(0), -1, k2.size(2) * k2.size(3))
        # k3 = self.pool3(x)
        # pk3 = k3.view(k3.size(0), -1, k3.size(2) * k3.size(3))
        # k4 = self.pool4(x)
        # pk4 = k4.view(k4.size(0), -1, k4.size(2) * k4.size(3))
        # proj_key = torch.cat((pk1, pk2, pk3, pk4), dim=2)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B N C
        # energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1)) # B C C
        energy = torch.bmm(pq1, pk1.permute(0, 2, 1))  # B C C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  #[0]只返回最大值的数值，不返回索引
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1) # B C N     -1表示剩下的值的个数一起构成一个维度

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CPAM2(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CPAM2, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.pool1 = nn.AvgPool2d(16)
        self.pool2 = nn.AvgPool2d(8)
        # self.pool3 = nn.AvgPool2d(4)
        # self.pool4 = nn.AvgPool2d(2)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q1 = self.pool1(x)
        pq1 = q1.view(q1.size(0), -1, q1.size(2) * q1.size(3))
        q2 = self.pool2(x)
        pq2 = q2.view(q2.size(0), -1, q2.size(2) * q2.size(3))
        # q3 = self.pool3(x)
        # pq3 = q3.view(q3.size(0), -1, q3.size(2) * q3.size(3))
        # q4 = self.pool4(x)
        # pq4 = q4.view(q4.size(0), -1, q4.size(2) * q4.size(3))
        proj_query = torch.cat((pq1, pq2), dim=2)
        # proj_query = x.view(m_batchsize, C, -1) # B C N
        k1 = self.pool1(x)
        pk1 = k1.view(k1.size(0), -1, k1.size(2) * k1.size(3))
        k2 = self.pool2(x)
        pk2 = k2.view(k2.size(0), -1, k2.size(2) * k2.size(3))
        # k3 = self.pool3(x)
        # pk3 = k3.view(k3.size(0), -1, k3.size(2) * k3.size(3))
        # k4 = self.pool4(x)
        # pk4 = k4.view(k4.size(0), -1, k4.size(2) * k4.size(3))
        proj_key = torch.cat((pk1, pk2), dim=2)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B N C
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1)) # B C C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  #[0]只返回最大值的数值，不返回索引
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1) # B C N     -1表示剩下的值的个数一起构成一个维度
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CPAM3(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CPAM3, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.pool1 = nn.AvgPool2d(16)
        self.pool2 = nn.AvgPool2d(8)
        self.pool3 = nn.AvgPool2d(4)
        # self.pool4 = nn.AvgPool2d(2)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q1 = self.pool1(x)
        pq1 = q1.view(q1.size(0), -1, q1.size(2) * q1.size(3))
        q2 = self.pool2(x)
        pq2 = q2.view(q2.size(0), -1, q2.size(2) * q2.size(3))
        q3 = self.pool3(x)
        pq3 = q3.view(q3.size(0), -1, q3.size(2) * q3.size(3))
        # q4 = self.pool4(x)
        # pq4 = q4.view(q4.size(0), -1, q4.size(2) * q4.size(3))
        proj_query = torch.cat((pq1, pq2, pq3), dim=2)
        # proj_query = x.view(m_batchsize, C, -1) # B C N
        k1 = self.pool1(x)
        pk1 = k1.view(k1.size(0), -1, k1.size(2) * k1.size(3))
        k2 = self.pool2(x)
        pk2 = k2.view(k2.size(0), -1, k2.size(2) * k2.size(3))
        k3 = self.pool3(x)
        pk3 = k3.view(k3.size(0), -1, k3.size(2) * k3.size(3))
        # k4 = self.pool4(x)
        # pk4 = k4.view(k4.size(0), -1, k4.size(2) * k4.size(3))
        proj_key = torch.cat((pk1, pk2, pk3), dim=2)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B N C
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1)) # B C C
        # energy = torch.bmm(pq1, pk1.permute(0, 2, 1))  # B C C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  #[0]只返回最大值的数值，不返回索引
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1) # B C N     -1表示剩下的值的个数一起构成一个维度

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class AConv(nn.Module): #非对称卷积组
    def __init__(self, in_ch, out_ch):
        super(AConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Double13(out_ch, out_ch))

    def forward(self, input):
        return self.conv(input)

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

class CPAM1Unet(nn.Module): #不同的CPAM，卷积都是ADConv
    def __init__(self, in_ch, out_ch):
        super(CPAM1Unet, self).__init__()
        # self.att = UPCM()
        self.att2 = CPAM1(2048)

        self.conv1 = AConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = AConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = AConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = AConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = AConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = AConv(1536, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = AConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = AConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = AConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        # c1_1, c1_2 = self.att(c1_1, c1_2)
        c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        # c2_1, c2_2 = self.att(c2_1, c2_2)
        c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        # c3_1, c3_2 = self.att(c3_1, c3_2)
        c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        # c4_1, c4_2 = self.att(c4_1, c4_2)
        c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        # c5_1, c5_2 = self.att(c5_1, c5_2)
        c5 = torch.cat((c5_1, c5_2), dim=1)
        c5 = self.att2(c5)

        d5 = self.drop5(c5)
        up_6 = self.up6(d5) #1024
        merge6 = torch.cat([up_6, c4], dim=1) #1024+512=1536
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = self.up7(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = self.up8(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = self.up9(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10) #因为BCELoss（二进制交叉熵）需要输入的是Sigmoid (0,1)
        return out