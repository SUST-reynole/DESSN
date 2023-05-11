import torch.nn as nn
import torch

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


class UPCM(nn.Module): #空间注意
    def __init__(self):
        super(UPCM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input1, input2):
        diff = torch.sub(input1, input2)
        avg_out = torch.mean(diff, dim=1, keepdim=True)
        max_out, _ = torch.max(diff, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        att = self.sigmoid(x)
        return input1 * att + input1, input2 * att + input2

#**********************商量之后的通道注意模块****************************************************
class DEM(nn.Module): #Difference enhancement module 差异增强模块
    """channel attention module"""
    def __init__(self, in_planes):
        super(DEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input1, input2):
        diff = torch.sub(input1, input2)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(diff))))
        att = self.sigmoid(avg_out + max_out)
        feature1 = input1 * att + input1
        feature2 = input2 * att + input2
        different = torch.sub(feature1, feature2)
        return feature1, feature2, different


class SSAM(nn.Module): #spatial-spectral attention module 空-谱注意力模块
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(SSAM, self).__init__()
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

class Net(nn.Module): #terminal module 最终的模块
    def __init__(self, in_ch, out_ch):
        super(Net, self).__init__()
        self.att1 = DEM(64)
        self.att2 = DEM(128)
        self.att3 = DEM(256)
        self.att4 = DEM(512)
        self.att5 = DEM(1024)
        self.att = SSAM(2048)

        self.conv1 = ADConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = ADConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = ADConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = ADConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = ADConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = ADConv(1536, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = ADConv(512, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = ADConv(256, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = ADConv(128, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x1, x2):
        c1_1 = self.conv1(x1)
        c1_2 = self.conv1(x2)
        c1_1, c1_2, c1 = self.att1(c1_1, c1_2)
        # c1 = torch.sub(c1_1, c1_2)
        p1_1 = self.pool1(c1_1)
        p1_2 = self.pool1(c1_2)
        d1_1 = self.drop1(p1_1)
        d1_2 = self.drop1(p1_2)
        c2_1 = self.conv2(d1_1)
        c2_2 = self.conv2(d1_2)
        c2_1, c2_2, c2 = self.att2(c2_1, c2_2)
        # c2 = torch.sub(c2_1, c2_2)
        p2_1 = self.pool1(c2_1)
        p2_2 = self.pool1(c2_2)
        d2_1 = self.drop1(p2_1)
        d2_2 = self.drop2(p2_2)
        c3_1 = self.conv3(d2_1)
        c3_2 = self.conv3(d2_2)
        c3_1, c3_2, c3 = self.att3(c3_1, c3_2)
        # c3 = torch.sub(c3_1, c3_2)
        p3_1 = self.pool3(c3_1)
        p3_2 = self.pool3(c3_2)
        d3_1 = self.drop3(p3_1)
        d3_2 = self.drop3(p3_2)
        c4_1 = self.conv4(d3_1)
        c4_2 = self.conv4(d3_2)
        c4_1, c4_2, c4 = self.att4(c4_1, c4_2)
        # c4 = torch.sub(c4_1, c4_2)
        p4_1 = self.pool4(c4_1)
        p4_2 = self.pool4(c4_2)
        d4_1 = self.drop4(p4_1)
        d4_2 = self.drop4(p4_2)
        c5_1 = self.conv5(d4_1)
        c5_2 = self.conv5(d4_2)
        c5_1, c5_2, c5 = self.att5(c5_1, c5_2)
        C5 = torch.cat((c5_1, c5_2), dim=1)

        C5 = self.att(C5)

        d5 = self.drop5(C5)
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