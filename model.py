import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

def MatrixShapeAlignment(Matrix1, Matrix2):
    # input is CHW
    diffY = Matrix1.size()[2] - Matrix2.size()[2]
    diffX = Matrix1.size()[3] - Matrix2.size()[3]

    Matrix2 = F.pad(Matrix2, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])  # 当pad有四个参数，代表对(batchSize, channel, height, width)中最后两个维度扩充，pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)
    return Matrix2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.identity_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_channel))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.double_conv(x)
        identity = self.identity_conv(x)
        out = out + identity
        out = self.relu(out)
        return out


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        out = self.max_pool(x)
        out = self.double_conv(out)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 当pad有四个参数，代表对最后两个维度扩充，pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x

class Up_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, attention):
        x1 = self.upsample(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 当pad有四个参数，代表对最后两个维度扩充，pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)
        x = torch.cat([x2, x1], dim=1)
        x = x * attention
        x = self.double_conv(x)
        return x

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        batch, channle, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channle)
        attention = self.fc(y).view(batch, channle, 1, 1)
        return x * attention.expand_as(x)

class RPN(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.double_conv = DoubleConv(input_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.tail1_SE_Block = SE_Block(64)
        self.tail1_final_conv = nn.Conv2d(64, output_channel, kernel_size=1)
        self.tail1_sigmoid = nn.Sigmoid()

        self.tail2_DoubleConv_1 = DoubleConv(64, 64)
        self.tail2_SE_Block_1 = SE_Block(128)
        self.tail2_DoubleConv_2 = DoubleConv(128, 128)
        self.tail2_SE_Block_2 = SE_Block(256)
        self.tail2_final_conv = nn.Conv2d(256, output_channel, kernel_size=1)
        self.tail2_sigmoid = nn.Sigmoid()

        self.tail2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.double_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)

        tail1 = self.tail1_SE_Block(up4)
        tail1 = self.tail1_final_conv(tail1)
        RSM = self.tail1_sigmoid(tail1)
        
        tail2 = torch.mul(x, RSM)
        tail2 = self.tail2_DoubleConv_1(tail2)
        tail2 = torch.cat([tail2, up4], dim=1)
        tail2 = self.tail2_SE_Block_1(tail2)
        tail2 = self.tail2_DoubleConv_2(tail2)
        upsample_up1 = self.tail2_upsample(up3)
        tail2 = torch.cat([tail2, upsample_up1], dim=1)
        tail2 = self.tail2_SE_Block_2(tail2)
        tail2 = self.tail2_final_conv(tail2)
        PFM = self.tail2_sigmoid(tail2)

        return RSM, PFM

if __name__ == '__main__':
    model = RPN(3, 1)
    model.to(device='cuda:0')
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 256, 256))
    print('finish')