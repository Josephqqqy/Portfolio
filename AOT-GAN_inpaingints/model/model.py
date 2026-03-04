import torch
import torch.nn as nn
import torch.optim as optim
from model.Common import init_weights
from torchsummary import summary
rates = [1,2,4,8]   #膨胀卷积率
block_num = 6  #AOT模块数

class InpaintGenerator(nn.Module):
    def __init__(self):  
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),  #变为(4,518,518)
            nn.Conv2d(4, 64, 7),    #变为(64,512,512)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), #变为(128,256,256)
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), #变为(256,128,128)
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        init_weights(self.encoder)
        init_weights(self.middle)
        init_weights(self.decoder)
    

    def forward(self, x):   #x是(4,512,512) x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates       #膨胀卷积率[1,2,4,8]
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),  #填充0，01，02，04，08
                nn.Sequential(
                    nn.ReflectionPad2d(rate),   #对输入进行反射填充
                    nn.Conv2d(dim, dim // len(rates), 3, padding=0, dilation=rate), #变为(原*1/4,原,原)，包括分裂，转换（膨胀）
                    nn.ReLU(True)  
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)  #将四个子核的输出在第一维度（通道数）上拼接
        out = self.fuse(out)   #对拼接后的特征图进行融合操作
        mask = my_layer_norm(self.gate(x))  #计算门值
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask #输入特征x1和学习到的残差特征x2与g加权求和

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d( 3, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
        init_weights(self.conv)
    def forward(self, x):
        feat = self.conv(x)
        return feat