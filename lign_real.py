import torch.nn as nn
import torch



class FE(nn.Module):
    def __init__(self,
                 in_channels, out_channels1, out_channels2, out_channels3, out_channels4,
                 ksize=3, stride=1, pad=1):

        super(FE, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 1, 1, 0),
            nn.ReLU(inplace=True))

        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True))


        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3, out_channels3, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        self.body4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels4, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.body1(x)

        out2 = self.body2(x)

        out3 = self.body3(x)

        out4 = self.body4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out


class OneBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OneBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class EBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Down_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Up_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Encoder(nn.Module):  # EB
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.en1 = EBlock(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = Down_Block(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3


class Decoder(nn.Module):  # EB
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.en1 = Up_Block(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = EBlock(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3


class Mid(nn.Module):
    def __init__(self, num, in_ch, out_ch):
        super(Mid, self).__init__()
        self.block = self._make_layer(Block, num, in_ch, out_ch)

    def _make_layer(self, block, block_num, in_channels, out_channels, **kwargs):
        layers = []

        for _ in range(1, block_num+1):
            layers.append(block(in_channels, out_channels, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)

        return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fe = FE(39, 32, 32, 32, 32)  # 13 for gray, 39 for color
        self.one = OneBlock(128, 64)
        self.en1 = Encoder(64, 64)
        self.en2 = Encoder(64, 128)
        self.en3 = Encoder(128, 256)
        self.en4 = Encoder(256, 512)
        self.mid = Block(512, 1024)  # 512-1024
        self.mid1 = Block(1024, 512)
        self.de4 = Decoder(512, 256)
        self.de3 = Decoder(256, 128)
        self.de2 = Decoder(128, 64)
        self.de1 = Decoder(64, 64)  # 64
        self.sp = nn.Conv2d(64, 1, 3, 1, 1)

        self.head1 = Block(128, 64)
        self.en11 = Encoder(64, 64)
        self.en22 = Encoder(64, 128)
        self.en33 = Encoder(128, 256)
        self.en44 = Encoder(256, 512)
        self.mid00 = Block(512, 1024)

        self.mid11 = Block(1024, 512)
        self.de44 = Decoder(512, 256)
        self.de33 = Decoder(256, 128)
        self.de22 = Decoder(128, 64)
        self.de11 = Decoder(64, 64)
        self.tail = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)  # 1 for gray

        self.weight = nn.Conv2d(1,1,1,1,0,bias=False)
        self.weight1 = nn.Conv2d(1,1,1,1,0,bias=False)

        self.m1 = Mid(4, 64, 64)
        self.m2 = Mid(3, 128, 128)
        self.m3 = Mid(2, 256, 256)
        self.m4 = Mid(1, 512, 512)

    def forward(self, x):
        fe = self.fe(x)
        head = self.one(fe)
        en1 = self.en1(head)
        en2 = self.en2(en1)
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        mid = self.mid(en4)
        mid1 = self.mid1(mid)

        de4 = self.de4(mid1+en4)
        de3 = self.de3(de4+en3)
        de2 = self.de2(de3+en2)
        de1 = self.de1(de2+en1)
        sp = self.sp(de1)
        mask = torch.sigmoid(10 * self.weight(sp.detach())) - self.weight1(torch.sigmoid(sp.detach()+10))

        head1 = self.head1(fe)
        en11 = self.en11(head1+mask)
        en22 = self.en22(en11)
        en33 = self.en33(en22)
        en44 = self.en44(en33)

        mid00 = self.mid00(en44)
        mid11 = self.mid11(mid00)
        de44 = self.de44(mid11+self.m4(en44))
        de33 = self.de33(de44+self.m3(en33))
        de22 = self.de22(de33+self.m2(en22))
        de11 = self.de11(de22+self.m1(en11))
        res = self.tail(de11) + x[:,0:3,:,:]
        # + x[:,0:3,:,:] for color
        return sp, res
