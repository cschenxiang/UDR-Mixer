import torch
import torch.nn as nn
import torch.nn.functional as F

# FFT
class FFML(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(FFML, self).__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1)
        )
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape

        dim = 1

        y = torch.fft.rfft2(x, norm=self.norm)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)

        y = self.main_fft(y_f)
        # y = y*y_f

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y*x

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)

# Gobal feature
class SFRL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        # print(x.shape,"a")
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # print(y.shape, "b")
        # b c h w -> b h w c
        y = self.act1(self.conv1(y)).permute(0, 2, 3, 1)
        # b h w c -> b w c h
        y = self.act2(self.conv2(y)).permute(0, 2, 3, 1)
        # b w c h -> b c h w
        y = self.act3(self.conv3(y)).permute(0, 2, 3, 1)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y

class SFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim)

        self.gobal = SFRL(dim)

        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)

        y = self.gobal(y)


        y = self.fc(self.norm2(y)) + y
        return y

class FFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.fft = FFML(dim)

        # Feedforward layer
        self.fc = FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)

        y = self.fft(y)


        y = self.fc(self.norm2(y)) + y
        return y
    
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class UDRMixer(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()

        self.to_feat1=nn.Conv2d(3, dim // 4, 3, 1, 1)
        self.to_feat2=nn.PixelUnshuffle(upscaling_factor)
        out_dim = upscaling_factor * dim

        self.feats1 = SFMB(dim, ffn_scale)
        self.feats2 = SFMB(dim, ffn_scale)
        self.down1_2 = Downsample(dim*1)
        self.feats3 = SFMB(out_dim*1, ffn_scale)
        self.feats4 = SFMB(out_dim*1, ffn_scale)
        self.feats6 = SFMB(out_dim*1, ffn_scale)
        self.feats7 = SFMB(out_dim*1, ffn_scale)
        self.up2_1 = Upsample(int(dim*2))
        self.feats8 = SFMB(dim, ffn_scale)
        self.feats9 = SFMB(dim, ffn_scale)
        self.to_img1=nn.Conv2d(dim, 48, 3, 1, 1)
        self.to_img2=nn.PixelShuffle(4)

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=False)

        #fft
        self.to_feat_fft = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            #ResBlock(32, 3, 1, 1)
        )
        # out_dim_fft = upscaling_factor_fft * 16
        self.feats_fft = nn.Sequential(*[FFMB(32, ffn_scale) for _ in range(4)])
        self.down_fft = nn.PixelUnshuffle(8)
        self.reduce_chan_fft = nn.Conv2d(2176, 128, 3, 1, 1)


    def forward(self, x):
        ########################################################
        x_fft =x
        x_fft = self.to_feat_fft(x_fft)
        x_fft = self.feats_fft(x_fft)
        x_fft = self.down_fft(x_fft)
        ########################################################
        x = F.interpolate(x, scale_factor=1/2, mode='bicubic', align_corners=False)
        x = self.to_feat1(x)
        x = self.to_feat2(x)
        x1 =x
        x = self.feats1(x)
        x = self.feats2(x)
        x_skip = x
        x = self.down1_2(x)
        x = self.feats3(x)
        x = self.feats4(x)
        x = torch.cat([x, x_fft], dim=1)
        x = self.reduce_chan_fft(x)
        x = self.feats6(x)
        x = self.feats7(x)
        x = self.up2_1(x)
        x = torch.cat([x,x_skip],1)
        x = self.reduce_chan_level2(x)
        x = self.feats8(x)
        x = self.feats9(x)
        x = self.to_img1(x+x1)
        x = self.to_img2(x)

        return x


if __name__ == '__main__':
    input = torch.rand(1, 3, 1024, 1024)
    model = UDRMixer(dim=64, n_blocks=8, ffn_scale=2.0)
    #output = model(input)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    flops = FlopCountAnalysis(model, input)

    print("FLOPs: ", flops.total())
    print(parameter_count_table(model))