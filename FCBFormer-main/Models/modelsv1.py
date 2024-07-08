from functools import partial
import numpy as np

import torch
from torch import nn
# from .pvt_v2 import *
from Models import pvt_v2
# import Models.pvt_v2
from timm.models.vision_transformer import _cfg
from Models.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from Models.A_nommer import NomMerAttn
from Models.swin_transformer_v2 import SwinTransformerV2
from thop import profile

import torch.nn.functional as F
from timm.models.layers import DropPath


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, in_channels,out_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,groups=in_channels)  # depthwise conv
        # self.dconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        # self.dropout = nn.Dropout(0.5)                 #过拟合
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # (1,384,8,8)
        x = self.dwconv(x)  # (1,192,8,8)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        # x = self.dropout(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W) # 2,32,256,256
        input = self.dconv(input)
        x = input + self.drop_path(x)
        return x

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            Block(in_channels,out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            Block(out_channels, out_channels),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
          #2,32,352,352
        return h

class TB1(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, num_classes=64,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super(TB1, self).__init__()
        self.swinunet = SwinTransformerSys(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                               embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                               attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                               use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes)
    def forward(self, x):
        x = self.swinunet(x)      # (2,32,192,192)
        return x

class TB2(nn.Module):
    def __init__(self,img_size=256):
        super(TB2, self).__init__()
        self.nommerr = NomMerAttn(input_size=img_size)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.GELU()
        )

    def forward(self, x):
        x = self.nommerr(x)
        x = self.conv1(x)
        return x

class TB3(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super(TB3, self).__init__()
        self.swinv2 = SwinTransformerV2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                              embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                              attn_drop_rate=attn_drop_rate,
                              drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                              use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes
                              )
        self.up_tosize = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.PH1 = nn.Sequential(
            RB(768, 64), RB(64, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(64, 1, kernel_size=1)
        )
        self.PH2 = nn.Sequential(
            RB(384, 64), RB(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(64, 1, kernel_size=1)
        )
        self.PH3 = nn.Sequential(
            RB(192, 64), RB(64, 64),RB(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.PH4 = nn.Sequential(
            RB(96, 64), RB(64, 64),RB(64, 64),RB(64, 64)
        )
        self.PH5 = nn.Sequential(
            RB(256, 64), RB(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )



    def forward(self,x):
        x1 = self.swinv2(x)                     #2,8,8,768
        x1 = x1[0].permute(0, 3, 1, 2)

        x1 = self.PH1(x1)

        x2 = self.swinv2(x)[1][2]
        B, L, C = x2.shape
        x2 = x2.contiguous().view(B, 16, 16, 384)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.PH2(x2)

        x3 = self.swinv2(x)[1][1]
        B, L, C = x3.shape
        x3 = x3.contiguous().view(B, 32, 32, 192)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.PH3(x3)

        x4 = self.swinv2(x)[1][0]
        B, L, C = x4.shape
        x4 = x4.contiguous().view(B, 64, 64, 96)
        x4 = x4.permute(0, 3, 1, 2)
        x4 = self.PH4(x4)
        # x2 = self.up_tosize[0](x1)              #  384
        # x3 = self.up_tosize[1](x1)              #  192
        # x4 = self.up_tosize[2](x1)              #  96

        x = torch.cat([x1,x2,x3,x4],dim=1)                  # x4 = 2,768,64,64
        # x = self.PH5(x)                         # 2,1,256,256
        return x

class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB( 64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l
                   # 2,64,88,88
        return l




class FCBFormer(nn.Module):
    def __init__(self, size=256):

        super().__init__()
        # self.TB1 = TB1(img_size=256)
        # self.TB2 = TB2(img_size=256)
        # self.TB3 = TB3(img_size=256)
        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(32, 32), nn.Conv2d(32, 1, kernel_size=1)
        )
        # self.up_tosize = nn.Upsample(size=size)
        # self.PH1 = nn.Sequential(
        #     RB(32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        # )
    def forward(self, x):
        # x1 = self.TB3(x)      #64
        x2 = self.FCB(x)     #32
        # x1 = self.up_tosize(x1)
        # x = torch.cat((x1, x2), dim=1)
        out = self.PH(x2)
        return out


if __name__ == '__main__':
    pass
    device = torch.device("cuda:0")
    model = FCBFormer().to(device)
    # model = TB3().to(device)
    img = torch.rand([1,3,256,256]).to(device)
    result = model(img)
    print(result)

    #测试flops
    # model = FCBFormer()
    # dummy_input = torch.randn(1, 3, 256, 256)
    # flops, params = profile(model.cuda(), (dummy_input.cuda(),))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


