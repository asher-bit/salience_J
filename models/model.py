import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torch.nn import init
import numpy as np
from models.swin_T import Block
from einops import rearrange 
from einops.layers.torch import Rearrange
from models.SCAB import SCAB

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Salicon(nn.Module):
    def __init__(self,
                 embed_dim=96, depths=[6, 6, 6, 6], head_dim=[6, 6, 6, 6],
                 window_size=10, drop_path_rate=0.1,):
        super(Salicon, self).__init__()

        self.num_layers = 2

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(32, 96, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96 // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(96 // 4, 96 // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(96 // 4, 96, 3, 1, 1)
        )
        self.conv4 = nn.Conv2d(96, 96, 3, 2, 1)

        self.mlp =  Mlp(96, 96, act_layer=nn.GELU, drop=0.)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Block(input_dim=embed_dim,
                          output_dim=embed_dim,
                          head_dim=head_dim[i_layer],
                          window_size=window_size,
                          drop_path=dpr[i_layer],
                          type='W',
                        )
            self.layers.append(layer)
        self.channel_layers_lm = nn.ModuleList()
        c_num_heads = [4, 4, 4, 4]
        for i_layer in range(self.num_layers):
            layer = SCAB(embed_dim//3,
                         num_heads=c_num_heads[i_layer],
                         ffn_expansion_factor=4,
                         bias=False,
                         LayerNorm_type='WithBias')
            self.channel_layers_lm.append(layer)

        self.channel_layers_rm = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SCAB(embed_dim//3,
                         num_heads=c_num_heads[i_layer],
                         ffn_expansion_factor=4,
                         bias=False,
                         LayerNorm_type='WithBias')
            self.channel_layers_rm.append(layer)

        self.mlps = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mlp(embed_dim, embed_dim, embed_dim, act_layer=nn.GELU, drop=0.)
            self.mlps.append(layer)
        self.conv_me1 = nn.Sequential(nn.Conv2d(2304, 1024, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv_me2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv_me3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True))

        self.multi_exposure = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.Sal_map = nn.Conv2d(256, 1, (1, 1))
        self.Sigmoid = nn.Sigmoid()

    def attn(self, x):
        #
        x = Rearrange('b c h w -> b h w c')(x)
        x1, x2, x3 = x.chunk(3, dim=3)
        x = self.mlp(x)
        for layer, layer_lm, layer_rm, mlp in zip(self.layers, self.channel_layers_lm, self.channel_layers_rm, self.mlps):
            x = layer(x)
            x_1, x_2, x_3 = x.chunk(3, dim=3)
            x_lm = layer_lm(x2+x_2, x1+x_1)
            x_rm = layer_rm(x2+x_2, x3+x_3)
            fuse = torch.cat((x_lm, x2 + x_2, x_rm), dim=3)
            x = mlp(fuse)
            # x = Rearrange('b h w c -> b c h w')(x)
            # x = self.conv4(x)
            # x = Rearrange('b c h w -> b h w c')(x)

        x = Rearrange('b h w c -> b c h w')(x)
        return x


    def forward(self, raw_img, img_me1, img_me2, img_me3, img_me4, img_me5, img_me6, img_me7, img_me8, img_me9):
        x_1 = torch.cat([img_me1, img_me2, img_me3], dim=1)
        x_2 = torch.cat([img_me4, img_me5, img_me6], dim=1)
        x_3 = torch.cat([img_me7, img_me8, img_me9], dim=1)
        x_1 = self.conv1_1(x_1)
        x_2 = self.conv1_2(x_2)
        x_3 = self.conv1_3(x_3)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = self.conv3(self.attn(x) + self.conv2(x_2))

        x = self.multi_exposure(x)

        sal_map = self.Sal_map(x)
        sal_map = self.Sigmoid(sal_map)
        #print('sal_map:',sal_map.size())
        return sal_map


