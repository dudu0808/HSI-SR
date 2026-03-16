#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=int(kernel_size // 2), bias=bias)


def get_Ep(MSI):  # 1*3*128*128 to 1*1*128*128
    input_tensor = 0.299 * MSI[:, 0, :, :] + 0.587 * MSI[:, 1, :, :] + 0.114 * MSI[:, 2, :,:]
    input_tensor = input_tensor.unsqueeze(1)

    # 定义Sobel卷积核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to('cuda')
    sobel_y = sobel_x.transpose(2, 3).to('cuda')

    # 执行卷积操作
    edge_x = F.conv2d(input_tensor, sobel_x, padding=1)
    edge_y = F.conv2d(input_tensor, sobel_y, padding=1)

    # 计算边缘强度
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    return edge_magnitude


class Conv_3D_Block(nn.Module):
    def __init__(self):
        super(Conv_3D_Block, self).__init__()
        self.body = nn.Conv3d(1, 1, (3, 3, 3), 1, (1, 1, 1), bias=True)

    def forward(self, x):
        x = self.body(x.unsqueeze(1))
        return x.squeeze(1)




class Res3DBlock(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(), res_scale=1):
        super(Res3DBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(1, n_feats, (3, 1, 1), 1, (1, 0, 0), bias=bias),
                                  act,
                                  nn.Conv3d(n_feats, 1, (1, 3, 3), 1, (0, 1, 1), bias=bias)
                                  )
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x.unsqueeze(1)) + x.unsqueeze(1)
        return x.squeeze(1)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class state_transmission(nn.Module):  # 3个块
    def __init__(self, n_features=64, act=nn.ReLU(), reduction=16):
        super(state_transmission, self).__init__()
        block = []
        block.append(nn.Conv2d(n_features, n_features, 3, 1, 1))
        # block.append(act)
        # block.append(nn.Conv2d(n_features, n_features, 3, 1, 1))
        # block.append(CALayer(n_features, reduction))
        self.body = nn.Sequential(*block)

    def forward(self, x):
        res = self.body(x)
        return res + x


class RCAB1(nn.Module):
    def __init__(self, n_feat=4, bn=False, res_scale=1):
        super(RCAB1, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat, 3, 1, 1), nn.ReLU(False)]
        # CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        return res

class Ep_Mul(nn.Module):
    def __init__(self, n_feat=4, reduction=16):
        super(Ep_Mul,self).__init__()
        Ep_multiply = [nn.Conv2d(n_feat, n_feat * reduction, kernel_size=1),
                       nn.Conv2d(n_feat * reduction, n_feat, kernel_size=1)]
        self.Ep_multiply_body = nn.Sequential(*Ep_multiply)
    def forward(self,ep):
        ep_mul=self.Ep_multiply_body(ep)
        return ep_mul

class Ep_Add(nn.Module):
    def __init__(self, n_feat=4, reduction=16):
        super(Ep_Add,self).__init__()
        Ep_add = [nn.Conv2d(n_feat, n_feat * reduction, kernel_size=1),
                       nn.Conv2d(n_feat * reduction, n_feat, kernel_size=1)]
        self.Ep_multiply_body = nn.Sequential(*Ep_add)
    def forward(self,ep):
        ep_add=self.Ep_multiply_body(ep)
        return ep_add

class RCAB2(nn.Module):
    def __init__(self, n_feat=4, res_scale=1):
        super(RCAB2, self).__init__()
        self.feat = n_feat
        self.conv = nn.Conv2d(self.feat, self.feat, 3, 1, 1)
        # self.BatchNorm = nn.BatchNorm2d(n_feat)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv(x)
        # res = self.BatchNorm(res)
        # res = self.body(x).mul(self.res_scale)
        return res

class RCAB(nn.Module):
    def __init__(self, n_feat=4, bn=False, res_scale=1,reduction=4):
        super(RCAB, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat*4, 3, 1, 1), nn.ReLU(True),nn.Conv2d(n_feat*4, n_feat, 3, 1, 1),CALayer(n_feat, reduction)]
        # CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x)
        res = self.body(x)*self.res_scale
        return res+x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)
        return x

class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super(Upsampler, self).__init__()
        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class FEN_HSI(nn.Module):
    def __init__(self):
        super(FEN_HSI, self).__init__()
        body = []
        body.append(nn.Conv2d(64, 4, 3, 1, 1))
        body.append(nn.ReLU())
        body.append(nn.Conv2d(4, 4, 3, 1, 1))
        body.append(nn.ReLU())
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class FEN_MSI(nn.Module):
    def __init__(self):
        super(FEN_MSI, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.Conv2d(4, 4, 3, 1, 1))
            body.append(nn.ReLU())
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)

# class SpatialAttentionBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SpatialAttentionBlock, self).__init__()
#
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels * reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels * reduction, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#
#         y = self.pool(x).view(b, c)
#         y = self.fc(y)
#         y=y.view(b, c, 1, 1)
#
#         return x * y.expand_as(x)
#
# class AttentionBlockModule(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=16):
#         super(AttentionBlockModule, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.attention = SpatialAttentionBlock(out_channels, reduction)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.attention(x)
#         x = self.relu(x)
#         return x
#
# class SpatialAttentionModel(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModel, self).__init__()
#
#         self.block1 = AttentionBlockModule(4, 4)
#         self.block2 = AttentionBlockModule(4, 4)
#         self.block3 = AttentionBlockModule(4, 4)
#         self.block4 = AttentionBlockModule(4, 4)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x
class SpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(SpatialAttentionBlock, self).__init__()
        self.embed_dim=embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Reshape to (seq_len, batch_size, embed_dim)
        h=x.size(2)
        w=x.size(3)
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1).contiguous()

        # Apply self-attention
        x, _ = self.attention(x, x, x)

        # Reshape back to (batch_size, seq_len, embed_dim)
        x = x.permute(1, 2, 0).contiguous()
        x=x.view(x.size(0),self.embed_dim,h,w)

        return x
class NL_BLOCK(nn.Module):
    def __init__(self,patch_size):
        super(NL_BLOCK, self).__init__()
        self.point_wise1 = nn.Conv2d(4, 2, 1)
        self.point_wise2 = nn.Conv2d(4, 2, 1)
        self.point_wise3 = nn.Conv2d(4, 2, 1)
        self.point_wise4 = nn.Conv2d(2, 4, 1)
        self.patch_size=patch_size


    def forward(self, x):#2*4*128*128
        b,c,h,w=x.size()
        x1 = self.point_wise1(x)#2*2*128*128
        x2 = self.point_wise2(x)#2*2*128*128
        x3 = self.point_wise3(x)#2*2*128*128
        avg_pooled_data1 = F.avg_pool2d(x1, kernel_size=16, stride=16)#2*2*8*8
        max_pooled_data1 = F.max_pool2d(x1, kernel_size=16, stride=16)#2*2*8*8
        # 将两者相加
        attention_x1= avg_pooled_data1 + max_pooled_data1
        avg_pooled_data2 = F.avg_pool2d(x2, kernel_size=16, stride=16)  # 2*2*8*8
        max_pooled_data2 = F.max_pool2d(x2, kernel_size=16, stride=16)  # 2*2*8*8
        # 将两者相加
        attention_x2 = avg_pooled_data2+ max_pooled_data2
        # 将两个张量重塑为2*64*2和2*2*64
        reshaped_tensor1 = attention_x1.view(b, 2, -1).permute(0, 2, 1)#2*64*2
        reshaped_tensor2 = attention_x2.view(b, 2, -1)#2*2*64
        # 相乘得到2*64*64的张量
        attention = torch.matmul(reshaped_tensor1, reshaped_tensor2)
        attention = nn.Softmax(dim=1)(attention)
        unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        # 将另一个2*512*64的张量乘以上面的张量
        reshaped_tensor3=unfold(x3)#2*512*64
        result = torch.matmul(reshaped_tensor3, attention)#2*512*64
        fold=nn.Fold(output_size=(h, w), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        # 最后将结果重塑为2*2*128*128
        result = fold(result)
        result = self.point_wise4(result)

        return result


class ShuffleDown(nn.Module):
    def __init__(self, scale):
        super(ShuffleDown, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, cin, hin, win = x.size()
        cout = cin * self.scale ** 2
        hout = hin // self.scale
        wout = win // self.scale
        output = x.view(b, cin, hout, self.scale, wout, self.scale)
        output = output.permute(0, 1, 5, 3, 2, 4).contiguous()
        output = output.view(b, cout, hout, wout)
        return output
class SpectralAttention(nn.Module):
    def __init__(self,n_feats):
        super(SpectralAttention,self).__init__()
        self.n_feats=n_feats
        self.conv_q = default_conv(n_feats,n_feats//2)
        self.conv_k = default_conv(n_feats,1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_v = default_conv(n_feats//2,n_feats,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        q=self.conv_q(x).flatten(2)
        k = self.conv_k(x).flatten(2).transpose(1,2)
        k = self.softmax(k)

        v= torch.bmm(q,k)
        v=self.conv_v(v.unsqueeze(-1))
        v=self.sigmoid(v)
        y=x*v
        return  y

class Net(nn.Module):
    def __init__(self, scale, seq_len, devices):
        super(Net, self).__init__()
        self.n_feats = 64
        self.kernel_size = 3
        self.devices = devices
        self.sub = 4
        self.scale = scale

        self.g = 8
        self.conv_pointwise1 = nn.Conv2d(3, 32, 1)
        self.conv_pointwise2 = nn.Conv2d(1, 32, 1)
        self.conv_pointwise3 = nn.Conv2d(8, 4, 1)

        self.layer1 = default_conv(self.sub + self.n_feats + self.sub * self.scale ** 2, self.n_feats, self.kernel_size)

        self.out_layer1 = default_conv(self.sub, self.sub, self.kernel_size)
        self.out_layer2 = default_conv(self.n_feats, self.n_feats, self.kernel_size)

        n_a = 8
        # body1 = [RCAB() for _ in range(n_a)]
        # self.RCAB1=RCAB1()
        # self.Ep_mul=Ep_Mul()
        # self.Ep_add=Ep_Add()
        # self.RCAB2=RCAB2()
        RCAB_list=[RCAB() for _ in range(2)]
        self.RCAB=nn.Sequential(*RCAB_list)

        body2 = [state_transmission() for _ in range(1)]
        self.RB2 = nn.Sequential(*body2)
        self.FEN_HSI = FEN_HSI()
        self.FEN_MSI = FEN_MSI()
        self.up = Upsampler(self.scale, self.n_feats)
        # self.block = ConvMod(64)
        # self.NL_BLOCK = NL_BLOCK(patch_size=16)
        # self.Attention= SpatialAttentionBlock(self.sub,2)
        self.down = ShuffleDown(self.scale)
        self.spectral_att = SpectralAttention(31)
        self.act = nn.ReLU()
        self.Con_3D = Conv_3D_Block()
        n_b = 1
        body2 = [Res3DBlock(seq_len) for _ in range(n_b)]
        self.body2 = nn.Sequential(*body2)

    def forward(self, x, y):  # x:1*31*16*16 y:1*3*64*64
        out = []  # 1*4*64*64
        B, C, h, w = x.shape  # 1*31*16*16

        p = self.sub - C % self.sub
        ini_x = torch.zeros(B, p, h, w).to(self.devices)
        x = torch.cat([x, ini_x], 1)  # x:1*32*16*16
        # Ep = get_Ep(y)  # 1*1*64*64
        y = self.conv_pointwise1(y)  # 1*32*64*64
        # Ep = self.conv_pointwise2(Ep)  # 1*32*64*64

        s = torch.zeros(B, self.n_feats, h, w).to(self.devices)  # 1*64*16*16
        sr = torch.zeros(B, self.sub * self.scale ** 2, h, w).to(self.devices)  # 1*64*16*16
        x_ilr_list = list(torch.chunk(x, self.g, 1))
        y_imsi_list = list(torch.chunk(y, self.g, 1))
        # Ep_imsi_list = list(torch.chunk(Ep, self.g, 1))
        for i in range(self.g):
            x_ilr = x_ilr_list[i]  # 1*4*16*16
            y_imsi = y_imsi_list[i]  # 1*4*64*64
            # ep_i = Ep_imsi_list[i]  # 1*4*64*64
            fl = self.act(self.layer1(torch.cat([s, sr, x_ilr], dim=1)))  # 1*64*16*16
            # after_attention=self.block(fl)
            upp = self.up(fl)  # 1*4*64*64
            s = self.RB2(fl)  # 1*64*16*16
            sr = torch.cat([self.FEN_HSI(upp), self.FEN_MSI(y_imsi)], 1)  # 1*8*64*64
            sr = self.conv_pointwise3(sr) #1*4*64*64
            # Ep_mul = self.Ep_mul(ep_i)
            # Ep_add = self.Ep_add(ep_i)
            # for _ in range(8):
            #     res=self.RCAB2(self.RCAB1(sr))
            #     # res1 = self.RCAB1(sr)
            #     # res2=Ep_mul*res1+Ep_add
            #     # res=self.RCAB2(res2)
            #     sr=sr+res
            #sr = self.NL_BLOCK(sr)#1*4*64*64
            #sr=self.Attention(sr)
            sr=self.RCAB(sr)
            # upp=F.interpolate(h1,(h*self.scale,w*self.scale))
            sr = self.out_layer1(sr) +   y_imsi
            #sr = self.out_layer1(sr) +  F.interpolate(x_ilr, (h * self.scale, w * self.scale))
            out.append(sr)
            sr = self.down(sr)

        out = torch.cat(out[:], 1)[:, 0:C, :, :]

        out = self.body2(out)
        # out= self.spectral_att(out)
        out = self.Con_3D(out)
        out = out
        return out
