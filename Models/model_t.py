from .Swin import SwinTransformer
from torch import nn, Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
import copy
from timm.models.layers import trunc_normal_
import numpy as np
import torch.nn.init as init

def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
def masked_fill(x, mask, value):
    """masked_fill"""
    y = torch.full(x.shape, value, x.dtype)
    return torch.where(mask, y, x)
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        ) 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            mask = mask.to(attn.device)
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class Block(nn.Module):
    """Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        epsilon=1e-6,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = (
            norm_layer(embed_dim, eps=epsilon) if norm_layer else Identity()
        )

    def forward(self, x):
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), "Input image size ({}*{}) doesn't match model ({}*{}).".format(
            H, W, self.img_size[0], self.img_size[1]
        )

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).permute((0, 2, 1))
        x = self.norm(x)
        return x
def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #----------encoder-------------
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        #----------encoder-------------

        #-----------------------------------------Expert_t-----------------------------------------------
        self.MHSA4_t = GMSA_ini(d_model=1024)
        self.MHSA3_t = GMSA_ini(d_model=512)
        self.MHSA2_t = GMSA_ini(d_model=256)
        self.MHSA1_t = GMSA_ini(d_model=128)
        #-----------------------------------------Expert_t-----------------------------------------------

        #----------decoder-------------
        self.convr4 = conv3x3_bn_relu(1024, 512)
        self.convr3 = conv3x3_bn_relu(512, 256)
        self.convr2 = conv3x3_bn_relu(256, 128)
        self.convr1 = conv3x3_bn_relu(128, 64)
        self.conv_dim = conv3x3(64, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        #----------decoder-------------

    def forward(self, t):
        #----------encoder-------------
        ft = self.t_swin(t) 
        #----------encoder-------------


        #-----------------------------------------Expert_t-----------------------------------------------
        flatten4 = ft[3].flatten(2).transpose(1, 2)
        flatten3 = ft[2].flatten(2).transpose(1, 2)
        flatten2 = ft[1].flatten(2).transpose(1, 2)
        t4 = self.MHSA4_t(flatten4, flatten4).view(flatten4.shape[0], int(np.sqrt(flatten4.shape[1])), int(np.sqrt(flatten4.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t3 = self.MHSA3_t(flatten3, flatten3).view(flatten3.shape[0], int(np.sqrt(flatten3.shape[1])), int(np.sqrt(flatten3.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t2 = self.MHSA2_t(flatten2, flatten2).view(flatten2.shape[0], int(np.sqrt(flatten2.shape[1])), int(np.sqrt(flatten2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t1 = ft[0]
        #-----------------------------------------Expert_t-----------------------------------------------

        #----------decoder-------------
        t4 = self.convr4(self.up2(t4))
        t3 = self.convr3(self.up2(t3 + t4))
        t2 = self.convr2(self.up2(t2 + t3))
        t1 = self.convr1(t1 + t2)
        out = self.up4(t1)
        out = self.conv_dim(out)
        #----------decoder-------------
        return out

    def load_pre(self, pre_model):
        self.t_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=4, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
    def forward(self, fr, ft):
        output = fr
        for layer in self.layers:
            output = layer(output, ft)
        return output
class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fr, ft, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        fr2 = self.multihead_attn(query=self.with_pos_embed(fr, query_pos).transpose(0, 1),#hw b c
                                   key=self.with_pos_embed(ft, pos).transpose(0, 1),
                                   value=ft.transpose(0, 1))[0].transpose(0, 1)#b hw c
        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)
        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))  #FFN
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)
        # print(fr.shape)
        return fr
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
