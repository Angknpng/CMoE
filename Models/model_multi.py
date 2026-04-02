from .Swin import SwinTransformer
from torch import nn, Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
# from modules import DeformConv
import copy
from timm.models.layers import trunc_normal_
import numpy as np
import torch.nn.init as init
# zeros_ = torch.nn.init.constant_(val=0.0)
import hashlib

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
    """Identity"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        """forward"""
        return input

class Mlp(nn.Module):
    """Mlp"""

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
        """forward"""
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
        )  # make torchscript happy (cannot use tensor as tuple)

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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
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
        """forward"""
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

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
        """forward"""
        # B, C, H, W = x.shape
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



class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // 2, 1, bias=False)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Conv2d(in_ch // 2, in_ch, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_weight(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_weight(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out #out * x


class Con_Router(nn.Module):
    def __init__(self, in_channels):
        super(Con_Router, self).__init__()
        self.ca = CA(in_channels * 2)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2),  # 输出 2 个权重
        )

        self.conv_fusion = conv3x3_bn_relu(in_channels * 2, in_channels * 2)
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape

        rgb_t_concat = torch.cat((rgb, t), dim=1)
        frt = self.conv_fusion(rgb_t_concat)

        frt_ca = self.ca(frt)
        weights = self.mlp(frt_ca.view(B, 2 * C))
        weights = F.softmax(weights, dim=-1)

        out = rgb * weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + t * weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return out

    # 保存权重日志到 .log 文件
    def save_log_to_file(self, filename="weights.log"):
        with open(filename, "w") as log_file:
            for i, (avg_weight_rgb, avg_weight_t) in enumerate(zip(self.weight_log['weight_rgb_avg'], self.weight_log['weight_t_avg'])):
                log_file.write(f"Batch {i + 1}:\n")
                log_file.write(f"Average Weight RGB: {avg_weight_rgb:.4f}\n")
                log_file.write(f"Average Weight T: {avg_weight_t:.4f}\n\n")


class CopyExpert(torch.nn.Module):
    def __init__(self):
        super(CopyExpert, self).__init__()
        pass
    def forward(self, inputs):
        return inputs

class ZeroExpert(torch.nn.Module):
    def __init__(self):
        super(ZeroExpert, self).__init__()
        pass
    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)

class ConstantExpert(nn.Module):
    def __init__(self, in_channels, height, width):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(torch.empty(1, in_channels, height, width))
        torch.nn.init.normal_(self.constant, mean=0.0, std=0.02)
        self.wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # (B, C)
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 2)
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, inputs):
        B, C, H, W = inputs.shape
        weights = self.wg(inputs)
        weights = self.softmax(weights)
        w_input = weights[:, 0].view(B, 1, 1, 1)
        w_const = weights[:, 1].view(B, 1, 1, 1)
        output = w_input * inputs + w_const * self.constant
        return output

class missRouter(nn.Module):
    def __init__(self, in_channel, size, mlp_ratio=0.1):
        super(missRouter, self).__init__()
        self.conv = nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(3*size*size, size),
            nn.ReLU(),
            nn.Linear(size, 3),
        )
    def forward(self, X):
        X = self.conv(X)
        B, C, H, W = X.shape
        X_flat = X.view(B, C * H * W)
        weights = self.mlp(X_flat)  
        weights = F.softmax(weights, dim=-1)
        weights = weights.type_as(X)  
        return weights

class missMoE(nn.Module):
    def __init__(self, in_channel, size):
        super(missMoE, self).__init__()
        self.copyExpert = CopyExpert()
        self.zeroExpert = ZeroExpert()
        self.switchExpert = ConstantExpert(in_channel, size, size)
        self.router = missRouter(in_channel=in_channel, size=size)
    def forward(self, x):
        fea_copy = self.copyExpert(x)
        fea_zero = self.zeroExpert(x)
        fea_switch = self.switchExpert(x)
        weights = self.router(x)
        weighted_feaCopy = fea_copy * weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        weighted_feaZero = fea_zero * weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        weighted_feaSwitch = fea_switch * weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        output = weighted_feaCopy + weighted_feaZero + weighted_feaSwitch
        return output

def modality_drop(x_rgb, x_depth, mode):
    if mode == 'train':
        modality_combination = [[1, 0], [0, 1], [1, 1]]
    elif mode == 'test':
        modality_combination = [[1, 1], [1, 1], [1, 1]]
    index_list = [x for x in range(3)]
    p = []
    prob = np.array(( 1 / 3 , 1 / 3, 1 / 3))
    for i in range(x_rgb.shape[0]):
        index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
        p.append(modality_combination[index])
    p = np.array(p)
    p = torch.from_numpy(p)
    p = torch.unsqueeze(p, 2)
    p = torch.unsqueeze(p, 3)
    p = torch.unsqueeze(p, 4)
    p = p.float().cuda()
    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]
    return x_rgb, x_depth, p[:, 0], p[:, 1]

#model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #----------Pretarined encoder-------------
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        for param in self.rgb_swin.parameters():
            param.requires_grad = False
        for param in self.t_swin.parameters():
            param.requires_grad = False
        #----------Pretarined encoder-------------
        
        #----------Trainable missMoE-------------
        self.missMoE4_rgb = missMoE(1024, 12)
        self.missMoE3_rgb = missMoE(512, 24)
        self.missMoE2_rgb = missMoE(256, 48)
        self.missMoE1_rgb = missMoE(128, 96)
        self.missMoE4_t = missMoE(1024, 12)
        self.missMoE3_t = missMoE(512, 24)
        self.missMoE2_t = missMoE(256, 48)
        self.missMoE1_t = missMoE(128, 96)
        #----------Trainable missMoE-------------

        #-----------------------------------------Pretarined Expert_rgb-----------------------------------------------
        self.MHSA4_rgb = GMSA_ini(d_model=1024)
        self.MHSA3_rgb = GMSA_ini(d_model=512)
        self.MHSA2_rgb = GMSA_ini(d_model=256)
        for param in self.MHSA4_rgb.parameters():
            param.requires_grad = False
        for param in self.MHSA3_rgb.parameters():
            param.requires_grad = False
        for param in self.MHSA2_rgb.parameters():
            param.requires_grad = False
        #-----------------------------------------Pretarined Expert_rgb-----------------------------------------------

        #-----------------------------------------Pretarined Expert_t-----------------------------------------------
        self.MHSA4_t = GMSA_ini(d_model=1024)
        self.MHSA3_t = GMSA_ini(d_model=512)
        self.MHSA2_t = GMSA_ini(d_model=256)
        for param in self.MHSA4_t.parameters():
            param.requires_grad = False
        for param in self.MHSA3_t.parameters():
            param.requires_grad = False
        for param in self.MHSA2_t.parameters():
            param.requires_grad = False
        #-----------------------------------------Pretarined Expert_t-----------------------------------------------

        #----------Trainable decoder-------------
        self.router4 = Con_Router(in_channels=1024)
        self.router3 = Con_Router(in_channels=512)
        self.router2 = Con_Router(in_channels=256)
        self.router1 = Con_Router(in_channels=128)
        #----------Trainable decoder-------------

        #----------Trainable decoder-------------
        self.convr4 = conv3x3_bn_relu(1024, 512)
        self.convr3 = conv3x3_bn_relu(512, 256)
        self.convr2 = conv3x3_bn_relu(256, 128)
        self.convr1 = conv3x3_bn_relu(128, 64)
        self.conv_dim = conv3x3(64, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        #----------Trainable decoder-------------
    def forward(self, rgb, t, mode):
        rgb_miss, t_miss, missing_rgb, missing_t = modality_drop(rgb, t, mode=mode)#rgb, t with different missing conditions. The value of missing_rgb and missing_t: 1 indicates noMissing, 0 indicates missing.
        #----------Pretarined encoder-------------
        fr = self.rgb_swin(rgb_miss)  
        ft = self.t_swin(t_miss)  
        fr_ori = self.rgb_swin(rgb)  # for loss
        ft_ori = self.t_swin(t)  # for loss
        #----------Pretarined encoder-------------

        #----------Trainable missMoE-------------
        #*Use Thermal_fea to implement the miss_RGB*
        r4 = fr[3] + self.missMoE4_rgb(ft[3])
        r3 = fr[2] + self.missMoE3_rgb(ft[2])
        r2 = fr[1] + self.missMoE2_rgb(ft[1])
        r1 = fr[0] + self.missMoE1_rgb(ft[0])
        #*Use RGB_fea to implement the miss_Thermal*
        t4 = ft[3] + self.missMoE4_t(fr[3])
        t3 = ft[2] + self.missMoE3_t(fr[2])
        t2 = ft[1] + self.missMoE2_t(fr[1])
        t1 = ft[0] + self.missMoE1_t(fr[0])
        #----------Trainable missMoE-------------*

        #-----------------------------------------Pretarined Expert_rgb-----------------------------------------------
        flatten4 = r4.flatten(2).transpose(1, 2)# 
        flatten3 = r3.flatten(2).transpose(1, 2)
        flatten2 = r2.flatten(2).transpose(1, 2)

        r4_exp = self.MHSA4_rgb(flatten4, flatten4).view(flatten4.shape[0], int(np.sqrt(flatten4.shape[1])), int(np.sqrt(flatten4.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r3_exp = self.MHSA3_rgb(flatten3, flatten3).view(flatten3.shape[0], int(np.sqrt(flatten3.shape[1])), int(np.sqrt(flatten3.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r2_exp = self.MHSA2_rgb(flatten2, flatten2).view(flatten2.shape[0], int(np.sqrt(flatten2.shape[1])), int(np.sqrt(flatten2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        r1_exp = r1
        #-----------------------------------------Pretarined Expert_rgb-----------------------------------------------

        #-----------------------------------------Pretarined Expert_t-----------------------------------------------
        flatten4 = t4.flatten(2).transpose(1, 2)# B HW dim
        flatten3 = t3.flatten(2).transpose(1, 2)
        flatten2 = t2.flatten(2).transpose(1, 2)
        t4_exp = self.MHSA4_t(flatten4, flatten4).view(flatten4.shape[0], int(np.sqrt(flatten4.shape[1])), int(np.sqrt(flatten4.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t3_exp = self.MHSA3_t(flatten3, flatten3).view(flatten3.shape[0], int(np.sqrt(flatten3.shape[1])), int(np.sqrt(flatten3.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t2_exp = self.MHSA2_t(flatten2, flatten2).view(flatten2.shape[0], int(np.sqrt(flatten2.shape[1])), int(np.sqrt(flatten2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        t1_exp = t1
        #-----------------------------------------Pretarined Expert_t-----------------------------------------------

        #----------Trainable Conditional_Router-------------
        rt4 = self.router4(r4_exp, t4_exp)
        rt3 = self.router3(r3_exp, t3_exp)
        rt2 = self.router2(r2_exp, t2_exp)
        rt1 = self.router1(r1_exp, t1_exp)
        #----------Trainable Conditional_Router-------------

        #----------Trainable decoder-------------
        rt4 = self.convr4(self.up2(rt4))
        rt3 = self.convr3(self.up2(rt3 + rt4))
        rt2 = self.convr2(self.up2(rt2 + rt3))
        rt1 = self.convr1(rt1 + rt2)
        out = self.up4(rt1)
        out = self.conv_dim(out)
        #----------Trainable decoder-------------

        #----------Upsample for loss-------------
        fr4_1 = fr_ori[3]
        fr4_2 = r4
        fr3_1 = fr_ori[2]
        fr3_2 = r3
        fr2_1 = fr_ori[1]
        fr2_2 = r2
        fr1_1 = fr_ori[0]
        fr1_2 = r1
        ft4_1 = ft_ori[3]
        ft4_2 = t4
        ft3_1 = ft_ori[2]
        ft3_2 = t3
        ft2_1 = ft_ori[1]
        ft2_2 = t2
        ft1_1 = ft_ori[0]
        ft1_2 = t1
        #----------Upsample for loss-------------
        return out, fr4_1, fr4_2, fr3_1, fr3_2, fr2_1, fr2_2, fr1_1, fr1_2, ft4_1, ft4_2, ft3_1, ft3_2, ft2_1, ft2_2, ft1_1, ft1_2#, missing_rgb, missing_t


    def load_pre(self, pre_model_rgb, pre_model_t):
        layer_name = 'layers.0.blocks.0.attn.qkv.weight'
        rgb_before_hash = get_tensor_hash(self.rgb_swin.state_dict()[layer_name].clone())
        t_before_hash = get_tensor_hash(self.t_swin.state_dict()[layer_name].clone())
        print("加载前RGB模型权重:", rgb_before_hash)
        print("加载前T模型权重:", t_before_hash)
        MHSA4_rgb_before_hash = get_tensor_hash(self.MHSA4_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA4_t_before_hash = get_tensor_hash(self.MHSA4_t.state_dict()['layers.0.linear1.weight'].clone())
        MHSA3_rgb_before_hash = get_tensor_hash(self.MHSA3_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA3_t_before_hash = get_tensor_hash(self.MHSA3_t.state_dict()['layers.0.linear1.weight'].clone())
        MHSA2_rgb_before_hash = get_tensor_hash(self.MHSA2_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA2_t_before_hash = get_tensor_hash(self.MHSA2_t.state_dict()['layers.0.linear1.weight'].clone())
        print("加载前MHSA4_rgb权重:", MHSA4_rgb_before_hash)
        print("加载前MHSA4_t权重:", MHSA4_t_before_hash)
        self.rgb_swin.load_state_dict({k.replace('module.rgb_swin.',''):v for k,v in torch.load(pre_model_rgb).items()}, strict=False)
        self.MHSA4_rgb.load_state_dict({k.replace('module.MHSA4_rgb.',''):v for k,v in torch.load(pre_model_rgb).items()}, strict=False)
        self.MHSA3_rgb.load_state_dict({k.replace('module.MHSA3_rgb.',''):v for k,v in torch.load(pre_model_rgb).items()}, strict=False)
        self.MHSA2_rgb.load_state_dict({k.replace('module.MHSA2_rgb.',''):v for k,v in torch.load(pre_model_rgb).items()}, strict=False)
        self.t_swin.load_state_dict({k.replace('module.t_swin.',''):v for k,v in torch.load(pre_model_t).items()}, strict=False)
        self.MHSA4_t.load_state_dict({k.replace('module.MHSA4_t.',''):v for k,v in torch.load(pre_model_t).items()}, strict=False)
        self.MHSA3_t.load_state_dict({k.replace('module.MHSA3_t.',''):v for k,v in torch.load(pre_model_t).items()}, strict=False)
        self.MHSA2_t.load_state_dict({k.replace('module.MHSA2_t.',''):v for k,v in torch.load(pre_model_t).items()}, strict=False)
        rgb_after_hash = get_tensor_hash(self.rgb_swin.state_dict()[layer_name])
        t_after_hash = get_tensor_hash(self.t_swin.state_dict()[layer_name])
        print("加载后RGB模型权重:", rgb_after_hash)
        print("加载后T模型权重:", t_after_hash)
        MHSA4_rgb_after_hash = get_tensor_hash(self.MHSA4_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA4_t_after_hash = get_tensor_hash(self.MHSA4_t.state_dict()['layers.0.linear1.weight'].clone())
        MHSA3_rgb_after_hash = get_tensor_hash(self.MHSA3_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA3_t_after_hash = get_tensor_hash(self.MHSA3_t.state_dict()['layers.0.linear1.weight'].clone())
        MHSA2_rgb_after_hash = get_tensor_hash(self.MHSA2_rgb.state_dict()['layers.0.linear1.weight'].clone())
        MHSA2_t_after_hash = get_tensor_hash(self.MHSA2_t.state_dict()['layers.0.linear1.weight'].clone())
        print("加载后MHSA4_rgb权重:", MHSA4_rgb_after_hash)
        print("加载后MHSA4_t权重:", MHSA4_t_after_hash)
        if (rgb_before_hash != rgb_after_hash) and (MHSA4_rgb_before_hash != MHSA4_rgb_after_hash) and (MHSA3_rgb_before_hash != MHSA3_rgb_after_hash) and (MHSA2_rgb_before_hash != MHSA2_rgb_after_hash):
            print("RGB模型权重已成功加载。")
        else:
            print("警告：RGB模型权重加载失败。")
        if (t_before_hash != t_after_hash) and (MHSA4_t_before_hash != MHSA4_t_after_hash) and (MHSA3_t_before_hash != MHSA3_t_after_hash) and (MHSA2_t_before_hash != MHSA2_t_after_hash):
            print("Thermal/Depth模型权重已成功加载。")
        else:
            print("警告：Thermal/Depth模型权重加载失败。")

def get_tensor_hash(tensor):
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()

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