# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from math import sqrt

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath as DropPath_


def get_2d_relative_position_index(H, W):
    r"""Get pair-wise relative position index for each token inside a (h, w) window.
    Args:
        H (int): height of window
        W (int): width of window

    Returns:
        relative_position_index: (H*W, H*W)
    """
    coords_h = torch.arange(H)
    coords_w = torch.arange(W)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
    relative_coords[:, :, 0] += H - 1  # shift to start from 0
    relative_coords[:, :, 1] += W - 1
    relative_coords[:, :, 0] *= 2 * W - 1
    relative_position_index = relative_coords.sum(-1)  # H*W, H*W
    return relative_position_index


def remap_position_index(position_index, shift=0):
    r"""Zip position index to consecutive integers and count from a specified value.
    
    Args:
        position_index: relative position indexs of (H*W, H*W)
        shift (int): shift position index with a specified value

    Returns:
        result: zipped and shifted realtive position indexs of (H*W, H*W)
        total_len (int): count of unique indexs
    """
    H, W = position_index.shape
    curr_values = position_index.view(-1).tolist()
    set_values = sorted(list(set(curr_values)))
    total_len = len(set_values)
    map_dict = {set_values[i]: shift + i for i in range(total_len)}
    result_values = [map_dict[curr_value] for curr_value in curr_values]
    result = torch.LongTensor(result_values, device=position_index.device).view(H, W)
    return result, total_len


class DropPath(DropPath_):
    def extra_repr(self) -> str:
        return 'drop_prob={}'.format(self.drop_prob)


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MultiHeadSelfAttention(nn.Module):
    r""" Multi-head self attention module with relative position bias for RSA and LSA.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        window_num (tuple[int]): The number of windows from height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, window_num, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.window_num = window_num  # nWh, nWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        relative_position_index_patch = get_2d_relative_position_index(*window_size)
        relative_position_index_patch, total_len_patch = remap_position_index(relative_position_index_patch)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(total_len_patch, num_heads))
        self.register_buffer("relative_position_index_patch", relative_position_index_patch)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, num_windows, C)
        """
        B_, N, C = x.shape
        if N == self.window_num[0] * self.window_num[1]:  # regional tokens
            mode = 'RSA'
        elif N == self.window_size[0] * self.window_size[1] + 1:  # intra window patchs + 1 regional token
            mode = 'LSA'
        else:
            raise ValueError
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Section 3. Relative Position Bias. only use relative position bias in LSA for patch/local tokens
        if mode == 'LSA':
            relative_position_bias_ = self.relative_position_bias_table[self.relative_position_index_patch.view(-1)].view(
                (self.window_size[0] * self.window_size[1]), (self.window_size[0] * self.window_size[1]), -1)  # Wh*Ww,Wh*Ww,nH
            WhWw, _, nH = relative_position_bias_.shape
            relative_position_bias = torch.zeros((WhWw + 1, WhWw + 1, nH), dtype=relative_position_bias_.dtype, device=relative_position_bias_.device)
            relative_position_bias[1:, 1:, :] = relative_position_bias_
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_num={self.window_num}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class R2LEncoderBlock(nn.Module):
    r""" R2L Transformer Encoder Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        windom_num (int): Num of Windows.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, window_num=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_num = window_num
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, window_size=to_2tuple(self.window_size), window_num=to_2tuple(self.window_num), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, regional_tokens, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        nW = regional_tokens.size(1)
        assert L == H * W, "input feature has wrong size"

        # -----------------------------------------------------------------------
        # RSA :: y_r = x_r + RSA(LN(x_r))
        regional_tokens = regional_tokens + self.drop_path(self.attn(self.norm1(regional_tokens)))

        # -----------------------------------------------------------------------
        # CAT :: y = y_r || x_l
        x = x.view(B, H, W, C)
        regional_tokens = regional_tokens.permute(1, 0, 2).contiguous().view(nW*B, 1, C)
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        regional_with_x_windows = torch.cat((regional_tokens, x_windows), 1)  # nW*B, 1+window_size*window_size, C

        # -----------------------------------------------------------------------
        # LSA :: z = y + LSA(LN(y))
        regional_with_x_windows = regional_with_x_windows + self.drop_path(self.attn(self.norm2(regional_with_x_windows)))

        # -----------------------------------------------------------------------
        # FFN :: x = z + FFN(LN(z))
        regional_with_x_windows = regional_with_x_windows + self.drop_path(self.mlp(self.norm3(regional_with_x_windows)))  # nW*B, 1+window_size*window_size, C

        # decode
        # (nW*B, 1, C), (nW*B, window_size*window_size, C)
        regional_windows, x_windows = torch.split(regional_with_x_windows, [1, self.window_size*self.window_size], dim=1)
        regional_tokens = regional_windows.view(nW, B, C).permute(1, 0, 2).contiguous()
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        return regional_tokens, x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, window_num={self.window_num}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * self.window_num * self.window_num
        # RSA
        flops += self.attn.flops(self.window_num * self.window_num)
        # norm2
        flops += self.dim * (H * W + self.window_num * self.window_num)
        # LSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size + 1)
        # mlp
        flops += 2 * (H * W + self.window_num * self.window_num) * self.dim * self.dim * self.mlp_ratio
        # norm3
        flops += self.dim * (H * W + self.window_num * self.window_num)
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(self.dim, self.dim*2, kernel_size=3, stride=2, padding=1, groups=self.dim)

    def forward(self, x):
        """
        x: B, H*W, C
        assert H == W
        """
        B, L, C = x.shape
        assert sqrt(L) == int(sqrt(L))
        H = W = int(sqrt(L))

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.reduction(x).permute(0, 2, 3, 1).contiguous()

        x = x.view(B, -1, 2 * C)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def flops(self, H, W):
        # flops for Conv2d: kernel_size * in_channels * filters_per_channel * output shape
        flops = (3 * 3) * self.dim * 2 * (H // 2) * (W // 2)
        return flops


class BasicLayer(nn.Module):
    """ A basic R2L Transformer Encoder layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        window_num = input_resolution[0] // window_size

        # build blocks
        self.blocks = nn.ModuleList([
            R2LEncoderBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 window_num=window_num,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        self.downsample = None
        if downsample is not None:
            self.downsample = downsample(dim=dim)

    def forward(self, regional_tokens, local_tokens):
        for blk in self.blocks:
            if self.use_checkpoint:
                regional_tokens, local_tokens = checkpoint.checkpoint(blk, local_tokens, regional_tokens)
            else:
                regional_tokens, local_tokens = blk(regional_tokens, local_tokens)
        if self.downsample is not None:
            local_tokens = self.downsample(local_tokens)
            regional_tokens = self.downsample(regional_tokens)
        return regional_tokens, local_tokens

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops(*self.input_resolution)
            flops += self.downsample.flops(self.input_resolution[0] // self.window_size, self.input_resolution[1] // self.window_size)
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, method='3-conv'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.method = method

        # GroupNorm(1, n) => LayerNorm
        # GroupNorm(n, n) => InstanceNorm
        if method == '3-conv':
            self.proj = nn.Sequential(
                nn.Conv2d(3, embed_dim // 4, 3, 2, 1),
                nn.GroupNorm(1, embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                nn.GroupNorm(1, embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
            )
        elif method == '1-conv':
            self.proj = nn.Conv2d(3, embed_dim, 8, 4, 3)
        elif method == 'linear':
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
                nn.Linear(patch_size[0] * patch_size[1] * in_chans, embed_dim)
            )
        else:
            raise NotImplementedError
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if 'conv' in self.method:
            x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        elif 'linear' in self.method:
            x = self.proj(x)
        else:
            raise ValueError
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        img_size = self.img_size[0]  # assert H == W
        if self.method == '3-conv':
            flops += 3 * 3 * 3 * (self.embed_dim // 4) * (img_size // 2) * (img_size // 2)
            flops += (self.embed_dim // 4) * (img_size // 2)
            flops += 3 * 3 * (self.embed_dim // 4) * (self.embed_dim // 2) * (img_size // 4) * (img_size // 4)
            flops += (self.embed_dim // 2) * (img_size // 4)
            flops += 3 * 3 * (self.embed_dim // 2) * self.embed_dim * (img_size // 4) * (img_size // 4)
        elif self.method == '1-conv':
            flops += 8 * 8 * 3 * self.embed_dim * (img_size // 4) * (img_size // 4)
        elif self.method == 'linear':
            flops += self.num_patches * self.patch_size[0] * self.patch_size[1] * self.in_chans * self.embed_dim
        else:
            raise ValueError
        return flops


class RegionViT(nn.Module):
    r""" Region ViT
        A PyTorch impl of : `RegionViT: Regional-to-Local Attention for Vision Transformers`  -
          https://arxiv.org/pdf/2106.02689
          based on https://github.com/microsoft/Swin-Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        local_tokenization_method (str): tokenization method for local tokens. Default: '3-conv'
        regional_tokenization_method (str): tokenization method for regional tokens. Default: 'linear'
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 local_tokenization_method='3-conv', regional_tokenization_method='linear',
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        window_num = img_size // patch_size // window_size
        self.num_windows = window_num * window_num
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, method=local_tokenization_method)
        self.regional_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size*7, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, method=regional_tokenization_method)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_windows, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, img):
        x = self.patch_embed(img)
        N, L, C = x.shape
        regional_tokens = self.regional_embed(img)
        _, LW, C = regional_tokens.shape
        regional_x = torch.cat((regional_tokens, x), dim=1)
        if self.ape:
            regional_x = regional_x + self.absolute_pos_embed
        regional_x = self.pos_drop(regional_x)
        regional_tokens, local_tokens = torch.split(regional_x, [LW, L], dim=1)

        for layer in self.layers:
            regional_tokens, local_tokens = layer(regional_tokens, local_tokens)

        x = self.norm(regional_tokens)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        flops += self.regional_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
