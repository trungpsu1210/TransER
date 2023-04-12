import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_

########################################################################################################################

class RLN(nn.Module):

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

##########################################################################

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

########################################################################################################################

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

########################################################################################################################

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

########################################################################################################################

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

########################################################################################################################

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out

########################################################################################################################

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn

        if not self.use_attn:
            self.mlp_norm = mlp_norm

            self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
            self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                                  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

            self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
            self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

        else:

            self.ffa_module = DehazeBlock(default_conv, dim, kernel_size=3)
            self.skfusion = SKFusion(dim=dim)

            self.mlp_norm = mlp_norm

            self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
            self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                                  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

            self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
            self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):

        if not self.use_attn:
            identity = x
            if self.use_attn: x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            if self.use_attn: x = x * rescale + rebias
            x = identity + x

            identity = x
            if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
            x = self.mlp(x)
            if self.use_attn and self.mlp_norm: x = x * rescale + rebias
            x = identity + x

        else:

            ffa_out = self.ffa_module(x)

            identity = x
            if self.use_attn: x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            if self.use_attn: x = x * rescale + rebias
            x = identity + x

            identity = x
            if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
            x = self.mlp(x)
            if self.use_attn and self.mlp_norm: x = x * rescale + rebias
            x = identity + x

            x = self.skfusion([ffa_out, x])

        return x

########################################################################################################################

class BasicLayer_1(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

########################################################################################################################

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x

########################################################################################################################

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

########################################################################################################################

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

########################################################################################################################

class DehazeFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(DehazeFormer, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # BACKBONE

        # Encoder 1
        self.layer1 = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # Encoder 2
        self.layer2 = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # Bottleneck
        self.layer3 = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]

        # Decoder 1 A
        self.fusion1_A = SKFusion(embed_dims[3])

        self.layer4_A = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                   num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                   norm_layer=norm_layer[3], window_size=window_size,
                                   attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2_A = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # Decoder 1 t
        self.fusion1_t = SKFusion(embed_dims[3])

        self.layer4_t = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                   num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                   norm_layer=norm_layer[3], window_size=window_size,
                                   attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2_t = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # Decoder 1 J
        self.fusion1_J = SKFusion(embed_dims[3])

        self.layer4_J = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                   num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                   norm_layer=norm_layer[3], window_size=window_size,
                                   attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2_J = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]

        # Decoder 2 A
        self.fusion2_A = SKFusion(embed_dims[4])

        self.layer5_A = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                   num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                   norm_layer=norm_layer[4], window_size=window_size,
                                   attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])


        self.patch_unembed_A = PatchUnEmbed(
            patch_size=1, out_chans=(out_chans - 1), embed_dim=embed_dims[4], kernel_size=3)

        # Decoder 2 t
        self.fusion2_t = SKFusion(embed_dims[4])

        self.layer5_t = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                   num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                   norm_layer=norm_layer[4], window_size=window_size,
                                   attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed_t = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # Decoder 2 J
        self.fusion2_J = SKFusion(embed_dims[4])

        self.layer5_J = BasicLayer_1(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                   num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                   norm_layer=norm_layer[4], window_size=window_size,
                                   attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed_J = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # Output of T map
        self.conv2D = nn.Conv2d(out_chans, 1, kernel_size=3, padding=1)
        self.sigmoid_t = nn.Sigmoid()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):

        # Encoder 1
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        # Encoder 2
        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        # Bottleneck
        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        # Decoder A
        x_A = self.fusion1_A([x, self.skip2(skip2)]) + x
        x_A = self.layer4_A(x_A)
        x_A = self.patch_split2_A(x_A)

        x_A = self.fusion2_A([x_A, self.skip1(skip1)]) + x_A
        x_A = self.layer5_A(x_A)
        x_A = self.patch_unembed_A(x_A)

        # Decoder J
        x_J = self.fusion1_J([x, self.skip2(skip2)]) + x
        x_J = self.layer4_J(x_J)
        x_J = self.patch_split2_J(x_J)

        x_J = self.fusion2_J([x_J, self.skip1(skip1)]) + x_J
        x_J = self.layer5_J(x_J)
        x_J = self.patch_unembed_J(x_J)

        # Decoder J
        x_t = self.fusion1_t([x, self.skip2(skip2)]) + x
        x_t = self.layer4_t(x_t)
        x_t = self.patch_split2_t(x_t)

        x_t = self.fusion2_t([x_t, self.skip1(skip1)]) + x_t
        x_t = self.layer5_t(x_t)
        x_t = self.patch_unembed_t(x_t)

        x_t = self.conv2D(x_t)
        x_t = self.sigmoid_t(x_t)
        x_t = torch.abs((x_t)) + (10 ** -10)
        x_t = x_t.repeat(1, 3, 1, 1)

        return x_A, x_t, x_J

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x_A, x_t, x_J = self.forward_features(x)

        K_J, B_J = torch.split(x_J, (1, 3), dim=1)

        x_J = K_J * x - B_J + x

        x_J = x_J[:, :, :H, :W]
        x_t = x_t[:, :, :H, :W]
        x_A = x_A[:, :, :H, :W]

        dehaze_reconstruct = (x - x_A * (1 - x_t)) / x_t

        output = (x_J + dehaze_reconstruct) / 2

        return output, dehaze_reconstruct, x_J, x_A

########################################################################################################################

class ConvLayer(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim

        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out

########################################################################################################################

class BasicBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
        super().__init__()
        self.norm = norm_layer(dim)
        self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv(x)
        x = identity + x
        return x

########################################################################################################################

class BasicLayer_2(nn.Module):
    def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


########################################################################################################################

class AFA(nn.Module):
    def __init__(self, m=-0.68):
        super(AFA, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

########################################################################################################################

class gUNet(nn.Module):
    def __init__(self, kernel_size=5,
                base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4],
                conv_layer=ConvLayer,
                norm_layer=nn.BatchNorm2d,
                gate_act=nn.Sigmoid,
                fusion_layer=SKFusion):

        super(gUNet, self).__init__()
        # setting
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2**i*base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num

        # input convolution
        self.inconv1 = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)
        self.inconv2 = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layers = nn.ModuleList()
        self.encoder2 = nn.ModuleList()
        self.downs1 = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips1 = nn.ModuleList()
        self.skips2 = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.afa = nn.ModuleList()

        self.afa_bottleneck = AFA()
        self.afa_input = AFA()

        for i in range(self.stage_num):
            self.layers.append(BasicLayer_2(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                                          conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num):
            self.downs1.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.downs2.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.skips1.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.skips2.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.encoder2.append(BasicLayer_2(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                                            conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))
            self.fusions.append(fusion_layer(embed_dims[i]))
            self.afa.append(AFA())

        # output convolution
        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)

    def forward(self, x1, x2):
        feat1 = self.inconv1(x1)
        feat2 = self.inconv2(x2)

        skips1 = []
        skips2 = []

        for i in range(self.half_num):

            feat1 = self.layers[i](feat1)
            skips1.append(self.skips1[i](feat1))
            feat1 = self.downs1[i](feat1)

            feat2 = self.encoder2[i](feat2)
            skips2.append(self.skips2[i](feat2))
            feat2 = self.downs2[i](feat2)

        feat = self.afa_bottleneck(feat1, feat2)

        feat = self.layers[self.half_num](feat)
        out_mid = feat

        for i in range(self.half_num-1, -1, -1):
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, self.afc[i](skips1[i], skips2[i])])
            feat = self.layers[self.stage_num-i-1](feat)

        out = self.outconv(feat) + self.afa_input(x1, x2)

        return out, out_mid

########################################################################################################################

def TFD():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1 / 2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def TRN():
    return gUNet(kernel_size=5,
        base_dim=24,
        depths=[2, 2, 2, 4, 2, 2, 2],
        conv_layer=ConvLayer,
        norm_layer=nn.BatchNorm2d,
        gate_act=nn.Sigmoid,
        fusion_layer=SKFusion)

def LER():
    return gUNet(kernel_size=5,
        base_dim=24,
        depths=[2, 2, 2, 4, 2, 2, 2],
        conv_layer=ConvLayer,
        norm_layer=nn.BatchNorm2d,
        gate_act=nn.Sigmoid,
        fusion_layer=SKFusion)


