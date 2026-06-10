import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import json,os
from SIMuLDiTex.ResizeRight import resize
import SIMuLDiTex.interp_methods as interp
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam,SGD

from torchvision import transforms as T, utils


from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from SIMuLDiTex.attend import Attend

from SIMuLDiTex.version import __version__

import torch
import torch.nn.functional as F
import numpy as np
from random import sample

  
   
import torchvision.models as models


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        if False: # center feature maps before gram matrix correlation: default= yes
            F=(F-F.mean(dim=-1,keepdim=True))/(F.std(dim=-1,keepdim=True)+10**-10)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)  # only divides by spatial dim
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


def gaussian_pyramid(x,scales,octaves):    
    l=[]
    for octave in range(octaves):
        for i in range(scales):
            l.append(resize(x,interp_method=interp.linear,out_shape=(int((2**(-i/scales-octave))*x.shape[-2]),int((2**(-i/scales-octave))*x.shape[-1]))))
    l.append(resize(x,interp_method=interp.linear,out_shape=(int((2**(-octaves))*x.shape[-2]),int((2**(-octaves))*x.shape[-1]))))
    return l




def SW2_hist_grad(img,gt):
    loss=0.
    b, c, h, w = img.shape
    v=torch.randn(c,1,dtype=img.dtype).to(img.device)
    v=v/v.norm()
    img_proj = torch.matmul(img.reshape(b, c, -1).transpose(1, 2), v).squeeze(-1)
    gt_proj = torch.matmul(gt.reshape(b, c, -1).transpose(1, 2), v).squeeze(-1)
    sorted, indices = torch.sort(img_proj)
    sorted_gt, indices_gt = torch.sort(gt_proj)
    if len(sorted_gt)<=len(sorted):
        stretched_proj = F.interpolate(sorted_gt.unsqueeze(1), size=indices.shape[-1],mode = 'nearest', recompute_scale_factor = False).squeeze(1) # handles a generated image larger than the ground tru
    else:
        stretched_proj=sorted_gt
        sorted=stretched_proj = F.interpolate(sorted.unsqueeze(1), size=indices_gt.shape[-1],mode = 'nmid_fourierearest', recompute_scale_factor = False).squeeze(1)
    _,inv_ind = torch.sort(indices)
    target=torch.gather(stretched_proj,1,inv_ind)-img_proj
    target=target.unsqueeze(1)*v.unsqueeze(0)
    target=target.view(b,c,h,w)
    return target



def patch_recursive(x,max_area=512,overlap=2,path=0):
    b,c,h,w=x.shape
    if h*w<=max_area:
        return [x],[path]
    if h*w<=max_area*2 or h*w>=max_area*8:
        if h>=w:
            p1,l1=patch_recursive(x[:,:,:h//2+overlap],max_area=max_area,overlap=overlap,path=10*path+1)
            p2,l2=patch_recursive(x[:,:,-overlap+h//2:],max_area=max_area,overlap=overlap,path=10*path+9)
            return p1+p2,l1+l2
        else:
            p1,l1=patch_recursive(x[:,:,:,:w//2+overlap],max_area=max_area,overlap=overlap,path=10*path+2)
            p2,l2=patch_recursive(x[:,:,:,-overlap+w//2:],max_area=max_area,overlap=overlap,path=10*path+8)
            return p1+p2,l1+l2
    if h>=w:
        cut=max_area//w-overlap
        p1,l1=patch_recursive(x[:,:,:cut+overlap],max_area=max_area,overlap=overlap,path=10*path+1)
        p2,l2=patch_recursive(x[:,:,-overlap+cut:],max_area=max_area,overlap=overlap,path=10*path+9)
        return p1+p2,l1+l2
    else:
        cut=max_area//h-overlap
        p1,l1=patch_recursive(x[:,:,:,:cut+overlap],max_area=max_area,overlap=overlap,path=10*path+2)
        p2,l2=patch_recursive(x[:,:,:,-overlap+cut:],max_area=max_area,overlap=overlap,path=10*path+8)
        return p1+p2,l1+l2
    
    

def depatch(patches,paths,overlap=2):
    f = lambda x: torch.where(x < 1/3, torch.tensor(0.0), torch.where(x < 2/3, 3 * (x - 1/3), torch.tensor(1.0)))
    while len(patches)!=1:
        path1=max(paths)
        r1=path1%10
        path2=path1+10-2*r1
        ind1,ind2=paths.index(path1),paths.index(path2)
        p1=patches.pop(max(ind1,ind2))
        p2=patches.pop(min(ind1,ind2))
        paths.pop(max(ind1,ind2))
        paths.pop(min(ind1,ind2))
        if ind2<ind1:
            p1,p2=p2,p1
        if r1 in [1,9]:
            p1=p1.permute(0,1,3,2)
            p2=p2.permute(0,1,3,2)
        
        ramp = f(torch.linspace(0,1,2*overlap)).view(1,1,1,-1)
        o1,o2=p1[...,-2*overlap:],p2[...,:2*overlap]
        overlap_region=(1-ramp)*o1+ramp*o2
        p_stitched=torch.cat((p1[...,:-2*overlap],overlap_region,p2[...,2*overlap:]),dim=-1)
        if r1 in [1,9]:
            p_stitched=p_stitched.permute(0,1,3,2)
        patches.append(p_stitched)
        paths.append(path1//10)
    return patches[0]




class DummyDset(Dataset):
    def __init__(self, length):
        self.length = length 
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return None
    
def gaussian_kernel(size, sigma):
    kernel = torch.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center +.5
            y = j - center +.5
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * torch.exp(-torch.tensor((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_filter(img, kernel_size=5, sigma=1.0,device="cuda:0"):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(img.size(1), 1, 1, 1).to(device)
    if img.dim() == 3:  
        img = img.unsqueeze(0)
    img_filtered = F.conv2d(img, kernel, padding=kernel_size // 2, groups=img.size(1))
    return img_filtered


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class Zero(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.
# model

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block


        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm).type(torch.complex64)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)


        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1]).to(torch.complex64)

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 4),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        mid_attn=False,
        mid_fourier=False):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.mid_fourier=mid_fourier

        if mid_fourier:
            self.FU=FourierUnit(dim*dim_mults[-1],dim*dim_mults[-1])
        else:
            self.FU = Zero()


        sinu_scale_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
        fourier_dim = learned_sinusoidal_dim + 1
        self.scale_mlp = nn.Sequential(
            sinu_scale_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim


        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), False)
            #full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)


        
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Zero(),#attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        if mid_attn:
            self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        else:
            self.mid_attn = Zero()
        
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Zero(),#attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time,scale , x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)+self.scale_mlp(scale)


        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        if self.mid_fourier:
            x = x + self.FU(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t,scale=1,   x_self_cond = None, clip_x_start = False, rederive_pred_noise = False,omega=0):
        if not isinstance(omega, torch.Tensor):
            if omega==0:
                model_output = self.model(x, t,scale, x_self_cond)
            else:
                model_output = self.model(x, t,scale, x_self_cond)*(1-omega) + omega * self.model2(x, t,scale, x_self_cond)
        else:
            model_output = self.model(x, t,scale, x_self_cond)*(1-omega) + omega * self.model2(x, t,scale, x_self_cond)
        
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, scale=1, omega=0, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t,scale, x_self_cond,omega=omega)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, scale=1, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        try:
            scale_cond = torch.full((b,), scale, device = device)
        except:
            scale_cond = torch.tensor(scale).to(device)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance( x,batched_times,scale_cond, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape,scale=1, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t,scale, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, scale=1, return_all_timesteps = False,omega=0):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img.cpu()]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            try:
                scale_cond = torch.full((batch,), scale, device = device)
            except:
                scale_cond = torch.tensor(scale).to(device)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond,scale=scale_cond, x_self_cond=self_cond, clip_x_start=True, rederive_pred_noise = True, omega=omega)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            #imgs.append(img.cpu())

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    

    
    def invert(self,img,scale=1):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        # Latents are now the specified start latents
        latents = self.normalize(img).to(device)
        #latents=self.normalize(img.gpu())
        batch=img.shape[0]
        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        times = torch.linspace(0, total_timesteps -1, steps = total_timesteps + 1)  
        times = list(times.int().tolist())
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        with torch.no_grad():
            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                try:
                    scale_cond = torch.full((batch,), scale, device = device)
                except:
                    scale_cond = torch.tensor(scale).to(device)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(latents, time_cond,scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                
                latents = (latents - (1-alpha).sqrt()*pred_noise)*(alpha_next.sqrt()/alpha.sqrt()) + (1-alpha_next).sqrt()*pred_noise


        return latents
    


    def scale_gen(self, size,scale_factor=1,gt=None,latent=None,scales=2,omega=0,octaves=3,drop_last_n_resolutions=0,time_ratio=1,histogram_matching=False, return_all_timesteps = False,noise=None,patch_size=4096):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        #torch.manual_seed(0)
        
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        
        border_hr=30
        overlap=100
        if gt is not None:
            gt=gt.cuda()
            gt=self.normalize(gt)

          

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = total_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if latent is None:
            img = torch.randn((1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)
        else:
            img=latent
        #gt_resized=resize(gt.to(img.device),scale_factors=1/scale_factor)
        img=img.to(device)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=omega)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        if histogram_matching and gt is not None:

            gt_normalized=resize(gt.to(img.device),scale_factors=1/scale_factor)
            for i in range(20):
                img=img+SW2_hist_grad(img,gt_normalized)
            img=img.cpu()
        return self.unnormalize(img)
    

    
    
    def inpaint(self, input,size,scale_factor=1,gt=None,scales=2,omega=0,octaves=3,drop_last_n_resolutions=0,time_ratio=1,histogram_matching=False, return_all_timesteps = False,noise=None,patch_size=4096):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        #torch.manual_seed(0)
        input=self.normalize(input).to(device)
        '''to generalize MS'''
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        
        border_hr=30
        overlap=100
        if gt is not None:
            gt=gt.cuda()
            gt=self.normalize(gt)

        

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = total_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

       
        img = torch.randn((1,3,int(size[0])//8*8,int(size[1])//8*8), device = device)

        #gt_resized=resize(gt.to(img.device),scale_factors=1/scale_factor)
        img=img.to(device)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=omega)

                if time_next < 0:
                    img = x_start
                    continue
                x_start[:,:,x_start.shape[2]//2-input.shape[2]//2:x_start.shape[2]//2+input.shape[2]-input.shape[2]//2,x_start.shape[3]//2-input.shape[3]//2:x_start.shape[3]//2+input.shape[3]-input.shape[3]//2]=input

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        if histogram_matching and gt is not None:

            gt_normalized=resize(gt.to(img.device),scale_factors=1/scale_factor)
            for i in range(20):
                img=img+SW2_hist_grad(img,gt_normalized)
            img=img.cpu()
        return self.unnormalize(img)
    




    def ms_gen(self, size,batch=1,gt=None,scales=2,omega=0,octaves=3,drop_last_n_resolutions=0,coarse_scale_steps=None,time_ratio=.7,histogram_matching=False, return_all_timesteps = False,noise=None,patch_size=4096,renoise_levels=None,seed=None,disable=False):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if seed is not None:
            torch.manual_seed(seed)

        shape=(batch,3,size[0],size[1])
        imgs=[]
        
        border_hr=30
        overlap=100
        if gt is not None:
            gt=gt.cuda()
            gt=self.normalize(gt)
            gt=gt.repeat(batch,1,1,1)
        coarse_scale_steps=total_timesteps if coarse_scale_steps is None else coarse_scale_steps 
        times = torch.linspace(-1, total_timesteps - 1, steps = coarse_scale_steps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(scale_factor)
        scale_factors.append(2 ** (octaves))
    
        
        scale_factors=scale_factors[:len(scale_factors)-drop_last_n_resolutions]
        scale_factor=scale_factors[-1]

        img = torch.randn((batch,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)

        for time, time_next in tqdm(time_pairs,disable=disable, desc = 'Coarse scale full diffusion',leave=True):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                scale_cond = torch.full((batch,), torch.tensor(scale_factor), device = device)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond,scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        

        for ind_s,scale_factor in tqdm(enumerate(reversed(scale_factors[:-1])),total=len(scale_factors)-1,desc='Coarse to fine scales',disable=disable):
                    
            if renoise_levels is not None: 
                try:
                    time_f=max(1,next(x[0] for x in enumerate((self.alphas_cumprod)) if x[1] < renoise_levels[ind_s]))
                except:
                    time_f=len(self.alphas_cumprod)
                
                times = torch.linspace(-1, int(time_f) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            else:
                times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            shape=(1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8)
            overlap_curr = (1+overlap//3)*3
            border_hr_curr=(border_hr//8)*8

            noise=torch.randn(shape)
            img=resize(img.cpu(),out_shape=(int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8),interp_method=interp.linear)
            alpha = self.alphas_cumprod[time_pairs[0][0]].cpu()
            alpha_next = self.alphas_cumprod[time_pairs[0][1]].cpu()
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            if renoise_levels is not None:
                c = (1 - renoise_levels[ind_s] - sigma ** 2).sqrt()
            else:
                c = (1 - alpha - sigma ** 2).sqrt()
            img = img.cpu() * alpha.sqrt() + \
                    c * noise

            
            patches,paths=patch_recursive(img,max_area=patch_size**2,overlap=32)
            if len(patches)>1:
                for i in tqdm(range(len(patches)),desc = 'Patches treated',leave=False,disable=disable):
                    
                    img=patches[i].cuda()
                    _,_,h,w=img.shape
                    
                    H,W=(math.ceil(h/8))*8,(math.ceil(w/8))*8
                    img=F.pad(img,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')


                    for time, time_next in time_pairs:
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            scale_cond = torch.full((batch,), torch.tensor(scale_factor), device = device)
                            self_cond = x_start if self.self_condition else None
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)
                            if time_next < 0:
                                img = x_start
                                continue
                        
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]
                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()
                            noise = torch.randn_like(img)

                            img = x_start * alpha_next.sqrt() + \
                                c * pred_noise + \
                                sigma * noise
                            
                    img=img[...,(H-h)//2:h+(H-h)//2,(W-w)//2:w+(W-w)//2]
                    patches[i]=img.detach().cpu()

                img=depatch(patches,paths,overlap=32).to(device)

            else: # don't patchify
                img=img.to(device)
                for time, time_next in tqdm(time_pairs, desc =  'Sampling steps',leave=False):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        scale_cond = torch.full((batch,), torch.tensor(scale_factor), device = device)
                        self_cond = x_start if self.self_condition else None
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True,omega=omega)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]
                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()
                        noise = torch.randn_like(img)

                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise

            if histogram_matching and gt is not None:

                gt_normalized=resize(gt.to(img.device),scale_factors=1/scale_factor)
                for i in range(20):
                    img=img+SW2_hist_grad(img,gt_normalized)
                img=img.cpu()


        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret
    


    def interp_scale(self, size,scales=2,octaves=3,time_ratio=.7, total_time_step_ratio = 1.,return_all_timesteps = False,noise=None):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        torch.manual_seed(0)
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        
        patch_size=4096
        border_hr=30
        overlap=100

        steps_count = int(total_time_step_ratio*+total_timesteps) + 1
        times = torch.linspace(-1, total_timesteps - 1, steps = steps_count)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        scale_factor = 2 ** (octaves)
        img = torch.randn((1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)
    
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True, omega=0)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                
                yield img

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(scale_factor)
        omegas=(torch.linspace(0,1,1+len(scale_factors))[1:]>.5).float()


        for ind,scale_factor in enumerate(reversed(scale_factors)):
            omega=omegas[ind]
            shape=(1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8)
            overlap_curr = (1+overlap//3)*3
            border_hr_curr=(border_hr//8)*8

            noise=torch.randn(shape)
            img=resize(img.cpu(),out_shape=(int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8),interp_method=interp.linear)
            alpha = self.alphas_cumprod[time_pairs[0][0]].cpu()
            alpha_next = self.alphas_cumprod[time_pairs[0][1]].cpu()
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha - sigma ** 2).sqrt()
            img = img.cpu() * alpha.sqrt() + \
                    c * noise

            yield img
            
            patches,paths=patch_recursive(img,max_area=patch_size**2,overlap=32)
            if len(patches)>1:
                for i in tqdm(range(len(patches)),desc = 'patches treated, scale %.2f'%scale_factor):
                    
                    img=patches[i].cuda()
                    _,_,h,w=img.shape
                    
                    H,W=(math.ceil(h/8))*8,(math.ceil(w/8))*8
                    img=F.pad(img,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')


                    for time, time_next in time_pairs:
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            self_cond = x_start if self.self_condition else None
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                            if time_next < 0:
                                img = x_start
                                continue
                            
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]

                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()

                            noise = torch.randn_like(img)

                            #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                            img = x_start * alpha_next.sqrt() + \
                                c * pred_noise + \
                                sigma * noise
                            
                    img=img[...,(H-h)//2:h+(H-h)//2,(W-w)//2:w+(W-w)//2]
                    patches[i]=img.detach().cpu()

                    yield img
                img=depatch(patches,paths,overlap=32).to(device)
                
                if False:
                    for i in tqdm(range(100),desc = 'histogram matching'):
                        img=img+.5*SW2_hist_grad(img,gt_resized)
                    img=img.cpu()
                
                #img=depatchify(p_hr,out_shape=shape,patch_size=patch_size,overlap=overlap_curr,border=border_hr_curr)
                yield img
            else: # don't patchify
                img=img.to(device)
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        self_cond = x_start if self.self_condition else None
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=omega)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
                        #img=img+SW2_hist_grad(img,resize(gt.to(img.device),scale_factors=1/scale_factor))

                        yield img
    

            

                  
            
            #imgs.append(img.cpu())

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        yield ret
        return ret
    


    def renoise_interp(self, size,scales=2,t_renoise=.5,octaves=3,time_ratio=.7, return_all_timesteps = False,noise=None):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        torch.manual_seed(0)
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        
        patch_size=4096
        border_hr=30
        overlap=100

        times = torch.linspace(-1, total_timesteps - 1, steps = total_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        times = torch.linspace(-1, int(total_timesteps*(t_renoise)) - 1, steps = sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs_renoise = list(zip(times[:-1], times[1:]))

        alpha = self.alphas_cumprod[time_pairs_renoise[0][0]].cpu()
        alpha_renoise= 1.*alpha
        alpha_next = self.alphas_cumprod[time_pairs_renoise[0][1]].cpu()
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c_renoise = (1 - alpha - sigma ** 2).sqrt()
        

        scale_factor = 2 ** (octaves)
        img = torch.randn((1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device),self_cond, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        #renoise
        noise=torch.randn(img.shape).to(img.device)
        img = img * alpha_renoise.sqrt() + \
                c_renoise * noise
        for time, time_next in tqdm(time_pairs_renoise, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=1)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        
                

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(scale_factor)


        for scale_factor in reversed(scale_factors):
            shape=(1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8)
            overlap_curr = (1+overlap//3)*3
            border_hr_curr=(border_hr//8)*8

            noise=torch.randn(shape)
            img=resize(img.cpu(),out_shape=(int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8),interp_method=interp.linear)
            alpha = self.alphas_cumprod[time_pairs[0][0]].cpu()
            alpha_next = self.alphas_cumprod[time_pairs[0][1]].cpu()
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha - sigma ** 2).sqrt()
            img = img.cpu() * alpha.sqrt() + \
                    c * noise


            patches,paths=patch_recursive(img,max_area=patch_size**2,overlap=32)
            if len(patches)>1:#patchified
                for i in tqdm(range(len(patches)),desc = 'patches treated, scale %.2f'%scale_factor):
                    
                    img=patches[i].cuda()
                    _,_,h,w=img.shape
                    
                    H,W=(math.ceil(h/8))*8,(math.ceil(w/8))*8
                    img=F.pad(img,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')


                    for time, time_next in time_pairs:
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            self_cond = x_start if self.self_condition else None
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True)

                            if time_next < 0:
                                img = x_start
                                continue
                            
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]

                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()

                            noise = torch.randn_like(img)

                            #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                            img = x_start * alpha_next.sqrt() + \
                                c * pred_noise + \
                                sigma * noise
                     #renoise
                    noise=torch.randn(img.shape).to(img.device)
                    img = img * alpha_renoise.sqrt() + \
                            c_renoise * noise
                    for time, time_next in tqdm(time_pairs_renoise, desc = 'sampling loop time step'):
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            self_cond = x_start if self.self_condition else None
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=1)

                            if time_next < 0:
                                img = x_start
                                continue
                            
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]

                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()

                            noise = torch.randn_like(img)
                            #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                            img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                                sigma * noise
                            
                    img=img[...,(H-h)//2:h+(H-h)//2,(W-w)//2:w+(W-w)//2]
                    patches[i]=img.detach().cpu()

                img=depatch(patches,paths,overlap=32).to(device)

                if False:
                    for i in tqdm(range(100),desc = 'histogram matching'):
                        img=img+.5*SW2_hist_grad(img,gt_resized)
                    img=img.cpu()
                
                #img=depatchify(p_hr,out_shape=shape,patch_size=patch_size,overlap=overlap_curr,border=border_hr_curr)

            else: # patchified
                img=img.to(device)
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        self_cond = x_start if self.self_condition else None
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device),  self_cond, clip_x_start = True, rederive_pred_noise = True)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
                #renoise
                noise=torch.randn(img.shape).to(img.device)
                img = img * alpha_renoise.sqrt() + \
                        c_renoise * noise
                for time, time_next in tqdm(time_pairs_renoise, desc = 'sampling loop time step'):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        self_cond = x_start if self.self_condition else None
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=1)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
            

            

                  
            
            #imgs.append(img.cpu())

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    

    
    def spatial_interp(self, size,scales=2,octaves=3,time_ratio=.7, return_all_timesteps = False,noise=None,omega=None,patch_size=3000, total_time_step_ratio = 1.):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        #torch.manual_seed(0)
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        if omega is None:
            omega=torch.linspace(0,1,size[1]).view(1,1,1,-1).to(device)

        
        border_hr=30
        overlap=100
        steps_count = int(total_time_step_ratio*+total_timesteps) + 1
        times = torch.linspace(-1, total_timesteps - 1, steps = steps_count)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        scale_factor = 2 ** (octaves)
        img = torch.randn((1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                omega=resize(omega,out_shape=(img.shape[-2],img.shape[-1])).to(img.device)
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond,torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                if time_next < 0:
                    img = x_start
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                
                yield img

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(scale_factor)


        for scale_factor in reversed(scale_factors):
            shape=(1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8)
            overlap_curr = (1+overlap//3)*3
            border_hr_curr=(border_hr//8)*8

            noise=torch.randn(shape)
            img=resize(img.cpu(),out_shape=(int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8),interp_method=interp.linear)
            alpha = self.alphas_cumprod[time_pairs[0][0]].cpu()
            alpha_next = self.alphas_cumprod[time_pairs[0][1]].cpu()
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha - sigma ** 2).sqrt()
            img = img.cpu() * alpha.sqrt() + \
                    c * noise
            
            yield img
            
            patches,paths=patch_recursive(img,max_area=patch_size**2,overlap=32)
            if len(patches)>1:
                omega=resize(omega.cpu(),out_shape=(img.shape[-2],img.shape[-1]))
                patches_omega,_=patch_recursive(omega,max_area=patch_size**2,overlap=32)
                for i in tqdm(range(len(patches)),desc = 'patches treated, scale %.2f'%scale_factor):
                    omega_p=patches_omega[i].to(device)
                    
                    img=patches[i].to(device)
                    _,_,h,w=img.shape
                    
                    H,W=(math.ceil(h/8))*8,(math.ceil(w/8))*8
                    img=F.pad(img,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')
                    omega_p=F.pad(omega_p,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')


                    for time, time_next in time_pairs:
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            self_cond = x_start if self.self_condition else None
        
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega_p)

                            if time_next < 0:
                                img = x_start
                                continue
                            
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]

                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()

                            noise = torch.randn_like(img)

                            #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                            img = x_start * alpha_next.sqrt() + \
                                c * pred_noise + \
                                sigma * noise
                            
                            yield img
                            
                    img=img[...,(H-h)//2:h+(H-h)//2,(W-w)//2:w+(W-w)//2]
                    patches[i]=img.detach().cpu()

                img=depatch(patches,paths,overlap=32).to(device)

                if False:
                    for i in tqdm(range(100),desc = 'histogram matching'):
                        img=img+.5*SW2_hist_grad(img,gt_resized)
                    img=img.cpu()
                
                #img=depatchify(p_hr,out_shape=shape,patch_size=patch_size,overlap=overlap_curr,border=border_hr_curr)
                yield img
            else: # don't patchify
                img=img.to(device)
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        self_cond = x_start if self.self_condition else None
                        omega=resize(omega.to(device),out_shape=(img.shape[-2],img.shape[-1]))
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device),  self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
                        #img=img+SW2_hist_grad(img,resize(gt.to(img.device),scale_factors=1/scale_factor))

                        yield img

            

                  
            
            #imgs.append(img.cpu())

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        yield ret
        return ret

    @torch.inference_mode()
    def scale_inference(self, y,scale=1,omega=0, zoom=1,time_ratio=1.,zeta=0, return_all_timesteps = False,noise=None):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        batch,c,h,w=y.shape
        shape=(batch,c,zoom*h,zoom*w)
        
        
        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        if noise is None:
            img = torch.randn(shape, device = device)
        else:
            img=noise.to(device)
        
        
        if time_ratio!=1:
            alpha = self.alphas_cumprod[time_pairs[0][0]]
            alpha_next = self.alphas_cumprod[time_pairs[0][1]]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            img = resize(self.normalize(y),out_shape=(noise.shape[-2],noise.shape[-1]),interp_method=interp.linear) * alpha_next.sqrt() + \
                    c * noise.to(device)
            print('img:',alpha.item(),'  noise:',c.item())
        imgs = [img.cpu()]


        if zeta!=0:
            ones = torch.ones(img.shape).requires_grad_(True).to(device)
            lr=resize(ones,out_shape=(h,w),interp_method=interp.gaussian)
            gradient = torch.autograd.grad(outputs=lr,grad_outputs=torch.ones(lr.shape).requires_grad_(True).to(device), inputs=ones, create_graph=True, retain_graph=False)[0]
            norm=gradient.detach()
            norm=norm/norm.mean()



            

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            with torch.no_grad():
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, scale, self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                
                
            if zeta!=0:
                img = img.clone().detach().requires_grad_(True)
                lr=resize(img,out_shape=(h,w),interp_method=interp.linear)
                #hrlr=resize(resize(self.normalize(y),out_shape=(img.shape[-2],img.shape[-1])),out_shape=(h,w),interp_method=interp.gaussian)
                difference = (self.normalize(y) - lr)
                #loss = .5*(difference**2).sum()
                #gradient = torch.autograd.grad(outputs=loss, inputs=img, create_graph=True, retain_graph=False, only_inputs=True)[0]
                gradient= resize(difference,out_shape=(h*zoom,w*zoom),interp_method=interp.linear)
                #img=img-zeta*gradient/(difference**2).mean().detach()

                img=img+zeta*gradient#/norm


            
                
            
            #imgs.append(img.cpu())

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    




    def stylize(self, size,input,gt,zeta=.5,output_texture_resolution_ind=0,scales=2,omega=0,octaves=3,time_ratio=.7, return_all_timesteps = False,noise=None):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        torch.manual_seed(0)
        gt=self.normalize(gt)
        input=self.normalize(input)
        shape=(1,3,size[0],size[1])
        batch=1
        imgs=[]
        img_steps = []
        
        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(scale_factor)
        scale_factors.append(2**octaves)
        scale_factors.reverse()
        scale_factors=scale_factors[:len(scale_factors)-output_texture_resolution_ind]
        scale_factors_apply_img=[f/scale_factors[-1] for f in scale_factors]

        


        
        patch_size=4096
        border_hr=30
        overlap=100
        scale_factor=scale_factors[0]
        shape=(int(size[0]/scale_factor*scale_factors[-1])//8*8,int(size[1]/scale_factor*scale_factors[-1])//8*8)
        colored_input=resize(input,out_shape=shape)
        colored_input=colored_input.to(device)
        gt=gt.to(device)
        with torch.no_grad():
            if True:
                for i in tqdm(range(1000),desc='pre-processing histogram matching'):
                    colored_input=colored_input+.5*SW2_hist_grad(colored_input,gt)
        #img=1.*colored_input
        img=torch.randn((1,3,int(size[0]/scale_factor*scale_factors[-1])//8*8,int(size[1]/scale_factor*scale_factors[-1])//8*8)).to(device)
                
                

        times = torch.linspace(-1, int(total_timesteps*(time_ratio)) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]



        for scale_factor in scale_factors:
            shape=(1,3,int(size[0]/scale_factor*scale_factors[-1])//8*8,int(size[1]/scale_factor*scale_factors[-1])//8*8)
            overlap_curr = (1+overlap//3)*3
            border_hr_curr=(border_hr//8)*8

            noise=torch.randn(shape)
            img=resize(img.cpu(),out_shape=(int(size[0]/scale_factor*scale_factors[-1])//8*8,int(size[1]/scale_factor*scale_factors[-1])//8*8),interp_method=interp.linear)
            alpha = self.alphas_cumprod[time_pairs[0][0]].cpu()
            alpha_next = self.alphas_cumprod[time_pairs[0][1]].cpu()
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha - sigma ** 2).sqrt()
            img = img.cpu() * alpha.sqrt() + \
                    c * noise


            patches,paths=patch_recursive(img,max_area=patch_size**2,overlap=32)
            if len(patches)>1:
                patches_input,_=patch_recursive(colored_input,max_area=patch_size**2,overlap=32)
                for i in tqdm(range(len(patches)),desc = 'patches treated, scale %.2f'%scale_factor):
                    
                    img=patches[i].cuda()
                    crop_in=patches_input[i].cuda()
                    _,_,h,w=img.shape
                    
                    H,W=(math.ceil(h/8))*8,(math.ceil(w/8))*8
                    img=F.pad(img,pad=((W-w)//2,W-w-(W-w)//2,(H-h)//2,H-h-(H-h)//2),mode='replicate')


                    for time, time_next in time_pairs:
                        with torch.no_grad():
                            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                            self_cond = x_start if self.self_condition else None
                            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True, omega=omega)

                            if time_next < 0:
                                img = x_start
                                continue
                            
                            alpha = self.alphas_cumprod[time]
                            alpha_next = self.alphas_cumprod[time_next]

                            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                            c = (1 - alpha_next - sigma ** 2).sqrt()

                            noise = torch.randn_like(img)

                            #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                            img = x_start * alpha_next.sqrt() + \
                                c * pred_noise + \
                                sigma * noise
                            
                            yield[ img,None]
                    

                        img=depatch(patches,paths,overlap=32).to(device)
                        yield [img,None]

                        img = img.clone().detach().requires_grad_(True)
                        lr=resize(img,out_shape=(crop_in.shape[-2],crop_in.shape[-1]),interp_method=interp.linear)
                        difference = (crop_in - lr)
                        #loss = .5*(difference**2).sum()
                        #gradient = torch.autograd.grad(outputs=loss, inputs=img, create_graph=True, retain_graph=False, only_inputs=True)[0]
                        gradient= resize(difference,out_shape=(H,W),interp_method=interp.linear)
                        print(gradient.shape,img.shape)
                        #img=img-zeta*gradient/(difference**2).mean().detach()
                        img=img+zeta*gradient#/norm

                    img=img[...,(H-h)//2:h+(H-h)//2,(W-w)//2:w+(W-w)//2]
                    patches[i]=img.detach().cpu()
                    
                    
                    yield [img,None]

            else: # don't patchify
                img=img.to(device)
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    with torch.no_grad():
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        self_cond = x_start if self.self_condition else None
                        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, torch.tensor(scale_factor).unsqueeze(0).to(device), self_cond, clip_x_start = True, rederive_pred_noise = True,omega=omega)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.alphas_cumprod[time]
                        alpha_next = self.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
                        
                        img = img.clone().detach().requires_grad_(True)
                        lr=resize(img,out_shape=(colored_input.shape[-2],colored_input.shape[-1]),interp_method=interp.linear)
                        difference = (colored_input - lr)
                        #loss = .5*(difference**2).sum()
                        #gradient = torch.autograd.grad(outputs=loss, inputs=img, create_graph=True, retain_graph=False, only_inputs=True)[0]
                        gradient= resize(difference,out_shape=(img.shape[-2],img.shape[-1]),interp_method=interp.linear)
                        #img=img-zeta*gradient/(difference**2).mean().detach()
                        img=img+zeta*gradient#/norm
                        #img=img+SW2_hist_grad(img,resize(gt.to(img.device),scale_factors=1/scale_factor))
                        
                        
                        
                        yield [img,None]
    

            

                  
            
            #imgs.append(img.cpu())
            img_steps.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        
        ret = self.unnormalize(ret)
        img_steps.append(ret)
        yield [ret, img_steps]
    


    @torch.inference_mode()
    def sample(self, batch_size = 16, scale=1, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        #sample_fn =  self.ddim_sample
        return sample_fn((batch_size, channels, h, w),scale=scale, return_all_timesteps = return_all_timesteps)
    

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, scale, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, scale).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, scale, x_self_cond)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img,scale, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t,scale, *args, **kwargs)

# dataset classes
class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        octaves = 3,
        scales = 2
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.scales=scales
        self.octaves=octaves
        path = self.paths
        for path in self.paths:
            x = T.ToTensor()(Image.open(path))
            c,h,w=x.shape
            assert h/2 ** (octaves) >= image_size[-2] and w/2 ** (octaves) >= image_size[-1]
        self.octaves=octaves
        self.scales=scales
        scale_factors=[]
        for octave in range(octaves):
            for scale in range(scales):
                scale_factor = 2 ** (scale / scales + octave)
                scale_factors.append(torch.tensor(scale_factor).half())
        
        scale_factor = 2 ** (octaves)
        scale_factors.append(torch.tensor(scale_factor).half())

        self.crop= T.RandomCrop((image_size[-2],image_size[-1]))

        self.scale_factors=scale_factors
        self.images=[T.ToTensor()(Image.open(path)) for path in self.paths]
        self.images=[[resize(img,interp_method=interp.linear,scale_factors=1/scale) for scale in scale_factors] for img in self.images]
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        if index>(3*self.__len__()//4):
            index=len(self.scale_factors)-1
        else:
            index=index%len(self.scale_factors)
        img = sample(self.images,1)[0][index]
        img=self.crop(img)
        scale=self.scale_factors[index]
        return img, scale
        

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        octaves=3,
        scales=2
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        #assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        if folder is not None:
            self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to,octaves=octaves,scales=scales)
        else:
            self.ds =DummyDset(100)

        #assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 8,drop_last=True)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from SIMuLDiTex.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'],strict=False)

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def load_model(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'],strict=False)


    def tune_ms_noise(self):
        histogram_matching=False
        vgg = models.vgg19(pretrained=False).features.cuda()
        vgg=vgg[:12]
        MEAN = (0.485, 0.456, 0.406)
        STD=(0.229, 0.224, 0.225)
        mean = torch.as_tensor(MEAN).view(-1, 1, 1).cuda()
        std = torch.as_tensor(STD).view(-1, 1, 1).cuda()

        def prep(x):
            x=x/2+.5
            #return x.sub_(mean).div_(std)
            return x.sub_(mean).div_(std).mul_(255)



        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.Conv2d):
                vgg[i] = nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=0)


        pretrained_dict = torch.load('../DIP_texture/vgg.pth')
        for param, item in zip(vgg.parameters(), pretrained_dict.keys()):
            param.data = pretrained_dict[item].type(torch.FloatTensor).cuda()
            
        vgg.eval()
        vgg=vgg.cuda()
        vgg.requires_grad_(False)
        outputs = {}
        def save_output(name):
            def hook(module, module_in, module_out):
                outputs[name] = module_out
            return hook
        layers = [1,6,11]#,1,6, 11, 20, 29]
        layers_weights = [1/n**2 for n in [64,128,256]]#64,128,256,512,512]]
        for layer in layers:
            handle = vgg[layer].register_forward_hook(save_output(layer))

        gt = self.model.normalize(self.ds.images[0].unsqueeze(0).to(self.device))
        rt= torch.full((len(self.ds.scale_factors)-1,),.5).to(self.device)
        rt.requires_grad = True 
        opt=Adam([rt],lr=5*10**-3,betas=(.5,.5))

        N=100
        scales=self.ds.scales
        octaves=self.ds.octaves

        pyr_gt=gaussian_pyramid(gt,scales,octaves)
        #pyr_gt=[pyr_gt[-1]]*len(pyr_gt)
        #for gt in pyr_gt:
        #    show(denorm(gt[0]))
        targets_gt=[]

        for gt_down in pyr_gt:
            vgg(prep(gt_down))
            out_vgg_real = [outputs[key] for key in layers] 
            style_targets = [GramMatrix()(outputs[key]) for key in layers] 
            targets_gt.append(style_targets)

        self.model.model.eval()






        for i in tqdm(range(N)):
            opt.zero_grad()
            device, total_timesteps, sampling_timesteps, eta, objective = self.model.device, self.model.num_timesteps, self.model.sampling_timesteps, self.model.ddim_sampling_eta, self.model.objective
            #torch.manual_seed(0)
            batch=1
            size=(512,512)
            shape=(batch,3,size[0],size[1])
            imgs=[]
            coarse_scale_steps=total_timesteps 
            times = torch.linspace(-1, total_timesteps - 1, steps = coarse_scale_steps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]





            scale_factors=[]
            for octave in range(octaves):
                for scale in range(scales):
                    scale_factor = 2 ** (scale / scales + octave)
                    scale_factors.append(scale_factor)
            scale_factors.append(2 ** (octaves))
            
            scale_factors=scale_factors[:len(scale_factors)]
            scale_factor=scale_factors[-1]

            img = torch.randn((batch,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8), device = device)
            for time, time_next in time_pairs:
                with torch.no_grad():
                    time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                    scale_cond = torch.full((batch,), torch.tensor(scale_factor), device = device)
                    self_cond = x_start if self.model.self_condition else None
                    pred_noise, x_start, *_ = self.model.model_predictions(img, time_cond,scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, omega=0)
                    if time_next < 0:
                        img = x_start
                        continue
                    alpha = self.model.alphas_cumprod[time]
                    alpha_next = self.model.alphas_cumprod[time_next]
                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    noise = torch.randn_like(img)
                    #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
            
            
            for ind_s,scale_factor in enumerate(reversed(scale_factors[:-1])):
                try:
                    time_f=max(1,next(x[0] for x in enumerate((self.model.alphas_cumprod)) if x[1] < rt[ind_s]))
                except:
                    time_f=len(self.model.alphas_cumprod)
                
                times = torch.linspace(-1, int(time_f) - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
                times = list(reversed(times.int().tolist()))
                time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


                shape=(1,3,int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8)

                noise=torch.randn(shape).to(self.device)
                img=resize(img,out_shape=(int(size[0]/scale_factor)//8*8,int(size[1]/scale_factor)//8*8),interp_method=interp.linear)
                alpha = self.model.alphas_cumprod[time_pairs[0][0]].to(self.device)
                alpha_next = self.model.alphas_cumprod[time_pairs[0][1]].to(self.device)
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt().to(self.device)
                c = (1 - rt[ind_s] - sigma ** 2).sqrt()
                img = img * alpha.sqrt() + \
                        c * noise

                img=img.to(device)
                for time, time_next in time_pairs:
                    if True:
                        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                        scale_cond = torch.full((batch,), torch.tensor(scale_factor), device = device)
                        self_cond = x_start if self.model.self_condition else None
                        pred_noise, x_start, *_ = self.model.model_predictions(img, time_cond, scale_cond, self_cond, clip_x_start = True, rederive_pred_noise = True,omega=0)

                        if time_next < 0:
                            img = x_start
                            continue
                        
                        alpha = self.model.alphas_cumprod[time]
                        alpha_next = self.model.alphas_cumprod[time_next]

                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        c = (1 - alpha_next - sigma ** 2).sqrt()

                        noise = torch.randn_like(img)
                        #x_start=x_start+SW2_hist_grad(x_start,gt_resized)
                        img = x_start * alpha_next.sqrt() + \
                            c * pred_noise + \
                            sigma * noise
                        #img=img+SW2_hist_grad(img,resize(gt.to(img.device),scale_factors=1/scale_factor))
                if histogram_matching and gt is not None:

                    gt_normalized=resize(gt.to(img.device),scale_factors=1/scale_factor)
                    for i in range(20):
                        img=img+SW2_hist_grad(img,gt_normalized)
                    #img=img.cpu()
        



            pyr=gaussian_pyramid(img,scales=scales,octaves=octaves)
            for im_res,target in zip(pyr,targets_gt):
                
                
                vgg(prep(im_res)) 
                style = [GramMatrix()(outputs[key]) for key in layers] 
                style_losses = [1.*layers_weights[a] * torch.mean((style[a]- target[a].detach())**2) / len(layers) for a in range(len(layers))]
            
                loss=sum(style_losses)



            loss.backward()
        
            opt.step()
            with torch.no_grad():
                rt.clamp_(0,0.9999)
        print(rt)
        json_filename=os.path.join(self.results_folder,'args.json')     
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)
        data['ms_noises'] = [e.item() for e in rt]

        with open(json_filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return   


         

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data,scale = next(self.dl)
                    data,scale=data.to(device),scale.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data,scale)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples,len(self.ds.scale_factors))# self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=len(self.ds.scale_factors),scale=self.ds.scale_factors), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
