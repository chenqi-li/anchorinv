# We use eeg-conformer as backbone for our experiments. https://github.com/eeyhsong/EEG-Conformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, norm_layer, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        if norm_layer == 'batch':
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 64), (1, 4)),
                nn.Conv2d(40, 40, (8, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                nn.Dropout(0.5),
            )
        elif norm_layer == 'layer':
            self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 64), (1, 4)),
            nn.Conv2d(40, 40, (8, 1), (1, 1)),
            nn.LayerNorm((40, 1, 945)),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
            )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class PatchEmbeddingBCI(nn.Module):
    def __init__(self, norm_layer, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        if norm_layer == 'batch':
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (22, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                nn.Dropout(0.5),
            )
        elif norm_layer == 'layer': # need number for x
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (22, 1), (1, 1)),
                nn.LayerNorm(40, 1, x),
                nn.ELU(),
                nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                nn.Dropout(0.5),
            )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class PatchEmbeddingGRABM(nn.Module):
    def __init__(self, norm_layer, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        if norm_layer == 'batch':
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 64), (1, 1)),
                nn.Conv2d(40, 40, (28, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 75), (1, 20)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                nn.Dropout(0.5),
            )
        elif norm_layer == 'layer': # need number for x
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 64), (1, 1)),
                nn.Conv2d(40, 40, (28, 1), (1, 1)),
                nn.LayerNorm(40, 1, x),
                nn.ELU(),
                nn.AvgPool2d((1, 75), (1, 20)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                nn.Dropout(0.5),
            )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class BackboneHead(nn.Sequential):
    def __init__(self):
        super().__init__()
    

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Conformer_backbone(nn.Sequential):
    def __init__(self, norm_layer='batch', emb_size=40, depth=6, feat_size=None, **kwargs):
        if feat_size==2360:
            super().__init__(
                PatchEmbedding(norm_layer, emb_size),
                TransformerEncoder(depth, emb_size),
                BackboneHead()
            )
        elif feat_size==2440:
            super().__init__(
                PatchEmbeddingBCI(norm_layer, emb_size),
                TransformerEncoder(depth, emb_size),
                BackboneHead()
            )
        elif feat_size==2320:
            super().__init__(
                PatchEmbeddingGRABM(norm_layer, emb_size),
                TransformerEncoder(depth, emb_size),
                BackboneHead()
            )