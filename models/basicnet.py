import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


from models.conformer import Conformer_backbone, PatchEmbedding, TransformerEncoder


class Classifier(nn.Sequential):
    def __init__(self, feat_size, n_classes):
        super().__init__()
        
        # global average pooling NOT USED BUT REMOVING WOULD AFFECT RANDOM SEED
        self.clshead = nn.Sequential(
            Reduce('b n e -> b n', reduction='mean'),
            nn.LayerNorm(40),
            nn.Linear(40, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class BasicNet(nn.Sequential): # Backbone has output dimension [batch_size, feature_size] # Classifier takes backbone output and outputs dimension [batch_size, n_classes]
    def __init__(self, backbone, n_classes, feat_size):
        super().__init__()
        if backbone == 'conformer':
            self.backbone = Conformer_backbone(feat_size=feat_size)
        self.classifier = Classifier(feat_size, n_classes)
    
    def forward(self, x):
        # print(f'Input {x.shape}')
        x = self.backbone(x)
        # print(f'Backbone output {x.shape}')
        out = self.classifier(x)
        # print(f'Classifier output {out.shape}')
        return x, out #outputs backbone features, classification result
