# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/NVlabs/SegFormer.
#
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .base import BaseHead
from .modules import ConvBNReLU

model_urls = {
    "cityscapes": {
        # same weights as provided by official SegFormer repo: https://github.com/NVlabs/SegFormer
        "mit_b5": "https://data.vision.ee.ethz.ch/brdavid/refign/segformer.b5.1024x1024.city.160k.pth",
    }
}


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self,
                 in_channels: List[int],
                 in_index: Union[List[int], int],
                 num_classes: int,
                 input_transform: Optional[str] = None,
                 channels: int = 256,
                 dropout_ratio: float = 0.1,
                 pretrained: Optional[str] = None,
                 ):
        super().__init__(num_classes, in_index, input_transform)

        self.in_channels = in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvBNReLU(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
        )

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.init_weights()
        self.load_pretrained(pretrained)

    def init_weights(self):
        # mmseg init
        nn.init.normal_(self.linear_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear_pred.bias, 0)
        for m in self.modules():
            # initialize ConvBNReLU as in mmsegmentation
            if isinstance(m, ConvBNReLU) and not m.depthwise_separable:
                nn.init.kaiming_normal_(
                    m.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                    nn.init.constant_(m.conv.bias, 0)
                if m.use_norm:
                    if hasattr(m.bn, 'weight') and m.bn.weight is not None:
                        nn.init.constant_(m.bn.weight, 1)
                    if hasattr(m.bn, 'bias') and m.bn.bias is not None:
                        nn.init.constant_(m.bn.bias, 0)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32

        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.functional.interpolate(
            _c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.functional.interpolate(
            _c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.functional.interpolate(
            _c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear_pred(x)

        return x

    def load_pretrained(self, pretrained):
        if pretrained is None:
            return

        if pretrained == 'cityscapes':
            pretrained = model_urls['cityscapes']['mit_b5']

        if os.path.exists(pretrained):
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrained)):
            checkpoint = torch.load(os.path.join(os.environ.get(
                'TORCH_HOME', ''), 'hub', pretrained), map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrained, progress=True, map_location=lambda storage, loc: storage)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('decode_head.'):
                new_k = k.replace('decode_head.', '')
                new_state_dict[new_k] = v
            elif k.startswith('head.'):
                new_k = k.replace('head.', '')
                new_state_dict[new_k] = v
            else:
                pass
        new_state_dict = {k: v for k, v in new_state_dict.items()
                      if not k.startswith('conv_seg.')}
        self.load_state_dict(new_state_dict, strict=True)
