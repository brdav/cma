# Copyright (c) OpenMMLab. All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/open-mmlab/mmsegmentation.
#
from typing import List, Optional, Sequence, Union

import torch.nn as nn

from .base import BaseHead
from .modules import ConvBNReLU


class ProjectionHead(BaseHead):
    """Projection Head for feature dimension reduction in contrastive loss.
    """

    def __init__(self,
                 in_channels: Union[int, List[int]],
                 in_index: Union[List[int], int],
                 input_transform: Optional[str] = None,
                 channels: int = 128,
                 kernel_size: int = 1):
        super().__init__(-1, in_index, input_transform)
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size

        if self.input_transform == 'multiple_select':
            assert isinstance(self.in_channels, Sequence)
            self.convs = nn.ModuleList([])
            for i in range(len(self.in_channels)):
                self.convs.append(nn.Sequential(
                    ConvBNReLU(
                        self.in_channels[i],
                        self.in_channels[i],
                        kernel_size=kernel_size,
                        norm_layer=None,
                        activation_layer=nn.ReLU),
                    ConvBNReLU(
                        self.in_channels[i],
                        self.channels,
                        kernel_size=kernel_size,
                        norm_layer=None,
                        activation_layer=None)))
        else:
            if self.input_transform == 'resize_concat':
                if isinstance(self.in_channels, Sequence):
                    in_channels = sum(self.in_channels)
                else:
                    in_channels = self.in_channels
            else:
                in_channels = self.in_channels
            self.convs = nn.Sequential(
                ConvBNReLU(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    norm_layer=None,
                    activation_layer=nn.ReLU),
                ConvBNReLU(
                    in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    norm_layer=None,
                    activation_layer=None))

        # mmseg init
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
        nn.init.normal_(
            self.convs[-1].conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.convs[-1].conv.bias, 0)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        if isinstance(x, list):
            # multiple_select
            output = [self.convs[i](x[i]) for i in range(len(x))]
        else:
            # resize_concat or single_select
            output = self.convs(x)
        return output
