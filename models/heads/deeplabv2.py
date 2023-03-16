import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .base import BaseHead

model_urls = {
    # DeepLabv2 trained on Cityscapes
    "cityscapes": "https://data.vision.ee.ethz.ch/brdavid/coma/deeplabv2_cityscapes.pth"
}


class DeepLabV2Head(BaseHead):

    def __init__(self,
                 in_channels: int,
                 in_index: Union[List[int], int],
                 num_classes: int,
                 input_transform: Optional[str] = None,
                 dilation_series: List[int] = [6, 12, 18, 24],
                 padding_series: List[int] = [6, 12, 18, 24],
                 pretrained: Optional[str] = None):
        super().__init__(num_classes, in_index, input_transform)
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        self.init_weights()
        self.load_pretrained(pretrained)

    def init_weights(self):
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self._transform_inputs(x)
        return sum(stage(x) for stage in self.conv2d_list)

    def load_pretrained(self, pretrained):
        if pretrained is None:
            return

        if pretrained == 'cityscapes':
            pretrained = model_urls['cityscapes']

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
            if k.startswith('head.'):
                new_k = k.replace('head.', '')
                new_state_dict[new_k] = v
            else:
                pass
        self.load_state_dict(new_state_dict, strict=True)
