import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, padding=None,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 depthwise_separable=False, inplace=True, affine=True):
        super().__init__()
        padding = dilation * \
            (kernel_size - 1) // 2 if padding is None else padding
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        self.depthwise_separable = depthwise_separable
        if bias == 'auto':
            bias = not self.use_norm
        if depthwise_separable:
            assert kernel_size > 1
            assert groups == 1  # not sure how to handle this
            self.depthwise_conv = ConvBNReLU(in_channels, in_channels, kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, groups=in_channels,
                                             norm_layer=norm_layer, activation_layer=activation_layer)
            self.pointwise_conv = ConvBNReLU(in_channels, out_channels, 1,
                                             norm_layer=norm_layer, activation_layer=activation_layer)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=bias)
            if self.use_norm:
                self.bn = norm_layer(out_channels, affine=affine)

        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        if self.depthwise_separable:
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)
            if self.use_norm:
                x = self.bn(x)
            if self.use_activation:
                x = self.activation(x)
        return x
