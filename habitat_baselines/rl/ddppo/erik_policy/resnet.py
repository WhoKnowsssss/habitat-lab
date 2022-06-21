#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from typing import List, Optional, Type, Union, cast

import einops.layers.torch
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d


class SpectralPool2d(nn.Module):
    def __init__(self, strides: numbers.Number):
        super().__init__()

        if isinstance(strides, numbers.Number):
            self.strides = (strides, strides)
        elif isinstance(strides, (list, tuple)):
            assert len(strides) == 2
            self.strides = tuple(strides)
        else:
            raise RuntimeError(f"Invalid strides: '{strides}'")

    def forward(self, x):
        *_, h, w = x.size()

        output_height = int(h // self.strides[0])
        output_height -= output_height % 2
        output_width = int(w // self.strides[1])

        output_height = max(output_height, 2)
        output_width = max(output_width, 2)

        lower_height = output_height // 2
        upper_height = h - lower_height
        upper_width = output_width // 2 + 1

        x = torch.fft.rfft2(x)
        x = torch.cat(
            (
                x[..., :lower_height, :upper_width],
                x[..., upper_height:, :upper_width],
            ),
            dim=2,
        )
        return torch.fft.irfft2(x, s=(output_height, output_width))


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=int((3 - 1) * dilation / 2),
        bias=bias,
        groups=groups,
        dilation=dilation,
    )


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1
) -> Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x).view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        downsample=None,
        cardinality=1,
        layer_scale_init: float = 1e-4,
        dilation: int = 1,
    ):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(
                inplanes, planes, stride, groups=cardinality, dilation=dilation
            ),
            norm_layer(planes),
            nn.ReLU(True),
            conv3x3(
                planes,
                planes,
                groups=cardinality,
                dilation=dilation,
            ),
            norm_layer(planes),
        )
        self.downsample = downsample
        self.layer_scale_gamma = nn.Parameter(
            torch.full((planes, 1, 1), layer_scale_init)
        )

    def _add_residual(self, residual, out):
        return torch.relu_(
            torch.addcmul(residual, self.layer_scale_gamma, out)
        )

    def forward(self, x):
        residual = x

        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self._add_residual(residual, out)


class SEBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        downsample=None,
        cardinality=1,
        layer_scale_init: float = 1e-4,
        dilation: int = 1,
    ):
        super().__init__(
            inplanes,
            planes,
            norm_layer,
            stride,
            downsample,
            cardinality,
            layer_scale_init,
            dilation,
        )
        self.se = SE(planes, r=4)

    def _add_residual(self, residual, out):
        return super()._add_residual(residual, out.mul_(self.se(out)))


def _build_bottleneck_branch(
    inplanes: int,
    planes: int,
    stride: int,
    expansion: int,
    norm_layer,
    groups: int = 1,
    dilation: int = 1,
) -> Sequential:
    return nn.Sequential(
        conv1x1(inplanes, planes, dilation=dilation),
        norm_layer(planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups, dilation=dilation),
        norm_layer(planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion, dilation=dilation),
        norm_layer(planes * expansion),
    )


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer,
        stride: int = 1,
        downsample: Optional[Sequential] = None,
        cardinality: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes,
            planes,
            stride,
            self.expansion,
            norm_layer,
            groups=cardinality,
            dilation=dilation,
        )
        self.downsample = downsample

    def _add_residual(self, residual, out):
        return torch.relu_(out.add_(residual))

    def _impl(self, x: Tensor) -> Tensor:
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self._add_residual(identity, out)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super().__init__(
            inplanes, planes, norm_layer, stride, downsample, cardinality
        )

        self.se = _build_se_branch(planes * self.expansion)

    def _add_residual(self, residual, out):
        return torch.relu_(torch.addcmul(residual, out, self.se(out)))


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


Block = Union[Type[Bottleneck], Type[BasicBlock]]


class SpaceToDepth(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self._impl = einops.layers.torch.Rearrange(
            "b c (h h2) (w w2) -> b (c h2 w2) h w",
            h2=patch_size,
            w2=patch_size,
        )

    def forward(self, x):
        return self._impl(x)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_planes: int,
        block: Block,
        layers: List[int],
        norm_layer,
        cardinality: int = 1,
        mode="v1",
    ) -> None:
        super(ResNet, self).__init__()
        self._mode = mode

        if self._mode == "v1":
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    base_planes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                norm_layer(base_planes),
                nn.ReLU(True),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.stem = nn.Sequential(
                SpaceToDepth(4),
                conv1x1(in_channels * 16, base_planes),
                norm_layer(base_planes),
            )
        self.cardinality = cardinality

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(
            block, base_planes, norm_layer, layers[0]
        )
        self.layer2 = self._make_layer(
            block, base_planes * 2, norm_layer, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, base_planes * 2 * 2, norm_layer, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, base_planes * 2 * 2 * 2, norm_layer, layers[3], stride=2
        )

    def _make_layer(
        self,
        block: Block,
        planes: int,
        norm_layer,
        blocks: int,
        stride: int = 1,
    ) -> Sequential:
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self._mode == "v1":
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    norm_layer(self.inplanes),
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=stride,
                        stride=stride,
                        bias=True,
                    ),
                )
                layers.append(downsample)
                downsample = None
                stride = 1
                self.inplanes = planes * block.expansion

        layers.append(
            block(
                self.inplanes,
                planes,
                norm_layer,
                stride,
                downsample,
                cardinality=self.cardinality,
                dilation=1 if self._mode == "v1" else 2,
            )
        )
        self.inplanes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer,
                    dilation=1 if self._mode == "v1" else 2,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        if self._mode == "v1":
            x = self.conv1(x)
            x = self.maxpool(x)
            x = cast(Tensor, x)
        else:
            x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _resnet_v1(in_channels, base_planes, ngroups, block_type, layers):
    model = ResNet(
        in_channels,
        base_planes,
        block_type,
        layers,
        mode="v1",
        norm_layer=lambda c: nn.GroupNorm(ngroups, c),
    )

    return model


def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        BasicBlock,
        [2, 2, 2, 2],
        norm_layer=lambda c: nn.GroupNorm(ngroups, c),
    )

    return model


def _resnet_v2(block_type, layers, in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        block_type,
        layers,
        mode="v2",
        norm_layer=lambda c: nn.GroupNorm(c // 16, c),
    )

    return model


def _resnet9_v2(block_type, in_channels, base_planes, ngroups):
    return _resnet_v2(
        block_type, [1, 1, 1, 1], in_channels, base_planes, ngroups
    )


def resnet9_v2(in_channels, base_planes, ngroups):
    return _resnet9_v2(BasicBlock, in_channels, base_planes, ngroups)


def se_resnet9_v2(in_channels, base_planes, groups):
    return _resnet9_v2(SEBasicBlock, in_channels, base_planes, ngroups)


def _resnet18_v2(block_type, in_channels, base_planes, ngroups):
    return _resnet_v2(
        block_type, [2, 2, 2, 2], in_channels, base_planes, ngroups
    )


def resnet18_v2(in_channels, base_planes, ngroups):
    return _resnet18_v2(BasicBlock, in_channels, base_planes, ngroups)


def se_resnet18_v2(in_channels, base_planes, ngroups):
    return _resnet18_v2(SEBasicBlock, in_channels, base_planes, ngroups)


def resnet50(in_channels: int, base_planes: int, ngroups: int) -> ResNet:
    model = _resnet_v1(
        in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3]
    )

    return model


def resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resnet50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3]
    )

    return model


def se_resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resneXt101(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model
