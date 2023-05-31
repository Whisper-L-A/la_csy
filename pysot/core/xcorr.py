# Copyright (c) SenseTime. All Rights Reserved.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

"""
三个函数是用于计算跨通道互相关的函数
"""
"""
通过循环计算互相关，速度较慢的版本。
对于输入的特征图x和卷积核kernel，首先对每个样本进行变形，
然后使用F.conv2d函数进行卷积操作，最后将结果拼接在一起返回。
"""
def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out

"""
这是一个较快的版本，使用分组卷积（group conv2d）计算互相关。它首先获取kernel的批次大小，
然后将x和kernel调整为适当的形状。接着，使用F.conv2d计算互相关，其中groups参数设置为批次大小。
最后，将结果调整为适当的形状并返回。
"""
def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


"""
这个函数计算深度互相关（depthwise cross-correlation）。它首先获取kernel的批次大小和通道数（channel）。
然后，将x和kernel调整为适当的形状。
接着，使用F.conv2d计算互相关，其中groups参数设置为批次大小乘以通道数。
最后，将结果调整为适当的形状并返回。
"""
def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
