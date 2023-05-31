# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from design2.pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from design2.pysot.models.backbone.mobile_v2 import mobilenetv2
from design2.pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50


"""
这个字典将不同的神经网络骨干网络（backbone）名称映射到相应的函数
"""
BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
            }


"""
name 参数用于指定要获取的骨干网络名称。
函数通过在 BACKBONES 字典中查找 name 参数对应的函数，并使用传入的关键字参数调用该函数，最后返回创建的骨干网络实例
"""
def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
