# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from design2.pysot.core.config import cfg
from design2.pysot.utils.bbox import IoU, corner2center
from design2.pysot.utils.anchor import Anchors


"""
用于生成锚点目标。
这个类主要用于目标跟踪算法中，通过计算正负样本和回归目标来生成锚点目标。
"""
class AnchorTarget:
    def __init__(self,):
        # 创建了一个Anchors对象，用于生成锚点。然后调用generate_all_anchors方法生成所有的锚点。
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)

    # target（目标框），size（输出大小）和neg（是否为负样本，默认为False）
    def __call__(self, target, size, neg=False):
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        # 从给定的位置中选择一定数量的样本
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        # 计算目标框的中心坐标和宽高
        tcx, tcy, tw, th = corner2center(target)

        """
         如果neg为True，表示处理负样本。首先将目标框附近的锚点类别设置为0（表示负样本），
         然后从这些负样本中随机选择一定数量的样本。
         最后，计算锚点与目标框的重叠度（overlap），
         并返回cls, delta, delta_weight, overlap。
        """
        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        """
        如果neg为False，表示处理正样本。
        首先计算锚点的坐标和中心坐标，然后计算回归目标（delta）。
        接着，计算锚点与目标框的重叠度（overlap），
        并根据重叠度阈值（cfg.TRAIN.THR_HIGH和cfg.TRAIN.THR_LOW）将锚点分为正样本和负样本。
        最后，从正负样本中分别选择一定数量的样本，并更新cls和delta_weight
        """
        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
            anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
            anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > cfg.TRAIN.THR_HIGH)
        neg = np.where(overlap < cfg.TRAIN.THR_LOW)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap
