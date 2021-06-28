from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchcv.utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def _focal_loss(self, x, y):
       
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y-1, self.num_classes)
        p = x.sigmoid()
        pt = torch.where(t>0, p, 1-p)    # pt = p if t > 0 else 1-p
        w = (1-pt).pow(gamma)
        w = torch.where(t>0, alpha*w, (1-alpha)*w)
        loss = F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
      
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.sum().item()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        #===============================================================
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        #===============================================================
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self._focal_loss(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss
