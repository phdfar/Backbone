from collections import OrderedDict
from functools import partial
from backbone.backbone import build_resnet_fpn_backbone
from config import cfg
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


backbone_type = cfg.MODEL.BACKBONE.TYPE
print('backbone_type',backbone_type)
backbone = build_resnet_fpn_backbone(cfg)

info_to_print = [
    "Backbone type: {}".format(cfg.MODEL.BACKBONE.TYPE),
    "Backbone frozen: {}".format("Yes" if cfg.TRAINING.FREEZE_BACKBONE else "No")
]

pretrained_wts_file = 'mask_rcnn_R_101_FPN_backbone.pth'
#print_fn("Restoring backbone weights from '{}'".format(pretrained_wts_file))

if os.path.exists(pretrained_wts_file):
    restore_dict = torch.load(pretrained_wts_file,map_location=torch.device('cpu'))
    backbone.load_state_dict(restore_dict, strict=True)
    print(backbone)
            

