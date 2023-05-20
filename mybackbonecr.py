from backbonecr.fpn import *
from backbonecr.config import cfg

import torch
import os

def load(cfg):

  backbone = build_fcos_resnet_fpn_backbone(cfg)
  pretrained_wts_file = 'CondInst_MS_R_50_1x.pth'

  if os.path.exists(pretrained_wts_file):

      print("Restoring backbone weights from '{}'".format(pretrained_wts_file))
      restore_dict = torch.load(pretrained_wts_file,map_location=torch.device('cpu'))
      restore_dict = {k.replace('backbone.', ''): v for k, v in restore_dict.items()}
      backbone.load_state_dict(restore_dict, strict=False)
  
  return backbone