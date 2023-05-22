import torch
import torch.nn as nn
from tqdm import tqdm

from collections import OrderedDict
from functools import partial
from backbonecr.fpn import *
from backbonecr.config import cfg
import torch
import os
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request
from copy import deepcopy
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import STL10
from tqdm.notebook import tqdm


  
def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


    
def loadbackbone(args,device):
    
    backbone = build_fcos_resnet_fpn_backbone(cfg)
    pretrained_wts_file = args.pretrain
    if os.path.exists(pretrained_wts_file):
        print('Backbone Loaded From ',pretrained_wts_file)
        restore_dict = torch.load(pretrained_wts_file,map_location=device)
        restore_dict = {k.replace('backbone.', ''): v for k, v in restore_dict.items()}
        backbone.load_state_dict(restore_dict, strict=False)
    
    else:
      print('NOT exist')
        
    return backbone

class SimCLR(L.LightningModule):
    def __init__(self, backbone , seg_head , hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        
        
        self.conv_zero = [nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding='same').to(torch.device("cuda:0")) for _ in range(0,5)]
        self.backbone = backbone
        self.seg_head = seg_head.to(torch.device("cuda:0"))
        
        """
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        """

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
      
      
    def convnet(self,x):
      features = self.backbone(x)
      for i, f in enumerate(features):
          if i == 0:
              x = self.conv_zero[i](features[f])
          else:
              x_p = self.conv_zero[i](features[f])
              target_h, target_w = x.size()[2:]
              h, w = x_p.size()[2:]
              assert target_h % h == 0
              assert target_w % w == 0
              factor_h, factor_w = target_h // h, target_w // w
              assert factor_h == factor_w
              x_p = aligned_bilinear(x_p, factor_h)
              x = x + x_p
      
      seg_map = self.seg_head(x)
      
      return seg_map

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)


        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        #print('feats',feats.size())
        #print('cos_sim',cos_sim.size())


        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")
        



def run(args,train_loader,val_loader):
        
    if args.gpu==True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    seg_head = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding='same'),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding='same'),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,padding='same'),
        nn.MaxPool2d(2),
        nn.Flatten(start_dim=1),
        nn.ReLU(inplace=True),
        nn.Linear(960, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128)
    )

    backbone = loadbackbone(args,device)
    
    max_epochs=args.epoch
    
    CHECKPOINT_PATH = args.basepath
    
    #os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    L.seed_everything(42)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = SimCLR(backbone=backbone,seg_head=seg_head, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=args.epoch)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        trainer.fit(model, train_loader, val_loader)

        # Load best checkpoint after training
        #model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    