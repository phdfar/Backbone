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


# Define the full model
class SegmentationModel(nn.Module):
    def __init__(self, backbone, seg_head):
        super().__init__()
        
        self.conv_zero = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.backbone = backbone
        self.seg_head = seg_head

    def forward(self, x):
        features = self.backbone(x)
        
        for i, f in enumerate(features):
            if i == 0:
                x = self.conv_zero(features[f])
            else:
                x_p = self.conv_zero(features[f])
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

def run(args,dataloader,dataloader_val):
        
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
        nn.Linear(960, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128)
    )

    backbone = loadbackbone(args,device)
    model = SegmentationModel(backbone, seg_head).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs=args.epoch;
    
    best_val_loss=10000;
    # Train the model
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader);i=0;
        for batch_idx, data in enumerate(pbar):
          
          data = data.to(device)
          

          # Positive and negative samples
          indices = torch.randperm(data.shape[0])
          anchor, positive = torch.split(data, split_size_or_sections=data.shape[0]//2, dim=0)
          negative_indices = indices[data.shape[0]//2:]
          negative = data[negative_indices]

          # Forward pass
          anchor_embed = model(anchor)
          pos_embed = model(positive)
          neg_embed = model(negative)

          # Contrastive loss
          pos_similarity = torch.cosine_similarity(anchor_embed, pos_embed, dim=1)
          neg_similarity = torch.cosine_similarity(anchor_embed, neg_embed, dim=1)
       
          # Compute contrastive loss
          similarity = torch.cat([pos_similarity, neg_similarity])
          labels = torch.zeros_like(similarity).float()
          labels[:pos_similarity.shape[0]] = 1 # Set labels for positive samples to 1
          loss = criterion(similarity, labels)

          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            #if (i+1)%args.saveiter==0:

              #validation ##################################
        print('validation...>')
        """
        # Evaluate the model on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            #pbar_val = tqdm(dataloader_val)
            for images_val, masks_val in dataloader_val:
                outputs_val = model(images_val.to(device))
                masks_val = masks_val.long().to(device)
                tmp = criterion(outputs_val, masks_val).item()
                val_loss += tmp
               

        val_loss /= len(dataloader_val)
        print('iter: '+str(i)+ f" ======>>> Epoch {epoch+1}/{num_epochs}, Mean VAL-Loss: {val_loss:.4f}")
        # Save the best model based on validation loss
        #print('')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_dir)

        model.train()
        
        """
        #print('##################################################')
      #i=i+1;
