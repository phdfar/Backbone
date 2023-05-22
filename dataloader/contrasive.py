import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import random
import pickle
import glob
import os
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       
        image= cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
       
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        t=([image[0],image[1]],-1)
        
        return t

class ValidDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       
        image= cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
       
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        t=([image[0],image[1]],1)
        
        return t
      
def run(args):
    
    
    NUM_WORKERS = os.cpu_count()


    sample_file=[]
    ann=glob.glob(args.rgbpath+'train/JPEGImages/*/*.jpg')
    for a in ann:
        if os.path.exists(a):
          sample_file.append({'img':a})

    class ContrastiveTransformations:
        def __init__(self, base_transforms, n_views=2):
            self.base_transforms = base_transforms
            self.n_views = n_views
    
        def __call__(self, x):
            return [self.base_transforms(x) for i in range(self.n_views)]
    
    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((720, 1280)),
            transforms.RandomResizedCrop(size=(384,640)),
            transforms.Resize((384, 640)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


    image_paths=[];
    random.Random(1337).shuffle(sample_file)
    
    ln = len(sample_file)
    b=int(ln*0.9)
    c=int(ln*0.05)
    d=int(ln*0.05)
    
    print('Number train samples '+str(b))
    print('Number valid samples '+str(c))
    print('Number test samples '+str(d))
    print('###################################')
    
    # Define the paths to your images and masks
    for i in range(0,b):
      image_paths.append(sample_file[i]['img'])
    
    # Create the dataset and dataloader
    dataset = TrainDataset(image_paths, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    #dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS, )
    
    image_paths=[];
    for i in range(b+1,b+1+c):
      image_paths.append(sample_file[i]['img'])

    # Create the dataset and dataloader
    datasetv = ValidDataset(image_paths, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    #dataloader_val = DataLoader(datasetv, batch_size=args.batchsize, shuffle=True)
    dataloader_val = DataLoader(datasetv, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS, )

    
    return dataloader,dataloader_val



  