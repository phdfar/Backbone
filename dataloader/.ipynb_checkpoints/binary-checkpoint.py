import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import random
import pickle
import glob

# Define your custom dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       
        image= cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        mask= cv2.imread(self.mask_paths[idx],0).astype('float')

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image.float(), mask.squeeze(0).float()

def run(args):
    
    
    sample_file=[]
    ann=glob.glob(args.otherpath+'*/*.png')
    for a in ann:
        sp = a.split('/');
        img=args.rgbpath+'train/JPEGImages/'+sp[-2]+'/'+sp[-1]
        sample_file.append({'img':img,'mask':a})
    

    # Define the transformation to apply to the image and mask
    transform = transforms.Compose([
        transforms.Resize((384, 640)),
        transforms.ToTensor()
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_paths=[];mask_paths=[]
    random.Random(1337).shuffle(sample_file)
    
    ln = len(sample_file)
    b=int(ln*0.85)
    c=int(ln*0.01)
    d=int(ln*0.14)
    
    print('Number train samples '+str(b))
    print('Number valid samples '+str(c))
    print('Number test samples '+str(d))
    print('###################################')
    
    # Define the paths to your images and masks
    for i in range(0,b):
      image_paths.append(sample_file[i]['img'])
      mask_paths.append(sample_file[i]['mask'])

    
    # Create the dataset and dataloader
    dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    
    image_paths=[];mask_paths=[]
    for i in range(b+1,b+1+c):
      image_paths.append(sample_file[i]['img'])
      mask_paths.append(sample_file[i]['mask'])

    # Create the dataset and dataloader
    datasetv= SegmentationDataset(image_paths, mask_paths, transform=transform)
    dataloader_val = DataLoader(datasetv, batch_size=args.batchsize, shuffle=True)
    
    return dataloader,dataloader_val


  asetv, batch_size=args.batchsize, shuffle=True)
    
    return dataloader,dataloader_val


  