
import os
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms


from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image

labels = {
        
        "CALC":0,
        "CIRC":1,
        "SPIC":2,
        "MISC":3,
        "ARCH":4,
        "ASYM":5,
        "NORM":6
    }

class MIAS(Dataset):

    def __init__(self, label_file, image_dir, transform=None, prior=False):
        self.img_labels = pd.read_csv(label_file, header=0, sep=' ')
        self.img_dir = image_dir
        self.transform = transform
        self.prior=prior

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        if self.prior: 
            # Pull the priors data if requested at classs instantiation
            img_path = os.path.join(self.img_dir, 'priors' ,f'{self.img_labels.iloc[idx,0]}_prior.png')
        else:
            img_path = os.path.join(self.img_dir,f'{self.img_labels.iloc[idx,0]}.png')
        image = Image.open(img_path)
        image = image.resize(512,512)
        #image = torch.flip(image, dims=[2]) # Otherwise the image is completely flipped


        #bg = self.img_labels.iloc[idx,1]
        classification = labels[self.img_labels.iloc[idx, 2]]
        #severity = self.img_labels.iloc[idx, 3]
        #x = self.img_labels.iloc[idx, 4]
        #y = self.img_labels.iloc[idx, 5]
        #radius = self.img_labels.iloc[idx, 6]

        transform = transforms.ToTensor()
        image = transform(image)
        
        return image, classification
