import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class GrayscaleToRGB(object):

    def __call__(self, img):
      if len(img.shape) != 3:
        img.unsqueeze_(0)
        img = img.repeat(3, 1, 1)
      elif img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
          
      return img

class LandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, K=4):

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.K = K
        transform_list = [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomChoice([
                    transforms.RandomResizedCrop(128),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)
                ]),
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(transform_list)
        
        self.landmark_to_class_id = {}
        self.weights_per_line = []
        for i in range(self.data_frame.shape[0]):
          num_images = len(self.data_frame.iloc[i,1].split())
          self.weights_per_line.append(num_images)
          self.landmark_to_class_id[int(self.data_frame.iloc[i,0])] = i

        self.total_images = sum(self.weights_per_line)

    def __len__(self):
      return self.data_frame.shape[0]

    def __getitem__(self, idx):
      	
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_list = np.random.choice(self.data_frame.iloc[idx,1].split(),self.K, replace=False)
        img_name = [os.path.join(self.root_dir,img.strip('.jpg') + '.jpg') for img in img_list]
        images = [self.transform(Image.open(img)) for img in img_name]
        images = torch.stack(images)

        landmark_id = self.landmark_to_class_id[int(self.data_frame.iloc[idx,0])]

        sample = {'images': images,  'landmark_id': landmark_id}
        return sample