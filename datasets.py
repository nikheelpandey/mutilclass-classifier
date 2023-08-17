import os
import cv2
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from csv_maker import createCSVFromFolder, getTrainTestSplit
from sklearn.model_selection import train_test_split
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# Define a function to get the data augmentation pipeline
def get_augmentation_pipeline(split):
    if split == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),  # Random crop and resize
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
            transforms.RandomRotation(10),  # Random rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(256),  # Resize for evaluation
            transforms.CenterCrop(224),  # Center crop
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

# Define the transform dictionary
transform = {
    'train': get_augmentation_pipeline('train'),
    'test': get_augmentation_pipeline('test')
}


# transform = {
#     'train': transforms.Compose(
#         [
#             # transforms.Resize([256, 256]),

#             # transforms.RandomCrop(224),
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#             ]),
            
#     'test': transforms.Compose([
#                         transforms.ToPILImage(),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#                             ])

#     }

normalize =  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

class CLFDataset(Dataset):

    def __init__(self, df, dim=(224,224), mode = 'test', image_ls = None, labels = None):

        with open('categories.json','r') as f:
            self.categories = json.load(f)

        self.img_dim = dim

        if mode != 'eval':
            self.images = df['file_path'].tolist()
            self.labels = df['category'].tolist()

        if mode == 'train':
            self.transform = transform['train']
        elif mode == 'test':
            self.transform = transform['test']

        elif mode == 'eval':
            assert image_ls != None
            
            self.transform = transform['test']
            self.images = image_ls
            self.labels = labels

        self.normalize = normalize

    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_path, class_name = self.images[idx], self.labels[idx]
        img = cv2.resize(cv2.imread(img_path), self.img_dim)

        class_id = self.categories[class_name]
        
        img_tensor = self.transform(img)
        # img_tensor = self.normalize(img_tensor)
        # print(type(img_tensor))
        # img_tensor = img_tensor.permute(2, 1) #channel first
        class_id = torch.tensor([class_id])
        
        return img_tensor, class_id


    def __getimg__(self, idx):

        img_path = self.images[idx]
        img = cv2.resize(cv2.imread(img_path), self.img_dim)

        img_tensor = self.transform(img)
        img_tensor = self.normalize(img_tensor/255)

        
        return img_tensor, img_path


def getDataLoader(dataset, batch_size=8, num_worker = 4):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_worker)
    




if __name__=='__main__':

    path = '../dogs-vs-cats/'
    df = createCSVFromFolder(path)
    train_df, test_df = getTrainTestSplit(df)
    # print(train_df['file_path'].tolist())

    ds = CLFDataset(train_df)

    dl = getDataLoader(ds)

    sample = next(iter(dl))
    inputs, labels = sample
    print(inputs)


