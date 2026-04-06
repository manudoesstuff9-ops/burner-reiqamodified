from __future__ import print_function

import math
import numpy as np
import torch
from torchvision import datasets
from torch.utils import data
import pandas as pd
from PIL import Image
from .iqa_distortions import *
import random
from torchvision import transforms



class ImageFolderInstance(datasets.ImageFolder):
    
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        
        path, target = self.imgs[index]
        image = self.loader(path)

        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


class IQAImageClass(data.Dataset):

    def __init__(self, csv_path, n_aug=7, n_scale=1, n_distortions=1, patch_size=224, swap_crops=1):

        super().__init__()
        df = pd.read_csv(csv_path)       
        self.image_name = df['Image_path']
        self.n_aug = n_aug
        self.n_scale = n_scale
        self.n_distortions = n_distortions
        self.patch_size = patch_size
        self.swap = (self.n_aug+1)//2
        self.swap_crops = swap_crops
        self.min_OLA = 0.10
        self.max_OLA = 0.30

    def __len__(self):
        return len(self.image_name)

    def iqa_transformations(self, choice, im, level):
        
        if choice == 1:
            im = imblurgauss(im, level)
        elif choice == 2:
            im = imblurlens(im, level)
        elif choice == 3:
            im = imcolordiffuse(im, level)
        elif choice == 4:
            im = imcolorshift(im, level)
        elif choice == 5:
            im = imcolorsaturate(im, level)
        elif choice == 6:
            im = imsaturate(im, level)
        elif choice == 7:
            im = imcompressjpeg(im, level)
        elif choice == 8:
            im = imnoisegauss(im, level)
        elif choice == 9:
            im = imnoisecolormap(im, level)
        elif choice == 10:
            im = imnoiseimpulse(im, level)
        elif choice == 11:
            im = imnoisemultiplicative(im, level)
        elif choice == 12:
            im = imdenoise(im, level)
        elif choice == 13:
            im = imbrighten(im, level)
        elif choice == 14:
            im = imdarken(im, level)
        elif choice == 15:
            im = immeanshift(im, level)
        elif choice == 16:
            im = imresizedist(im, level)
        elif choice == 17:
            im = imsharpenHi(im, level)
        elif choice == 18:
            im = imcontrastc(im, level)
        elif choice == 19:
            im = imcolorblock(im, level)
        elif choice == 20:
            im = impixelate(im, level)
        elif choice == 21:
            im = imnoneccentricity(im, level)
        elif choice == 22:
            im = imjitter(im, level)
        elif choice == 23:
            im = imresizedist_bilinear(im, level)
        elif choice == 24:
            im = imresizedist_nearest(im, level)
        elif choice == 25:
            im = imresizedist_lanczos(im, level)
        elif choice == 26:
            im = imblurmotion(im, level)

        return im

    def choose_y(self):
        low_val = self.patch_size * (1 - self.max_OLA)
        high_val = self.patch_size * (1 - self.min_OLA)

        low_ = int(math.floor(low_val))
        high_ = int(math.ceil(high_val))
        high_ = max(high_, low_ + 1)

        return np.random.randint(low=low_, high=high_)
    
    def choose_x(self, y):
        denom = max(self.patch_size - y, 1)
        low_val = ((1 - self.max_OLA) * (self.patch_size ** 2) - self.patch_size * y) / denom
        high_val = ((1 - self.min_OLA) * (self.patch_size ** 2) - self.patch_size * y) / denom

        low_ = int(math.floor(max(0, low_val)))
        high_ = int(math.ceil(max(0, high_val)))
        high_ = max(high_, low_ + 1)

        return np.random.randint(low=low_, high=high_)

    def crop_transform(self, image, crop_size=224):

        if image.shape[2] < crop_size or image.shape[3] < crop_size:
            image = transforms.transforms.CenterCrop(crop_size)(image)
        else:
            image = transforms.transforms.RandomCrop(crop_size)(image)

        return image

    def __getitem__(self, idx):

        image = Image.open(self.image_name[idx]).convert('RGB')
        
        img_pair1 = transforms.ToTensor()(image)
        chunk1 = img_pair1.unsqueeze(0)

        choices = list(range(1, 27))
        random.shuffle(choices)
        
        for i in range(0, self.n_aug):
            if self.n_distortions == 1:
                level = random.randint(0, 4)
                img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image, level))
                img_aug_i = img_aug_i.unsqueeze(0)
                chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
            else:
                j = random.randint(0, 25)
                if random.random() > 0.1:
                    level = random.randint(0, 4)
                    img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image, level))
                else:
                    level1 = random.randint(0, 4)
                    level2 = random.randint(0, level1)
                    level1 = level1 - level2
                    img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[j], self.iqa_transformations(choices[i], image, level1), level2))
                img_aug_i = img_aug_i.unsqueeze(0)
                chunk1 = torch.cat([chunk1, img_aug_i], dim=0)

        start_y = self.choose_y()
        start_x = self.choose_x(start_y)
        chunk1_1 = self.crop_transform(chunk1, crop_size=self.patch_size+max(start_y, start_x))

        chunk1_2 = chunk1_1[:, :, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
        chunk1_1 = chunk1_1[:, :, :self.patch_size, :self.patch_size]

        if self.swap_crops == 1:
            temp_chunk1 = chunk1_1[0:self.swap].clone()
            chunk1_1[0:self.swap] = chunk1_2[0:self.swap]
            chunk1_2[0:self.swap] = temp_chunk1
            
        t1 = torch.cat((chunk1_1, chunk1_2), dim=1)

        if self.n_scale == 2:
            chunk3 = torch.nn.functional.interpolate(chunk1, size=(chunk1.shape[2]//2, chunk1.shape[3]//2), mode='bicubic', align_corners=True)

            start_y = self.choose_y()
            start_x = self.choose_x(start_y)
            chunk2_1 = self.crop_transform(chunk3, crop_size=self.patch_size+max(start_y, start_x))

            chunk2_2 = chunk2_1[:, :, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
            chunk2_1 = chunk2_1[:, :, :self.patch_size, :self.patch_size]

            if self.swap_crops == 1:
                temp_chunk2 = chunk2_1[self.swap:].clone()
                chunk2_1[self.swap:] = chunk2_2[self.swap:]
                chunk2_2[self.swap:] = temp_chunk2
                
            t2 = torch.cat((chunk2_1, chunk2_2), dim=1)
            return torch.cat((t1, t2), dim=0)
        else:
            return t1
