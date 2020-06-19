# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *
"""
miniImageNet_path               = './filelists/miniImagenet/data'
miniImageNet_val_path           = './filelists/miniImagenet/val'
miniImageNet_test_path           = './filelists/miniImagenet/test'
"""

identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        d = ImageFolder(miniImageNet_path)

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class SetDataset:
    def __init__(self, batch_size, transform, mode='train'):

        self.sub_meta = {}
        print("[SetDataset] mode:%s"%(mode))

        if mode == 'train':
            self.cl_list = range(64)#64-classes
            d = ImageFolder(miniImageNet_path)
        elif mode == 'val':
            self.cl_list = range(16)#16-classes
            d = ImageFolder(miniImageNet_val_path)
        else:
            self.cl_list = range(20)#20-classes
            d = ImageFolder(miniImageNet_test_path)

        for cl in self.cl_list:
            self.sub_meta[cl] = []


        #=====================
        
        flag = False
        """
        for i in os.listdir("."):
            if mode+"_loader.npy" == i and mode == "train":
                # [FOR DEBUG PURPOSE]
                flag = True
                self.sub_meta = np.load(i,allow_pickle='TRUE').item()
                print("load dataset from %s"%(i))
        """
        if flag == False:
            for i, (data, label) in enumerate(d):#this line needs to be accelerated!
                self.sub_meta[label].append(data)#label2data dict
            """
            if mode == "train":
                for key, item in self.sub_meta.items(): self.sub_meta[key] = self.sub_meta[key][:150]
                np.save(mode+'_loader.npy', self.sub_meta)
                print("save to %s"%(mode+'_loader.npy'))
            """
        #=====================
        """
        for i, (data, label) in enumerate(d):#this line needs to be accelerated!
            self.sub_meta[label].append(data)#label2data dict
        """


        for key, item in self.sub_meta.items():#0~64
            print (len(self.sub_meta[key]))#number of items in the corresponding key
            #print "600" 200 times
    
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)    
        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
            #64,105,3,224,224

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes#20
        self.n_way = n_way#5
        self.n_episodes = n_episodes#600

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]#fetch the top-5-class loader from dataloader pools (class-1 ~ class-20 loaders)

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, mode='train', n_way=5, n_support=5, n_query=16, n_eposide = 300):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size#224
        self.mode = mode#"test"
        self.n_way = n_way#5
        self.batch_size = n_support + n_query#5+15
        self.n_eposide = n_eposide#600

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        #import pdb
        #print("A")
        transform = self.trans_loader.get_composed_transform(aug)
        #print("B")
        dataset = SetDataset(self.batch_size, transform, self.mode)#batch_size=20
        #print("C")
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )#len(dataset)=20,self.n_way=5,self.n_eposide=600
        #print("D")
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True)       
        #print("E")
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        #print("F")
        return data_loader

if __name__ == '__main__':
    pass
