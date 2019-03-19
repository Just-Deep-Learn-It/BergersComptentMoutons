import os
import os.path
import numpy as np
import torch 
import torch.utils.data as data 
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


class ShanghaiTechLoader(torch.utils.data.Dataset):

    def __init__(self, path_to_data, mode='TRAIN', transform=None, target_transform=None):
        
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode  # training, validation or test set
        

        if self.mode == 'TRAIN':
            data_dir = os.path.join(path_to_data, 'train_data/')
            count_path=os.path.join(data_dir,'count_train.npy')
        elif self.mode == 'TEST':
            data_dir = os.path.join(path_to_data, 'test_data/')
            count_path=os.path.join(data_dir,'count_test.npy')
        elif self.mode == 'VAL':
            data_dir = os.path.join(path_to_data, 'validation_data/')
            count_path=os.path.join(data_dir,'count_validation.npy')
        
        images_dir=os.listdir(os.path.join(data_dir,'images_cropped/'))
        N=len(images_dir)
        self.counts=[]
        self.data=[]
        self.targets=[]
        
        count=np.load(count_path)
        for image_name in images_dir:
            image_path=os.path.join(data_dir,'images_cropped/',image_name)
            x = image_name.split('.jpg',1)[0].split('_')
            i,num=int(x[-1]), int(x[-2]) # dertmining original image index and numvber of the crop , works only for train data
            
            density_name='GT_density_'+image_name.split('.jpg',1)[0]+'.npy'
            density_path=os.path.join(data_dir,'GT_density_map_cropped/',density_name)
            #print(os.path.join(data_dir,'images/'))
            image=plt.imread(image_path)
            image=np.reshape(image,(3,224,224))
            self.data.append(image)

            density_map = np.load(density_path)
            self.counts.append(count[i+9*(num-1)])
            self.targets.append(density_map)
            
                
        self.data, self.targets= np.asarray(self.data, dtype=float) , np.asarray(self.targets, dtype=float)
        self.counts=np.asarray(self.counts, dtype=float)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the density map.
        """
        img= self.data[index] 
        target=[self.counts[index],self.targets[index]]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def labels(self):
        return self.targets
    
    def data(self):
        return self.data