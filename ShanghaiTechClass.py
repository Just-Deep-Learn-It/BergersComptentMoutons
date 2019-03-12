#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import numpy as np
import torch 
import torch.utils.data as data 
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


# In[22]:


target=np.array([[1,2,3], [4,5,6]])
target[:,1]


# In[130]:


class ShanghaiTech(data.Dataset):
    

    def __init__(self, root='~/Desktop/EA', train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.base_folder='ShanghaiTech/part_A/'
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        

        if self.train:
            data_dir = os.path.join(self.root, self.base_folder, 'train_data/')
            count_path=os.path.join(data_dir,'count_train_part_A.npy')
        else:
            data_dir = os.path.join(self.root, self.base_folder, 'test_data/')
            count_path=os.path.join(data_dir,'count_test_part_A.npy')
        
        images_dir=os.listdir(os.path.join(data_dir,'images/'))
        N=len(images_dir)
        self.counts=[]
        self.data=[]
        self.targets=[]
        
        count=np.load(count_path)
        i=0
        for image_name in images_dir:
            image_path=os.path.join(data_dir,'images/',image_name)
            
            density_name='GT_density_'+image_name.split('.jpg',1)[0]+'.npy'
            density_path=os.path.join(data_dir,'GT_density_map/',density_name)
            #print(os.path.join(data_dir,'images/'))
            self.data.append(plt.imread(image_path)[:224,:224])
            self.counts.append(count[i])
            self.targets.append(np.load(density_path)[:224,:224])
            
            i+=1
            if i>2:
                break
                
        self.data, self.targets,self.counts= np.asarray(self.data) , np.asarray(self.targets),np.asarray(self.counts)
        
        #self.data, self.targets = torch.from_numpy(self.data) , torch.from_numpy(self.targets)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[:,index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

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


# In[131]:


dataset=ShanghaiTech('~/Desktop/EA') # Change the path to the root folder


# In[132]:


dataset.data.shape


# In[133]:


dataset.targets.shape


# In[134]:


dataset.counts.shape


# In[ ]:




