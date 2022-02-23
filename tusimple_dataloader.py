import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import cv2
import ujson as json

from albumentations import (
    PadIfNeeded,
    Rotate,
    Flip,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    Cutout,
    RandomBrightnessContrast,    
    RandomGamma,
    ShiftScaleRotate,
    CoarseDropout,
    Resize,
    GaussNoise,
    Blur,
    GridDropout,
    Downscale,
    RandomFog
)

#VGG_MEAN=[103.939,116.779,123.68]
VGG_MEAN=[0,0,0]

class tusimple_dataset(Dataset):
    def __init__(self,dataset_dir,phase,size=(512, 256),transform=None):
        self.dataset_dir=dataset_dir
        self.phase=phase
        self.size=size
        self.transform=transform

        assert os.path.exists(dataset_dir),'Directory {} does not exist!'.format(dataset_dir)

        if phase=='train' or phase=='val':
            label_files=list()
            if phase=='train':
                label_files.append(os.path.join(dataset_dir,'label_data_0313.json'))
                label_files.append(os.path.join(dataset_dir,'label_data_0531.json'))
            elif phase=='val':
                label_files.append(os.path.join(dataset_dir,'label_data_0601.json'))
            self.image_list=[]
            self.lanes_list=[]
            for file in label_files:
                for line in open(file).readlines():
                    info_dict=json.loads(line)
                    self.image_list.append(info_dict['raw_file'])
                    h_samples=info_dict['h_samples']
                    lanes=info_dict['lanes']
                    xy_list=list()
                    for lane in lanes:
                        y=np.array([h_samples]).T
                        x=np.array([lane]).T
                        xy=np.hstack((x,y))
                        index=np.where(xy[:,0]>2)
                        xy_list.append(xy[index])
                    self.lanes_list.append(xy_list)

        elif phase=='test':
            task_file=os.path.join(dataset_dir,'test_tasks_0627.json')
            self.image_list=[json.loads(line)['raw_file'] for line in open(task_file).readlines() ]
        elif phase=='test_extend':
            task_file = os.path.join(dataset_dir, 'test_tasks_0627.json')
            self.image_list=[]
            for line in open(task_file).readlines():
                path=json.load(line)['raw_file']
                dir=os.path.join(dataset_dir,path[:-7])
                for i in range(1,21):
                    self.image_list.append(os.path.join(dir,'%d.jpg'%i))
                    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if self.phase=='train' or self.phase=='val':
                        
            img_path=os.path.join(self.dataset_dir,self.image_list[idx])
            image=cv2.imread(img_path,cv2.IMREAD_COLOR)
            h,w,c=image.shape
            image=cv2.resize(image,self.size,interpolation=cv2.INTER_LINEAR)
            
            if self.phase=='train':
                if self.transform:
                    img = self.transform(img)
                else:
                    aug = Compose([
                              OneOf([VerticalFlip(p=0.5),
                                     HorizontalFlip(p=0.5),
                                     ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.05, 0.05), scale_limit=(-0.3, 0.3), rotate_limit=(-30, 30), interpolation=2, border_mode=0, value=(0, 0, 0),                                            mask_value=None),
                                     RandomBrightnessContrast(always_apply=False, p=0.3, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
                                     RandomGamma(always_apply=False,p=0.3, gamma_limit=(70,130))
                                    ],p=0.4),
                     OneOf([RandomFog(always_apply=False, p=0.6, fog_coef_lower=0.1, fog_coef_upper=0.18, alpha_coef=0.2),
                            OpticalDistortion(always_apply=False, p=0.3, distort_limit=(-0.1, 0.1), shift_limit=(-0.1, 0.1), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                            ElasticTransform(always_apply=False, p=0.3, alpha=1.0, sigma=50.0, alpha_affine=20.1299991607666, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None,                                                 approximate=False),
                            #GaussNoise(var_limit=(0, 0.001), p=0.3),
                            #Blur(always_apply=False, p=0.1, blur_limit=(1, 3)),
                            #GridDropout(ratio=0.01, p=0.3)     
                              ],p=0.3)
                    ])
            else:
                aug = Compose([])#ToTensor(num_classes=2)]) 
                
            #image=image.astype(np.float32)
            #image-=VGG_MEAN
            
            #image=torch.from_numpy(image).float()/255
            
            bin_seg_label=np.zeros((h,w),dtype=np.uint8)
            inst_seg_label=np.zeros((h,w),dtype=np.uint8)
            lanes=self.lanes_list[idx]

            for idx ,lane in enumerate(lanes):
                cv2.polylines(bin_seg_label,[lane],False,1,10)
                cv2.polylines(inst_seg_label,[lane],False,idx+1,10)
            
            bin_seg_label=cv2.resize(bin_seg_label,self.size,interpolation=cv2.INTER_NEAREST)
            inst_seg_label=cv2.resize(inst_seg_label,self.size,interpolation=cv2.INTER_NEAREST)
            
            bin_seg_label = np.broadcast_to(bin_seg_label[...,None],bin_seg_label.shape+(3,))
            inst_seg_label = np.broadcast_to(inst_seg_label[...,None],inst_seg_label.shape+(3,))         
                         
            augmented = aug(image=image, masks=[bin_seg_label, inst_seg_label]) 
            
            image = augmented['image']       
            bin_seg_label = augmented['masks'][0]
            inst_seg_label = augmented['masks'][1]

            image=torch.from_numpy(image).float()#/255
            bin_seg_label = bin_seg_label[:,:,0]
            inst_seg_label = inst_seg_label[:,:,0]
            
            bin_seg_label=torch.from_numpy(bin_seg_label.copy())#.long()
            inst_seg_label=torch.from_numpy(inst_seg_label.copy())#.long()
            
            image=np.transpose(image,(2,0,1))
            #image *= (1.0/image.max())
            #bin_seg_label=np.transpose(image,(2,0,1))
            #inst_seg_label=np.transpose(image,(2,0,1))
            
            #print(bin_seg_label.shape)
            #print(inst_seg_label.shape)
            #sample={'input_tensor':image,'binary_tensor':bin_seg_label,'instance_tensor':inst_seg_label,'raw_file':self.image_list}
            #return sample
            return image, bin_seg_label, inst_seg_label, img_path[15:]
        
        elif self.phase=='test' or self.phase=='test_extend':
            img_path=os.path.join(self.dataset_dir,self.image_list[idx])
            image=cv2.imread(img_path,cv2.IMREAD_COLOR)
            image=cv2.resize(image,self.size,interpolation=cv2.INTER_NEAREST)
            image=image.astype(np.float32)
            image-=VGG_MEAN
            image=np.transpose(image,(2,0,1))
            image *= (1.0/image.max())
            image=torch.from_numpy(image).float()/255
            clip,seq,frame=self.image_list[idx].split('/')[-3:]
            path='/'.join([clip,seq,frame])
            sample={'input_tensor':image,'raw_file':self.image_list[idx],'path':path}
            #return sample
            return input_tensor