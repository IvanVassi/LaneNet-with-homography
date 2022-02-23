from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import random
import pandas as pd
import warnings
import os,json
import numpy as np
import cv2
from scipy.ndimage.morphology import grey_dilation
from scipy.interpolate import CubicSpline
from skimage import filters

warnings.filterwarnings("ignore")

class TUSimpleHNet(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.LINE_SIZE = 30
        self.transform = transform
        sub    = [i for i in os.listdir(self.path) if i!=".DS_Store"]
        labels = [self.path + "/" + i for i in sub if i[-4:]=="json"]
        images_root_path = self.path + "/clips"
        images = list()
        self.labels = dict()
        images_folders = [self.path+"/clips/"+i for i in os.listdir(images_root_path) if i!=".DS_Store"]
        for imgs_folder in images_folders:
            for i in os.listdir(imgs_folder):
                if("DS" in i):
                    continue

                tmp_path = imgs_folder + "/" +i
                lst_of_imgs = [imgs_folder + "/" + i+"/"+j for j in os.listdir(tmp_path) if j=="20.jpg"]
                images += lst_of_imgs

        self.images = images
        for label_path in labels:
            with open(label_path,"r") as f:
                for i in f.readlines():
                    todict = json.loads(i[:-1])
                    label_img_name = todict['raw_file']
                    self.labels[label_img_name] = todict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        key_ind = image_path.split("/").index("clips")
        key_path = os.path.join( *image_path.split("/")[key_ind:])
        abs_path = self.path +"/"+os.path.join( *image_path.split("/")[key_ind:])

        label = self.labels[key_path]
        lanes_w = np.array(label['lanes'])
        lanes_h = np.array(label['h_samples'])
        lane_cnt = lanes_w.shape[0]

        image = plt.imread(image_path) #(720, 1280, 3)
        #image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        image = cv2.resize(image, dsize=(128,64), interpolation=cv2.INTER_AREA)
        image = np.asarray(image).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image *= (1.0/image.max())
        lane_pair = list()
        point = 0
        
        xs = (lanes_w[0,:]-8) / 10
        if xs.shape[0] == 48:
            for k in range(8):
                xs = np.append(xs,[0]) 
        ys = lanes_h / 11.25
        ys = np.clip(ys, 0, 127)
        if ys.shape[0] == 48:
            for k in range(8):
                ys = np.append(ys,[0]) 
        pair = np.stack([xs,ys])
        lane_pair.append(pair)
        
        #mask = (lanes_w[1,:] * lanes_h) > 0
        xs = (lanes_w[lane_cnt-1,:]-8) / 10
        if xs.shape[0] == 48:
            for k in range(8):
                xs = np.append(xs,[0]) 
        ys = lanes_h / 11.25
        ys = np.clip(ys, 0, 127)
        if ys.shape[0] == 48:
            for k in range(8):
                ys = np.append(ys,[0]) 
        pair = np.stack([xs,ys])
        lane_pair.append(pair)
                
        return np.array(image), np.array(lane_pair).astype(np.float32)
