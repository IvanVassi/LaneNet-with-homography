import torch
import torch.nn.functional as F
import numpy as np 
import cv2
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.cluster import MeanShift,DBSCAN


class LaneClusterHnet():
    def __init__(self,image,modelh,modelseg,degree=3,method='DBSCAN'):

        
        self.image=image
        self.degree=degree
        self.method=method
        self.modelh = modelh
        self.modelseg = modelseg
        
    def _segment(self):
        threshold = 0.75
        img = cv2.resize(self.image, (512, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        #img *= (1.0/img.max())
        img = img.unsqueeze(0)
        segmentation,embeddings=self.modelseg(img)

        binary_mask = torch.argmax(F.softmax(segmentation, dim=1), dim=1, keepdim=True)
        binary_mask=segmentation.data.cpu().numpy()
        binary_mask=binary_mask.squeeze() 

        exp_mask=np.exp(binary_mask-np.max(binary_mask,axis=0))
        binary_mask=exp_mask/exp_mask.sum(axis=0)
        threshold_mask=binary_mask[1,:,:]>threshold
        threshold_mask=threshold_mask.astype(np.uint8)
        threshold_mask=threshold_mask#*255
        
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(1, 1))
        cv2.rectangle(threshold_mask, (0, 0), (512, 100), (0, 0, 0), thickness=-1) ##SIC! in order to get decent predictions, I have to cut top of the segmented image
        threshold_mask = cv2.dilate(threshold_mask,kernel,iterations=1)
        mask=cv2.connectedComponentsWithStats(threshold_mask, connectivity=4, ltype=cv2.CV_32S)
        output_mask=np.zeros(threshold_mask.shape,dtype=np.uint8)
        for label in np.unique(mask[1]):
            if label==0:
                continue
            labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
            labelMask[mask[1] == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 500:
                output_mask = cv2.add(output_mask,labelMask)    
        output_mask=output_mask.astype(np.float)/255
        
        self.embedding=embeddings.squeeze().data.cpu().numpy()
        self.binary = output_mask
        print("Segmentation output")
        plt.imshow(self.binary)
        return output_mask, embeddings

    def _get_lane_area(self):      
        idx=np.where(self.binary.T==1)
        lane_area=[]
        lane_idx=[]
        for i,j in zip(*idx):
            lane_area.append(self.embedding[:,j,i])
            lane_idx.append((j,i))
        return np.array(lane_area),lane_idx

    def _cluster(self,prediction):
        if self.method=='Meanshift':
            clustering=MeanShift(bandwidth=1.5 ,bin_seeding=True,min_bin_freq=50,n_jobs=8).fit(prediction)
        elif self.method=='DBSCAN':
            clustering = DBSCAN(eps=0.5,min_samples=500).fit(prediction)
        return clustering.labels_
 

    def _get_instance_masks(self):
        gt_img = self.image
        gt_img = cv2.resize(gt_img, (128, 64), interpolation=cv2.INTER_LINEAR)
        gt_img = gt_img*(1.0/gt_img.max())
        gt_img = np.rollaxis(gt_img, 2, 0)
        hnet_im = np.expand_dims(gt_img, 0) 
        hnet_im = torch.FloatTensor(hnet_im)
        
        out = self.modelh(hnet_im)
        transformation_coeffcient = torch.cat([out[0], torch.tensor([1.0], dtype=torch.float32)], -1).type(torch.float32)
        mult = torch.tensor([1e-02, 1e-01, 1e-01,  1e-01, 1e-01,1e-03,1]).type(torch.float32)
        transformation_coeffcient = transformation_coeffcient*mult
        H_indices = torch.tensor([[0], [1], [2], [4], [5], [7], [8]], requires_grad=False)
        R = torch.tensor([-2.0484e-01,     -1.7122e+01,     3.7991e+02,     -1.6969e+01,     3.7068e+02,     -4.6739e-02,  0.0000e+00])
        result = torch.zeros(9, dtype=torch.float32)
        result[H_indices[:, 0]] = R + transformation_coeffcient 
        H = torch.reshape(result, shape=[3, 3])
        
        print(H)
        xx = self._segment()
        lane_area,lane_idx=self._get_lane_area()
        lane_idx=np.array(lane_idx)
  
        image=self.image
        mask=np.zeros_like(image)
        segmentation_mask=np.zeros_like(image)
        if len(lane_area.shape)!=2:
            return image
        labels=self._cluster(lane_area)

        _,unique_label=np.unique(labels,return_index=True)
        unique_label=labels[np.sort(unique_label)]
        color_map={}
        polynomials=defaultdict(list)
        for index,label in enumerate(unique_label):
            color_map[label]=index
        for index,label in enumerate(labels):
            #segmentation_mask[lane_idx[index][0],lane_idx[index][1],:]=self.color[color_map[label]]
            if len(polynomials[label])==0:
                polynomials[label].append([lane_idx[index][0],lane_idx[index][1],1])
            elif 30>lane_idx[index][1]-polynomials[label][-1][1]>5:
                polynomials[label].append([lane_idx[index][0],lane_idx[index][1], 1])
        #print(polynomials)
        x_for_ypos = []
        for label in polynomials.keys():
            a = np.array(polynomials[label])[:,1]/4,np.array(polynomials[label])[:,0]/4
            a = np.array(a, np.float32)
            #print(a.T)
            a = torch.FloatTensor(a)
            line = torch.cat((a.T, torch.ones(a.size(1),1)),1)
            #line = line[2:-2]
            
            if line.shape[0]<5:
                continue        
            
            
            line_projected = torch.mm(H, line.T)
            line_projected = torch.div(line_projected, line_projected[2,:])
            
            #print(ypos_projected)
            
            X = line_projected[0, :].view(-1, 1)  # (n * 1)
            Y = line_projected[1, :].view(-1, 1)

            if self.degree == 2:
                Y_mat = torch.cat([torch.pow(Y, 2), Y, torch.ones_like(Y, dtype=torch.float32)], dim=1)
            elif self.degree == 3:
                Y_mat = torch.cat([torch.pow(Y, 3), torch.pow(Y, 2), Y, torch.ones_like(Y, dtype=torch.float32)], dim=1)
            else:
                raise ValueError('Unknown order', order) 

            w = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(Y_mat.T, Y_mat)), Y_mat.T), X)  # (4 * 1)
            x_pred = torch.mm(Y_mat, w)
            
            line_pred = torch.cat([x_pred, Y, torch.ones_like(Y, dtype=torch.float32)], dim=1).t()
            line_back = torch.mm(H.pinverse(), line_pred)
            line_back = torch.div(line_back, line_back[2,:]).T
            
           
            
            line_back = line_back[line_back[:,0]>0]
            
            x_for_ypos.append(line_back.detach().cpu().numpy())
            #print('Back',line_back)
            #print('GT', line)
            if line_back.shape[0]<10:
                continue  
        
        lane_cnt = len(x_for_ypos)
        
        plt.figure(figsize = (10,10))
        #plt.subplot(2,1,1)
        #plt.imshow(img)# + np.array([0.485, 0.456, 0.406]))
        #plt.subplot(2,1,2)
        print("Final lane predicts after lane fitting")
        plt.imshow(self.image) #+ np.array([0.485, 0.456, 0.406]))
        for i in range(lane_cnt):
            plt.scatter(x_for_ypos[i][:,0]*10,x_for_ypos[i][:,1]*11.25, marker='x',s=50, cmap='hsv')
        plt.xlim([0, 1280])
        plt.ylim([720,0])
        
        plt.show()
            
        return x_for_ypos
    
    def __call__(self):
        return self._get_instance_masks()