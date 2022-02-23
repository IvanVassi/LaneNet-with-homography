import torch
import torch.nn as nn
import torch.nn.functional as F

class HNetLoss(object):
    def hnet_loss_single(self, pts_gt, trans_coef):
        pts_gt = pts_gt.cuda()
        
        trans_coef = trans_coef.cuda()
        pts_gt = pts_gt.view(-1, 3)
        trans_coef = trans_coef.view(6)
        #print(trans_coef)
        mult = torch.tensor([1e-02, 1e-01, 1e-01,  1e-01, 1e-01,1e-03]).type(torch.float32).cuda()
        trans_coef = trans_coef * mult
        trans_coef = torch.cat([trans_coef, torch.tensor([1.0]).type(torch.FloatTensor).cuda()])
      
        H_indices = torch.tensor([[0], [1], [2], [4], [5], [7], [8]], requires_grad=False).cuda()

        R = torch.tensor([-2.0484e-01,     -1.7122e+01,     3.7991e+02,     -1.6969e+01,     3.7068e+02,     -4.6739e-01, 0.0000e+00]).type(torch.FloatTensor).cuda()
                         ##    [1e-02,     1e-00,            1e+01,           1e+00,            1e+01,            1e-02]
        H = torch.zeros(9, dtype=torch.float32).cuda()
        H[H_indices[:, 0]] = R + trans_coef 
       
        H = torch.reshape(H, shape=[3, 3])       
        
        pts_gt = pts_gt[pts_gt[:,2]==1]

        #pts_gt = pts_gt[5:]

        pts_gt = pts_gt.view(-1, 3).to(torch.float32).t()  # (3 * n)
        
        
        pts_projected = torch.mm(H, pts_gt) 
        if pts_gt.shape[1]<3:
            return 1.0
        
        '''
        pts_projected = pts_projected.T
        condition = pts_projected[:,2] < 0
        row_cond = condition#.any(1)
        pts_projected = pts_projected[row_cond, :]
        pts_projected = pts_projected.T
        
        pts_gt = pts_gt.T   
        
        row_cond = row_cond.unsqueeze(1)
        pts_gt=pts_gt.masked_select(row_cond).view(-1, 3).t()
        '''
        pts_projected = torch.div(pts_projected, pts_projected[2,:])
        
        # least squares closed-form solution
        X = pts_projected[0, :].view(-1, 1)  # (n * 1)
        Y = pts_projected[1, :].view(-1, 1)  # (n * 1)
        #print("X+++++++++++++",X,"Y+++++++++++++++++",Y)
        
        
        #Y_mat = torch.cat([torch.pow(Y, 3), torch.pow(Y, 2), Y, torch.ones_like(Y, dtype=torch.float32)], dim=1)  # (n * 4)
        Y_mat = torch.cat([torch.pow(Y, 2), Y, torch.ones_like(Y, dtype=torch.float32)], dim=1)  # (n * 4)
        
        #w = Y_mat.t().mm(Y_mat).inverse().mm(Y_mat.t()).mm(X)  # (4 * 1)
        w = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(Y_mat.T, Y_mat)), Y_mat.T), X)
        # re-projection and compute loss

        x_pred = torch.mm(Y_mat, w)  # (n * 1)
        pts_pred = torch.cat([x_pred, Y, torch.ones_like(Y, dtype=torch.float32)], dim=1).t()  # (3 * n)

        pts_back = torch.mm(H.pinverse(), pts_pred)
        pts_back = torch.div(pts_back, pts_back[2,:])

        #print(pts_projected, pts_back)

        loss = torch.mean(torch.pow(pts_gt[0, :] - pts_back[0, :], 2))
        #print("pts_gt",pts_gt[1, :].T, "pts_back",pts_back[1, :].T)
        #print(pts_gt[0, :] - pts_back[0, :])
        return loss
    

    def hnet_loss(self, pts_batch, coef_batch):
        
        pts_batch = pts_batch.cuda()
        coef_batch = coef_batch.cuda()

        batch_size = coef_batch.size()[0]

        loss_acc = torch.zeros(batch_size, dtype=torch.float64)

        for i in range(batch_size):
            
            loss_acc[i] = self.hnet_loss_single(pts_batch[i], coef_batch[i])
            

        #loss = loss_acc[~torch.isnan(loss_acc)].mean()
        return loss
    
    
    def hnet_loss_two_lines(self, pts_batch, coef_batch):
        
        pts_batch = pts_batch.cuda()
        coef_batch = coef_batch.cuda()

        batch_size = coef_batch.size()[0]

        loss_acc = torch.zeros(batch_size, dtype=torch.float64)
        
        for i in range(batch_size):
            pts_size = pts_batch.size()[1]
            loss_line = torch.zeros(pts_size, dtype=torch.float64)
            for k, line in enumerate(pts_batch[i]):
                line = line.T
                line = line[line[:,0]>0]
                line = torch.cat((line, torch.ones(line.size(0),1).cuda()),1)
                loss_one_line = self.hnet_loss_single(line, coef_batch[i])
                loss_line[k]=loss_one_line
            #loss_acc[i] = loss_line[~torch.isnan(loss_line)].mean()
            loss_acc[i] = loss_line.mean()
        #loss = loss_acc[~torch.isnan(loss_acc)].mean()
        loss = loss_acc.mean()
        #print(loss)
        return loss