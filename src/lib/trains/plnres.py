from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import _slow_neg_loss
from models.losses import L2Loss, FocalLoss
from models.decode import plnres_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

def box_rmse(a, b):
    return ((a[0] - b[0])**2 +
              (a[1] - b[1])**2 +
              (a[2] - b[2])**2 +
              (a[3] - b[3])**2)


def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if (w < 0 or h < 0):
        return 0
    area = w * h
    return area

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left

def convbox_match(pred, label, num_box, num_class):
    batch_size = label.shape[0];
    rows = label.shape[1];
    cols = label.shape[2];
    channels = label.shape[3];

    pred_mask_output = np.zeros((batch_size, rows, cols, channels * num_box))
    label_mask_output = np.zeros((batch_size, rows, cols, channels))
    update_label_output = np.zeros((batch_size, rows, cols, channels))
    truth = np.zeros(4)
    out = np.zeros(4)
    #label = np.zeros((batch_size, rows, cols, channels * num_box))
    #row = np.zeros((num_box))
    #col = np.zeros((num_box))

    xm, ym, xc, yc = 0, 0, 0, 0
    iou, rmse = 0, 0
    cur_match, cur_match, best_match, count = 0, 0, 0, 0
    avg_dist, avg_cat, avg_allcat, avg_conn, avg_allconn, avg_obj, avg_anyobj = 0, 0, 0, 0, 0, 0, 0
    pred_conn, pred_conn_m, pred_conn_c, pred_cat_m, pred_cat_c, pred_cat = 0, 0, 0, 0, 0, 0

    for batch in range(batch_size):
        for rowm in range(rows):
            for colm in range(cols):
                for c in range(channels):
                    for i in range(num_box):
                        #每个通道设置num_box个
                        pred_mask_output[batch, rowm, colm, c*num_box+i] = 0
                        label_mask_output[batch, rowm, colm, c] = 0
                        update_label_output[batch, rowm, colm, c] = label[batch, rowm, colm, c]
          
        for rowm in range(rows):
            for colm in range(cols):
                for i in range(2*num_box):
                    avg_anyobj += pred[batch, rowm, colm, i]
            
                if label[batch, rowm, colm, 0] == 1:
                    #match by overlap
                    best_index = -1
                    rowc = -1
                    colc = -1
                    for i in range(rows):
                         if label[batch, rowm, colm, 6+i] == 1:
                             rowc = i
                             break

                    for i in range(cols):
                        if label[batch, rowm, colm, 6+rows+i] == 1:
                            colc = i
                            break

                    xm = (label[batch, rowm, colm, 2] + colm) / cols
                    ym = (label[batch, rowm, colm, 3] + rowm) / rows
                    xc = (label[batch, rowc, colc, 4] + colc) / cols
                    yc = (label[batch, rowc, colc, 5] + rowc) / rows


                    truth[0] = xm
                    truth[1] = ym
                    truth[2] = abs(xm - xc) * 2.0
                    truth[3] = abs(ym - yc) * 2.0
      
                    best_match = -100;
                    for i in range(num_box):

                        pred_cat = 0.0;
                        for j in range(num_class):
                            pred_cat_m =  pred[batch, rowm, colm, (6+2*(rows+cols))*num_box+i*num_class+j]
                            pred_cat_c =  pred[batch, rowc, colc, (6+2*(rows+cols)+num_class)*num_box+i*num_class+j]
                            label_cat_m = label[batch, rowm, colm, 6+2*(rows+cols)+j]
                            label_cat_c = label[batch, rowc, colc, 6+2*(rows+cols)+num_class+j]
                            pred_cat += (pred_cat_m - label_cat_m) ** 2
                            pred_cat += (pred_cat_c - label_cat_c) ** 2


                        pred_conn_m = pred[batch, rowm, colm, 6*num_box+i*(rows+cols)+rowc] * pred[batch, rowm, colm, 6*num_box+i*(rows+cols)+rows+colc]
                        pred_conn_c = pred[batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rowm] * pred[batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rows+colm]
                        pred_conn = (pred_conn_m + pred_conn_c) / 2.0
                
                        xm = (pred[batch, rowm, colm, ((num_box+i)*2+0) + colm]) / cols;
                        ym = (pred[batch, rowm, colm, ((num_box+i)*2+1) + rowm]) / rows;
                        xc = (pred[batch, rowc, colc, ((num_box*2+i)*2+0) + colc]) / cols;
                        yc = (pred[batch, rowc, colc, ((num_box*2+i)*2+1) + rowc]) / rows;

                        out[0] = xm
                        out[1] = ym
                        out[2] = abs(xm - xc) * 2.0
                        out[3] = abs(ym - yc) * 2.0
                        iou = box_iou(out, truth)
                        rmse = box_rmse(out, truth)
                
                        cur_match = pred_conn * (iou - rmse + 0.1) + 0.1 * (2 - pred_cat)
                
                        if cur_match > best_match:
                            best_match = cur_match;
                            best_index = i;

              

                    assert best_index != -1
                    assert rowc != -1
                    assert colc != -1
              
                    row = [rowm, rowc]
                    col = [colm, colc]
                    
                    for n in range(2):
                        pred_mask_output[batch, row[n], col[n], n*num_box+best_index] = 1
                        label_mask_output[batch, row[n], col[n], n] = 1
                        avg_obj += pred[batch, row[n], col[n], n*num_box+best_index]
                        
                        pred_mask_output[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0] = 1
                        pred_mask_output[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1] = 1
                        label_mask_output[batch, row[n], col[n], 2*(1+n)+0] = 1
                        label_mask_output[batch, row[n], col[n], 2*(1+n)+1] = 1
                        avg_dist += (pred[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0] - label[batch, row[n], col[n], 2*(1+n)+0])**2 + (pred[batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1] - label[batch, row[n], col[n], 2*(1+n)+1])**2
                        
                        for i in range(rows+cols):
                            pred_mask_output[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i] = 1
                            label_mask_output[batch, row[n], col[n], 6+n*(rows+cols)+i] = 1
                            if label[batch, row[n], col[n], 6+n*(rows+cols)+i] == 1:
                                avg_conn += pred[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i]
                            avg_allconn += pred[batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i]
        
        
                        for i in range(num_class):
                            pred_mask_output[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i] = 1
                            label_mask_output[batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i] = 1
                            if label[batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i] == 1:
                                avg_cat += pred[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i]
                            avg_allcat += pred[batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i]
        
                        
                        count += 1;

      
    if count > 0:
        print("Detection Avg Dist: ", avg_dist/count)
        print(", Pos Cat: ", avg_cat/count)
        print(", All Cat: " ,avg_allcat/(count*num_class))
        print(", Pos Conn: " ,avg_conn/(count*2))
        print(", All Conn: " ,avg_allconn/(count*(rows + cols)))
        print(", Pos Obj: " ,avg_obj/count)
        print(", Any Obj: " ,avg_anyobj/(batch_size*rows*cols*num_box*2))
        print(", count: " ,count)

    return torch.from_numpy(pred_mask_output).cuda(), torch.from_numpy(label_mask_output).cuda(), torch.from_numpy(label_mask_output).cuda()

class plnresCtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(plnresCtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss(reduction='sum')
        self.opt = opt
        self.object_scale = 1.0
        self.noobject_scale = 0.3
        self.connection_scale = 1.0
        self.coord_scale = 5.0
        self.class_scale = 1.0     
        self.num_box = 2   

    '''def forward(self, outputs, batch):
        opt = self.opt
        ct_exist_loss = 0
        ct_pt_loss = 0
        ct_nopt_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['ct'] = output['ct'].to(self.opt.device)
            #ct_exist_loss += self.crit(output['ct'][:, :, :, 0], batch['ct']) / self.opt.num_stacks
            #print(output['ct'][:, :, :, 0].shape, batch['ct'].shape)
            output_ct_pt = output['ct'][:, :, :, 0] * batch['ct']   #  -1 **2
            output_ct_nopt = output['ct'][:, :, :, 0] * (1 - batch['ct'])  # **2
            ct_pt_loss += self.crit(output_ct_pt, batch['ct']) / self.opt.num_stacks
            ct_nopt_loss += (output_ct_nopt ** 2).sum() / self.opt.num_stacks
            #ct_nopt_loss += self.crit(output_nopt, batch['ct']) / self.opt.num_stacks
        ct_exist_loss = self.opt.ct_pt_weight * ct_pt_loss + self.opt.ct_nopt_weight * ct_nopt_loss
        loss_stats = {'ct_exist_loss': ct_exist_loss, 'ct_pt_loss': ct_pt_loss,
                      'ct_nopt_loss': ct_nopt_loss}
        loss = ct_exist_loss
        return loss, loss_stats
    '''

    '''def forward(self, outputs, batch):
        opt = self.opt
        ct_exist_loss = 0
        ct_pt_loss = 0
        ct_nopt_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['ct'] = output['ct'].to(self.opt.device)
            for i in range(len(output)):
                gt = batch['ct'][i].contiguous().view(-1)
                pred = output['ct'][:, :, :, 0][i].contiguous().view(-1)
                pos_inds = gt.eq(1).float()
                neg_inds = gt.lt(1).float()
                num_pos = pos_inds.sum()
                #if num_pos == 0:
                    #continue
                num_neg = num_pos * 3
                #pred_pos = (pos_inds * pred).float()
                pred_neg = (neg_inds * pred).float()
                pred_neg = pred_neg.sort(descending=True)   
                if num_pos == 0:
                    num_pos = 1         
                ct_pt_loss += (((pred - 1) ** 2) * pos_inds).sum()  / self.opt.num_stacks / num_pos
                ct_nopt_loss += (pred_neg[0][0: num_neg.int()] ** 2).sum() / self.opt.num_stacks / num_pos
        ct_exist_loss = self.opt.ct_pt_weight * ct_pt_loss + self.opt.ct_nopt_weight * ct_nopt_loss
        
        loss_stats = {'ct_exist_loss': ct_exist_loss, 'ct_pt_loss': ct_pt_loss,
                      'ct_nopt_loss': ct_nopt_loss}
        loss = ct_exist_loss
        return loss, loss_stats  '''
        
    def forward(self, outputs, batch):
        opt = self.opt
        ct_exist_loss = 0
        ct_pt_loss = 0
        ct_nopt_loss = 0
        for s in range(opt.num_stacks):
            outputs = outputs[0]
            for i in range(len(outputs)):
                output = outputs[i].to(self.opt.device)
                gt = batch['ct'][i].contiguous().view(-1)
                pred = output[ :, :, 0].contiguous().view(-1)
                pos_inds = gt.eq(1).float()
                neg_inds = gt.lt(1).float()
                num_pos = pos_inds.sum()
                #if num_pos == 0:
                    #continue
                num_neg = num_pos * 3
                #pred_pos = (pos_inds * pred).float()
                pred_neg = (neg_inds * pred).float()
                pred_neg = pred_neg.sort(descending=True)
                if num_pos == 0:
                    num_pos = 1
                ct_pt_loss += (((pred - 1) ** 2) * pos_inds).sum()  / self.opt.num_stacks / num_pos
                ct_nopt_loss += (pred_neg[0][0: num_neg.int()] ** 2).sum() / self.opt.num_stacks / num_pos
        ct_exist_loss = self.opt.ct_pt_weight * ct_pt_loss + self.opt.ct_nopt_weight * ct_nopt_loss

        loss_stats = {'ct_exist_loss': ct_exist_loss, 'ct_pt_loss': ct_pt_loss,
                      'ct_nopt_loss': ct_nopt_loss}
        loss = ct_exist_loss
        return loss, loss_stats  
    
    '''def forward(self, outputs, batch): 
        batch['gt'] = batch['gt'].permute(1, 0, 2, 3, 4)
        #loss = [self.convbox_loss(outputs[i], batch['gt'][i]) for i in range(4)]
        ct_exist_loss =      
 
        #return loss'''
      
    def convbox_loss(self, det, label, weight=1.0, scope=None):
        opt = self.opt
        batch_size = det.shape[0]
        object_scale = opt.object_scale * weight / batch_size
        noobject_scale = opt.noobject_scale * weight / batch_size
        coord_scale = opt.coord_scale * weight / batch_size
        connection_scale = opt.connection_scale * weight / batch_size
        class_scale = opt.class_scale * weight / batch_size
        num_box = opt.num_box
        num_class = opt.num_class
        output_row = opt.output_row
        output_col = opt.output_col
        det_mask, label_mask, update_label = convbox_match(det, label, num_box, num_class)
        '''det_mask.set_shape([None, None, None, None])
        label_mask.set_shape([None, None, None, None])
        update_label.set_shape([None, None, None, None])
        '''
        noobj = det[:, :, :, :num_box] * (1 - det_mask[:, :, :, :num_box])
        l1 = (noobj ** 2).sum() * noobject_scale

        obj_1 = det[:, :, :, :num_box] * det_mask[:, :, :, :num_box]
        obj_2 = update_label[:, :, :, :1] * label_mask[:, :, :, :1]
        l2 = ((obj_1 - obj_2) ** 2).sum() * object_scale

        cls_1 = det[:, :, :, num_box:num_box*(1+num_class)] * det_mask[:, :, :, num_box:num_box*(1+num_class)]
        cls_2 = update_label[:, :, :, 1:1+num_class] * label_mask[:, :, :, 1:1+num_class]
        l3 = ((cls_1 - cls_2) ** 2).sum() * coord_scale

        coord_1 = det[:, :, :, num_box*(1+num_class):] * det_mask[:, :, :, num_box*(1+num_class):]
        coord_2 = update_label[:, :, :, 1+num_class:] * label_mask[:, :, :, 1+num_class:]
        l4 = ((coord_1 - coord_2) ** 2).sum() * connection_scale

        cls_1 = det[:, :, :, (6+2*(output_row+output_col))*num_box:] * det_mask[:, :, :, (6+2*(output_row+output_col))*num_box:]
        cls_2 = update_label[:, :, :, 6+2*(output_row+output_col):] * label_mask[:, :, :, 6+2*(output_row+output_col):]
        l5 = ((cls_1 - cls_2) ** 2).sum() * class_scale

        loss = l1 + l2 + l3 + l4 + l5

        return loss

class plnresCtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(plnresCtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['ct_exist_loss', 'ct_pt_loss', 'ct_nopt_loss']
        loss = plnresCtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = plnres_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


