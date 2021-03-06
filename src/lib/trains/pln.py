from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import _slow_neg_loss
from models.losses import L2Loss, FocalLoss
from models.decode import pln_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class plnCtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(plnCtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss(reduction='sum')
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        ct_exist_loss = 0
        ct_pt_loss = 0
        ct_nopt_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['ct'] = output['ct'].to(self.opt.device)
            #ct_exist_loss += self.crit(output['ct'][:, :, :, 0], batch['ct']) / self.opt.num_stacks
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


class plnCtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(plnCtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['ct_exist_loss', 'ct_pt_loss', 'ct_nopt_loss']
        loss = plnCtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = pln_decode(
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
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
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


