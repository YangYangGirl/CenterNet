from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class plnresCtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(plnresCtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
      RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
      NormRegL1Loss() if opt.norm_wh else \
        RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    lk_loss, lk_off_loss = 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['lk'] = _sigmoid(output['hm'])
        output['reg_lk'] = _sigmoid(output['reg_lk'])
      # if opt.eval_oracle_wh:
      #   output['wh'] = torch.from_numpy(gen_oracle_map(
      #     batch['wh'].detach().cpu().numpy(),
      #     batch['ind'].detach().cpu().numpy(),
      #     output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      # if opt.eval_oracle_offset:
      #   output['reg'] = torch.from_numpy(gen_oracle_map(
      #     batch['reg'].detach().cpu().numpy(),
      #     batch['ind'].detach().cpu().numpy(),
      #     output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      lk_loss += self.crit(output['lk'], batch['lk']) / opt.num_stacks

      lk_off_loss += self.crit_reg(output['reg_lk'], batch['reg_mask'],
                                  batch['ind'], batch['reg_lk']) / opt.num_stacks

    loss = opt.hm_weight * lk_loss + opt.off_weight * lk_off_loss
    loss_stats = {'loss': loss, 'lk_loss': lk_loss, 'lk_off_loss': lk_off_loss}
    return loss, loss_stats

    # loss = hm_loss
    # loss_stats = {'loss': loss, 'hm_loss': hm_loss}
    # return loss, loss_stats


class plnresCtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(plnresCtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    loss_states = ['loss', 'lk_loss', 'lk_off_loss']
    # loss_states = ['loss', 'hm_loss']
    loss = plnresCtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
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
