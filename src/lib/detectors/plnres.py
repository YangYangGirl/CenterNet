from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from datetime import datetime

from external.nms import soft_nms
from models.decode import plnres_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class PlnresDetector(BaseDetector):
    def __init__(self, opt):
        super(PlnresDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            time_str = time.strftime('%m-%d-%H-%M-%S-%f')
            visimage = images[0].cpu().numpy()
            visimage = visimage.transpose(1, 2, 0)
            #visimage = canny(visimage)
            cv2.imwrite("./vis_ct/" + '{}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) + ".png", visimage)
            output = self.model(images)
            pred_ct = np.zeros((3, 112, 112), np.uint8)
            #pred_ct.fill(255)

            pred_ct[0] = output[0][:, :, 0] * 255
            pred_ct[1] = output[0][:, :, 0] * 255
            pred_ct[2] = output[0][:, :, 0] * 255

            visct = pred_ct.transpose(1, 2, 0)
            visct = 255 - visct
            visct = cv2.resize(visct, (448, 448), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./vis_ct/" + 'pred_{}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) + ".png", visct)
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = plnres_decode(output)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def process_ct(self, images, return_time=False):
        with torch.no_grad():
            time_str = time.strftime('%m-%d-%H-%M-%S-%f')
            visimage = images[0].cpu().numpy()
            visimage = visimage.transpose(1, 2, 0)
            #visimage = canny(visimage)
            output = self.model(images)
            pred_ct = np.zeros((3, 112, 112), np.uint8)
            #pred_ct.fill(255)

            pred_ct[0] = output[0][:, :, 0] * 255
            pred_ct[1] = output[0][:, :, 0] * 255
            pred_ct[2] = output[0][:, :, 0] * 255

            visct = pred_ct.transpose(1, 2, 0)
            visct = 255 - visct
            visct = cv2.resize(visct, (448, 448), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./vis_ct/" + 'pred_{}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) + ".png", visct)
            torch.cuda.synchronize()
            forward_time = time.time()

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)

