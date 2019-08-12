from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.image import draw_ct_heatmap
import math


class plnresCTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
   
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        output_h = input_h // 4#self.opt.down_ratio
        output_w = input_w // 4#self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        gt_ct = np.zeros((output_h, output_w), dtype=np.float32) # batch["ct"]

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_ct_heatmap(gt_ct, ct_int, output_h / 112) #self.opt.grid_size)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        gt_det = np.array(gt_det)
        #ret = {'input': inp, 'ct': gt_ct}
        
        channels = 3 + output_h + output_w + num_classes
        shape = (output_h, output_w, channels * 2)
        multilabels = [np.zeros(shape).astype(np.float32) for _ in range(4)]
        ss = [(-2, -2), (+2, -2), (-2, +2), (+2, +2)]
 
        for k in range(4):
            for i in range(num_objs):
                ann = anns[i]
                bbox = self._coco_box_to_bbox(ann['bbox'])
                bbox[0] /= width
                bbox[2] /= width
                bbox[1] /= height
                bbox[3] /= height
                
                if (bbox[2] < 0.01 or bbox[3] < 0.01):
                    continue 
                cat = int(self.cat_ids[ann['category_id']])
                xm, ym = bbox[:2]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                xc = bbox[0] + bbox[2] / ss[k][0]
                yc = bbox[1] + bbox[3] / ss[k][1]
            
                colm = int(xm * output_w)
                rowm = int(ym * output_h)
                colc = int(xc * output_w)
                rowc = int(yc * output_h)
                
                if colm == output_w: colm -= 1
                if rowm == output_h: rowm -= 1
                if colc == output_w: colc -= 1
                if rowc == output_h: rowc -= 1

                if ((multilabels[k][rowm, colm, 0] != 0) or (multilabels[k][rowc, colc, 1] != 0)):
                    continue                
                 
                xm = xm * output_w - colm
                ym = ym * output_h - rowm
                xc = xc * output_w - colc
                yc = yc * output_h - rowc
 
                multilabels[k][rowm, colm, 0] = 1.0
                multilabels[k][rowm, colm, 2] = xm
                multilabels[k][rowm, colm, 3] = ym
                multilabels[k][rowm, colm, 6+rowc] = 1.0
                multilabels[k][rowm, colm, 6+output_h+colc] = 1.0
                multilabels[k][rowm, colm, 6+2*output_h+2*output_w+cat] = 1.0
                multilabels[k][rowc, colc, 1] = 1.0
                multilabels[k][rowc, colc, 4] = xc
                multilabels[k][rowc, colc, 5] = yc
                multilabels[k][rowc, colc, 6+output_h+output_w+rowm] = 1.0
                multilabels[k][rowc, colc, 6+2*output_h+output_w+colm] = 1.0
                multilabels[k][rowc, colc, 6+2*output_h+2*output_w+num_classes+cat] = 1.0               
        
        ret = {'input': inp, 'gt': np.array(multilabels), 'ct': gt_ct}
        return ret

