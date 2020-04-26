from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.pln import plnCTDetDataset
from .sample.plnres import plnresCTDetDataset
from .sample.landmark import LandmarkDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.face_pas import FaceVOC
from .dataset.coco_lm_hp import COCOHP_LM
dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'face': FaceVOC,
  'landmark': COCOHP_LM,
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'landmark': LandmarkDataset,
  'pln': plnCTDetDataset,
  'plnres': plnresCTDetDataset, 
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
