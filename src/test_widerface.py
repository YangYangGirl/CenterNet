import sys
import os
import scipy.io as sio
import cv2
import torch

path = os.path.dirname(__file__)
CENTERNET_PATH = os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
from datasets.dataset_factory import get_dataset


def test_img(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    detector = detector_factory[opt.task](opt)

    img = '../readme/000388.jpg'
    ret = detector.run(img)['results']

def test_vedio(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    detector = detector_factory[opt.task](opt)

    vedio = vedio_path if vedio_path else 0
    cap = cv2.VideoCapture(vedio)
    while cap.isOpened():
        det = cap.grab()
        if det:
            flag, frame = cap.retrieve()
            res = detector.run(frame)
            cv2.imshow('face detect', res['plot_img'])

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_wider_Face(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    Path = '/home/yy/github/CenterNet/data/face/WIDER_val/images'
    wider_face_mat = sio.loadmat('/home/yy/github/ThunderNet/data/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = opt.outdir

    detector = detector_factory[opt.task](opt)

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img_path = os.path.join(Path, zip_name)
            dets = detector.run(img_path)['results']
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    opt = opts().parse()
    test_wider_Face(opt)
