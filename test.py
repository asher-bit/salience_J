import numpy as np
import torch
import torch.nn as nn
import cv2

import os
import argparse
import glob
from dataset.HDR_data import getTrainVal_loader,getTest_loader
from models.model import Salicon
from rich.progress import track

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def test(model, test_loader, save_path):
    model.eval()

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _, (inputs, target, names) in track(enumerate(test_loader), total=len(test_loader)):
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)

            # coarse_img=coarse_img.unsqueeze(0).to(device)

            pred = model(*inputs)

            pred_image = pred.squeeze()
            h, w = target.shape[-2], target.shape[-1]

            smap = (pred_image - torch.min(pred_image)) / ((torch.max(pred_image) - torch.min(pred_image)))
            smap = smap.cpu().numpy()
            smap = smap * 255
            smap = np.uint8(smap)
            # print('smap:',smap)
            smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)
            smap = cv2.GaussianBlur(smap, (75, 75), 25, cv2.BORDER_DEFAULT)
            path = os.path.join(save_path, str(names[0])+'.jpg')
            cv2.imwrite(path, smap)


def main():
    parser = argparse.ArgumentParser()
    np.random.seed(12)
    resnet_path = './resnet50_caffe.pth'
    # dataset type
    parser.add_argument('--test_dataset', type=str, default='osie')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    # gpu
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--resnet', type=str, default=resnet_path)
    # model dir #0-459
    #parser.add_argument('--model_dir', type=str,
    #                    default='/raid/wangxu/Program/Salicon_pytorch-master/save_path/GNN_IMLHDR_sm_k5_salmodel6/salicon_126.pth')

    args = parser.parse_args()

    # Get dataloader (test)
    test_dataset_dir = './IMLHDR'
    test_img_dir = 'image_mantiuk'
    test_label_dir = 'density'
    test_mutilexp_dir = 'multi_exposure'
    save_path = './results'
    model_path = './checkpoints/salicon_150.pth'

    save_dir_img = save_path +'/'
    if not os.path.isdir(save_dir_img):
        os.makedirs(save_dir_img)

    dataloaders = getTest_loader(test_dataset_dir, test_img_dir, test_mutilexp_dir, test_label_dir)

    # init the model
    #model_weight = os.path.join(os.path.abspath('..'), args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = Salicon()
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    print("Begin test, Device: {}".format(device))

    # test the model
    test(model, dataloaders, save_dir_img)


if __name__ == '__main__':
    main()