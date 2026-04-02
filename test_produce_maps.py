import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import argparse
import cv2
from Models.model_multi import Net
from Models.data_multi import test_dataset
import time
from Models.data_multi import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--test_path', type=str, default='./datasets/RGBD/', help='test dataset path')
opt = parser.parse_args()
dataset_path = opt.test_path
# load the model
model = Net()
model.cuda()
#set checkpoint path
model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load('./SPNet_epoch_best.pth', map_location='cuda:0').items()})
model.eval()
# test
test_datasets = ['STERE1000', 'SIP', 'ReDWeb-S']#['VT5000', 'VT1000', 'VT821']
condition = ['Modality-Complete', 'Modality-Only-RGB', 'Modality-Only-D']#
for con in condition:
    for dataset in test_datasets:
        save_path = './test_maps/' + con + '/' + dataset + '/' 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
# RGB-D  
        if con == 'Modality-Complete':
            image_root = dataset_path + dataset + '/test_data/test_images/'
            gt_root = dataset_path + dataset + '/test_data/test_masks/'
            depth_root = dataset_path + dataset + '/test_data/test_depth/'
        elif con == 'Modality-Only-RGB':
            image_root = dataset_path + dataset + '/test_data/test_images/'
            gt_root = dataset_path + dataset + '/test_data/test_masks/'
            depth_root = dataset_path + dataset + '/test_data/black/'
        elif con == 'Modality-Only-D':
            image_root = dataset_path + dataset + '/test_data/black/'
            gt_root = dataset_path + dataset + '/test_data/test_masks/'
            depth_root = dataset_path + dataset + '/test_data/test_depth/'
# RGB-T    
#        if dataset == 'VT5000':
#            if con == 'Modality-Complete':
#                image_root = dataset_path + dataset + '/Test/RGB/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/T/'
#            elif con == 'Modality-Only-RGB':
#                image_root = dataset_path + dataset + '/Test/RGB/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/black/'
#            elif con == 'Modality-Only-T':
#                image_root = dataset_path + dataset + '/Test/black/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/T/'
#        else: 
#            if con == 'Modality-Complete':
#                image_root = dataset_path + dataset + '/Test/RGB/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/T/'
#            elif con == 'Modality-Only-RGB':
#                image_root = dataset_path + dataset + '/Test/RGB/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/black/'
#            elif con == 'Modality-Only-T':
#                image_root = dataset_path + dataset + '/Test/black/'
#                gt_root = dataset_path + dataset + '/Test/GT/'
#                depth_root = dataset_path + dataset + '/Test/T/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    img_num = len(test_loader)
    time_s = time.time()
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda() 
        depth = depth.cuda()
        pre_res = model(image, depth, 'test')
        res = pre_res[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))
    print('Test Done!')