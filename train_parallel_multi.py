import os
# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Models.model_multi import Net
from Models.data_multi import SalObjDataset, test_dataset
from Models.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import torch.nn as nn
import argparse
import Models.misc as utils
from Models.default import _C as cfg
import random
from torch.utils.data import DataLoader
import Models.samplers as samplers
cudnn.benchmark = True
cfg = cfg
utils.init_distributed_mode(cfg.TRAIN)
device = torch.device(cfg.TRAIN.device)
seed = cfg.TRAIN.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model = Net()
if (opt.load is not None):
    model.load_pre(opt.load_rgb, opt.load_t)
    print('load model from ', opt.load_rgb, opt.load_t)
model.to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)
print("\n不固定的参数：")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
params = model.parameters()
optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=cfg.TRAIN.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)
train_image_root = opt.rgb_label_root
train_gt_root = opt.gt_label_root
train_depth_root = opt.depth_label_root
val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('load data...')
dataset_train = SalObjDataset(train_image_root, train_gt_root, train_depth_root, trainsize=opt.trainsize)
dataset_test = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)
if cfg.TRAIN.distributed:
    sampler_train = samplers.DistributedSampler(dataset_train)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, opt.batchsize, drop_last=True)

train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               num_workers=16,
                               pin_memory=True)
total_step = len(train_loader)
if cfg.TRAIN.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAIN.gpu], find_unused_parameters=True)
logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))
CE = torch.nn.BCEWithLogitsLoss()
step = 0
best_mae = 1
best_epoch = 0
print(len(train_loader))
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
dice = SoftDiceLoss()
def l2_loss(fr1, fr2):
    return F.mse_loss(fr2, fr1)
def miss_loss(fr4_1, fr4_2, fr3_1, fr3_2, fr2_1, fr2_2, fr1_1, fr1_2, ft4_1, ft4_2, ft3_1, ft3_2, ft2_1, ft2_2, ft1_1, ft1_2):
    total_loss = 0.0
    feature_pairs = [
        (fr1_1, fr1_2), (fr2_1, fr2_2), (fr3_1, fr3_2), (fr4_1, fr4_2), 
        (ft1_1, ft1_2), (ft2_1, ft2_2), (ft3_1, ft3_2), (ft4_1, ft4_2)
    ]
    for feature1, feature2 in feature_pairs:
        total_loss += torch.nn.functional.mse_loss(feature1, feature2)
    return total_loss 

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    if cfg.TRAIN.distributed:
        sampler_train.set_epoch(epoch)
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.to(device)
            depths = depths.to(device)
            gts = gts.to(device)
            out, fr4_1, fr4_2, fr3_1, fr3_2, fr2_1, fr2_2, fr1_1, fr1_2, ft4_1, ft4_2, ft3_1, ft3_2, ft2_1, ft2_2, ft1_1, ft1_2 = model(images, depths, 'train')
            loss1_fusion = F.binary_cross_entropy_with_logits(out, gts)
            dice_loss = dice(out, gts)
            loss_miss = miss_loss(fr4_1, fr4_2, fr3_1, fr3_2, fr2_1, fr2_2, fr1_1, fr1_2, ft4_1, ft4_2, ft3_1, ft3_2, ft2_1, ft2_2, ft1_1, ft1_2)
            loss_seg = loss1_fusion + dice_loss + loss_miss
            loss = loss_seg
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}] || Step [{:04d}/{:04d}] || loss: {:.4f} || bce: {:0.4f} || dice: {:0.4f} || miss: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1_fusion.data,
                           dice_loss.data, loss_miss))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f} || loss1_fusion: {:0.4f} || dice: {:0.4f} || miss: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss1_fusion.data, dice_loss.data, loss_miss))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.module.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()

            out= model(image, depth, 'test')
            res = out[0]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
        #Test during training to save the best pth file
        if epoch >= 60 or epoch ==1:
           val(dataset_test, model, epoch, save_path)
