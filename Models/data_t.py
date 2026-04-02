import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

def cv_random_flip(label, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return label, depth
def randomCrop(label, depth):
    border = 30
    image_width = depth.size[0]
    image_height = depth.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return label.crop(random_region), depth.crop(random_region)
def randomRotation(label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return label, depth
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

class SalObjDataset(data.Dataset):
    def __init__(self, gt_root, depth_root, trainsize):
        self.trainsize = trainsize
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')
                    or f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.depths)
        print("#####")
        print(len(self.depths))
        print(len(self.gts))
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
    def __getitem__(self, index):
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        gt, depth = cv_random_flip(gt, depth)
        gt, depth = randomCrop(gt, depth)
        gt, depth = randomRotation(gt, depth)
        gt = randomPeper(gt)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        return gt, depth
    def filter_files(self):
        assert len(self.depths) == len(self.gts) and len(self.gts) == len(self.depths)
        gts = []
        depths = []
        for gt_path, depth_path in zip(self.gts, self.depths):
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            gts.append(gt_path)
            depths.append(depth_path)
        self.gts = gts
        self.depths = depths
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

# test dataset and loader
class test_dataset:
    def __init__(self, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.depths)
        self.index = 0
    def load_data(self):
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.depths[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.depths[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return gt, depth, name, np.array(image_for_post)
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

