#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *

class listDataset(Dataset):
    def __init__(self, root, shape=None, transform=None, target_transform=None, train=False, test_txt=False,
                 imgdirpath='', labdirpath=''):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test_txt = test_txt
        self.shape = shape
        self.imgdirpath = imgdirpath
        self.labdirpath = labdirpath

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgp = self.lines[index].rstrip()
        imgpath = imgp + '.jpg'
        imgpath = os.path.join(self.imgdirpath, imgpath)

        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, self.labdirpath, imgp)
            label = torch.from_numpy(label).float()
        elif self.test_txt:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
            if self.transform is not None:
                img = self.transform(img)
            return img, imgp
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)

            labpath = imgp + '.txt'
            labpath = os.path.join(self.labdirpath, labpath)
            label = torch.zeros(50 * 5)
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 6.0 / img.width).astype('float32'))
            except Exception:
                print(' No target !!!!!!!!!')
                tmp = torch.zeros(1, 5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 50 * 5:
                label = tmp[0:50 * 5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)