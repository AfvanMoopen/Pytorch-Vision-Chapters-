from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class ListDataset(data.Dataset):

    def __init__(self, root, list_file, transform=None):
       
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
         
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):

        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels

    def __len__(self):
        return self.num_imgs
