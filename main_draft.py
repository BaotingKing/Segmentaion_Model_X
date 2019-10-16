#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/15 17:22
import os
from Demo_dataset import Dataset
from utils_sm import visualize

DATA_DIR = 'F:\\projects\\GitHub\\Segmentation\\SegNet-Tutorial-master\\CamVid'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


def show_data(x_dir, y_dir):
    dataset = Dataset(x_dir, y_dir, classes=['car', 'pedestrian'])

    image, mask = dataset[5]  # get some sample
    visualize(
        image=image,
        cars_mask=mask[..., 0].squeeze(),
        sky_mask=mask[..., 1].squeeze(),
        background_mask=mask[..., 2].squeeze(),
    )


if __name__ == '__main__':
    show_data(x_train_dir, y_train_dir)
