#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/15 20:10
import os
import numpy as np
import cv2
import keras
import json
from utils_sm import annToMask


# classes for data loading and preprocessing
class SunDataset:
    """Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, images_info_json, classes=None, augmentation=None, preprocessing=None):
        self.images_info, self.ids = [], []
        self.CLASSES = classes
        with open(images_info_json, 'r') as f:
            infile = json.load(f)
        for img_info in infile['annotations']:
            self.images_info.append(img_info)
            self.ids.append(img_info['img_name'])

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_info[i]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = self.images_info[i]['height']
        w = self.images_info[i]['width']

        # extract certain classes from mask (e.g. cars)
        masks = []
        for cl in self.CLASSES:
            mask = np.zeros([h, w])
            for obj in self.images_info[i]['object']:
                if obj['class'] == cl:
                    mask_temp = annToMask(obj, h, w)
                    mask = np.where(
                        mask_temp == 1,
                        mask_temp,
                        mask)
            masks.append(mask)

        # masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class SunDataloader(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
