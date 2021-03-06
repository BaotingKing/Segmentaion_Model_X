#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/15 16:39
import matplotlib.pyplot as plt
import random
import skimage.io
from skimage import transform
import time
import segmentation_models as sm
from Demo_dataset import *
from utils_sm import *
import main_draft
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BACKBONE = "efficientnetb3"    # resnet50/101/152、resnext50/101、densenet121/169、xception、inceptionv3、mobilenetv2

BATCH_SIZE = 8
# CLASSES = ['car', 'sky', 'pavement']
CLASSES = ['car']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
# model = sm.PSPNet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

method = 'detection'      # detection   eval
if method == 'train':
    # Dataset for train images & validation images
    train_dataset = Dataset(
        main_draft.x_train_dir,
        main_draft.y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataset = Dataset(
        main_draft.x_valid_dir,
        main_draft.y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./model/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
elif method == 'eval':
    test_dataset = Dataset(
        main_draft.x_test_dir,
        main_draft.y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights('./model/best_model-10-15.h5')
    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    n = 100
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    for i in ids:
        image, gt_mask = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()

        visualize(
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask[..., 0].squeeze(),
            pr_mask=pr_mask[..., 0].squeeze(),
        )
elif method == 'detection':
    # load best weights
    model.load_weights('./model/best_model-10-15.h5')

    IMAGE_DIR = 'F:\\projects\\GitHub\Segmentation\\SegNet-Tutorial-master\\CamVid\\test'
    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    while True:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
        image = transform.resize(image, (1024, 1024))
        image = np.expand_dims(image, axis=0)

        # Step3: Run detection
        begin_time = time.time() * 1000
        # results = model.detect([image], verbose=1)
        pr_mask = model.predict(image).round()
        end_time = time.time() * 1000
        run_time = int(round(end_time - begin_time))
        print('begin_time = {0}/ms end_time = {1}/ms run_time = {2}/ms'.format(begin_time, end_time, run_time))
        visualize(
            image=denormalize(image.squeeze()),
            pr_mask=pr_mask[..., 0].squeeze(),
        )
