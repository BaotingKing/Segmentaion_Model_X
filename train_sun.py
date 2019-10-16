#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/15 20:15
import time
import segmentation_models as sm
from SUNdataset import *
from utils_sm import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# mobilenetv2、efficientnetb3、resnet50/101/152、resnext50/101、densenet121/169、xception、inceptionv3
BACKBONE = "efficientnetb3"
BATCH_SIZE = 8
CLASSES = ['grass']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.PSPNet(BACKBONE,
                  classes=n_classes,
                  activation=activation)

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

method = 'train'
if method == 'train':
    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images & validation images
    train_dir = 'F:\\projects\\self-studio\\log\\test_label.json'
    train_dataset = SunDataset(
        train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dir = 'F:\\projects\\self-studio\\log\\train_label.json'
    valid_dataset = SunDataset(
        valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = SunDataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = SunDataloader(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
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
elif method == 'eval':
    x_test_dir = ''
    y_test_dir = ''
    test_dataset = SunDataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    test_dataloader = SunDataloader(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights('best_model.h5')
    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    n = 5
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
