#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/15 20:15
import random
import skimage
from skimage import transform
import time
import segmentation_models as sm
from SUNdataset import *
from utils_sm import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# mobilenetv2、efficientnetb3、resnet50/101/152、resnext50/101、densenet121/169、xception、inceptionv3
BACKBONE = "resnext50"
BATCH_SIZE = 8
CLASSES = ['grass']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
# model = sm.PSPNet(BACKBONE,
#                   classes=n_classes,
#                   activation=activation)

print('**************************************************************\n', model.summary())
method = 'eval'     # detection   eval
if method == 'train':
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images & validation images
    train_dir = 'F:\\projects\\self-studio\\log\\train_label.json'
    train_dataset = SunDataset(
        train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dir = 'F:\\projects\\self-studio\\log\\test_label.json'
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
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        keras.callbacks.ModelCheckpoint('./model/best_model.h5',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='min'),
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
    # valid_dir = 'F:\\projects\\self-studio\\log\\test_label.json'
    valid_dir = 'F:\\projects\\self-studio\\log\\train_label.json'
    if True:
        test_dataset = SunDataset(
            valid_dir,
            classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )
    else:
        test_dataset = SunDataset(
            valid_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )

    test_dataloader = SunDataloader(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights('./model/best_model-10-22-ResNeXt50.h5')
    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    n = 100
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    for i in ids:
        image, gt_mask = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        begin_time = time.time() * 1000
        pr_mask = model.predict(image).round()
        end_time = time.time() * 1000
        run_time = int(round(end_time - begin_time))
        print('begin_time = {0}/ms end_time = {1}/ms run_time = {2}/ms'.format(begin_time, end_time, run_time))
        visualize(
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask[..., 0].squeeze(),
            pr_mask=pr_mask[..., 0].squeeze(),
        )
elif method == 'detection':
    # load best weights
    model.load_weights('./model/best_model-10-22-ResNeXt50.h5')

    IMAGE_DIR = './image/0'
    # IMAGE_DIR = 'G:\\Dataset\\SUN\\SUN2012pascalformat\\SUN2012pascalformat\\JPEGImages'
    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    cnt = 0
    while True:
        img_name = random.choice(file_names)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, img_name))
        if image.shape[-1] != 3:
            image = skimage.color.rgba2rgb(image)

        if max(image.shape[:2]) <= 512:
            shape = (512, 512)
        elif 512 < max(image.shape[:2]) <= 1024:
            shape = (1024, 1024)
        elif 1024 < max(image.shape[:2]) <= 2048:
            shape = (1024, 2048)
        shape = (384, 384)
        image = transform.resize(image, shape)
        image = np.expand_dims(image, axis=0)
        # Step3: Run detection
        begin_time = time.time() * 1000
        print('---------------: {0}'.format(img_name))
        pr_mask = model.predict(image).round()
        end_time = time.time() * 1000
        run_time = int(round(end_time - begin_time))
        print('begin_time = {0}/ms end_time = {1}/ms run_time = {2}/ms: filename is {3}'.format(begin_time, end_time, run_time, img_name))
        # sun_akpxdxlfwqblllrw  sun_agiissezaoydgqyj
        visualize(
            image=denormalize(image.squeeze()),
            pr_mask=pr_mask[..., 0].squeeze(),
        )
        cnt += 1
        if cnt >= 15000:
            break
