#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : YunYang1994
#   Created date: 2018-12-18 12:34:23
#   Description :
#
#================================================================

import sys
import argparse
import numpy as np
import tensorflow as tf
import glob, re
import cv2

#prepare data.txt: image_file_path boundingbox(top_left, bottom_right) class
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

outputfolder='/raid/wanfang/git-project/finetune_alexnet_with_tensorflow/trash_data'
tfrecord_path_prefix='./'
carboard_images = sorted(glob.glob(outputfolder + "/cardboard/*.jpg"), key=numericalSort)
glass_images = sorted(glob.glob(outputfolder + "/glass/*.jpg"), key=numericalSort)
metal_images = sorted(glob.glob(outputfolder + "/metal/*.jpg"), key=numericalSort)
paper_images = sorted(glob.glob(outputfolder + "/paper/*.jpg"), key=numericalSort)
plasticd_images = sorted(glob.glob(outputfolder + "/plastic/*.jpg"), key=numericalSort)
trash_images = sorted(glob.glob(outputfolder + "/trash/*.jpg"), key=numericalSort)

all_images = []
all_images.append(carboard_images)
all_images.append(glass_images)
all_images.append(metal_images)
all_images.append(paper_images)
all_images.append(plasticd_images)
all_images.append(trash_images)

train_file = tfrecord_path_prefix+"train.tfrecords"
val_file = tfrecord_path_prefix+"val.tfrecords"

images_num=0
with tf.python_io.TFRecordWriter(train_file) as record_writer:
    for l in range(6):
        for i in range(int(len(all_images[l])*0.8)):
            image = tf.gfile.FastGFile(all_images[l][i], 'rb').read()
            label = l
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'label' :tf.train.Feature(float_list = tf.train.FloatList(value = [label])),
                }
            ))
            record_writer.write(example.SerializeToString())
            images_num += 1
    print(">> Saving %d images in %s" %(images_num, train_file))

images_num=0
with tf.python_io.TFRecordWriter(val_file) as record_writer_1:
    for l in range(6):
        for i in range(int(len(all_images[l])*0.8), len(all_images[l])):
            image = tf.gfile.FastGFile(all_images[l][i], 'rb').read()
            label = l
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'label' :tf.train.Feature(float_list = tf.train.FloatList(value = [label])),
                }
            ))
            record_writer_1.write(example.SerializeToString())
            images_num += 1
    print(">> Saving %d images in %s" %(images_num, val_file))
