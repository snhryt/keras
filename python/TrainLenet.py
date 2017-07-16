#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import random
import pickle
import cv2
from argparse import ArgumentParser
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model
from CommonFunc import mergeFilepaths, countLineNum, showProcessingTime, loadSingleImage
from augmentation import augmentImage
import numpy as np


def buildLenet(class_num, height=28, width=28, channel_num=1):
  model = Sequential()
  # model.add(Convolution2D(filters=20, kernel_size=(5, 5), strides=(1, 1), 
  #                         input_shape=(img_height, img_width, channel_num), name='conv1'))
  model.add(Convolution2D(filters=20, kernel_size=(8, 8), strides=(3, 3), 
                          input_shape=(height, width, channel_num), name='conv1'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Convolution2D(filters=50, kernel_size=(5, 5), strides=(1, 1), name='conv2'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(500))
  model.add(Activation('relu'))
  model.add(Dense(class_num))
  model.add(Activation('softmax'))
  model.summary()
  return model


def storeImageFilepathsAndLabels(txt_filepath, class_num):
  print('\n[Loading "{}"]'.format(txt_filepath))
  total = countLineNum(txt_filepath)
  print('.. total: {} files'.format(total))
  
  filepaths, labels = [], []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      filepath = line.split(' ')[0]
      filepaths.append(filepath)
      label = line.split(' ')[1]
      label = label.split('\n')[0]
      labels.append(label)
      line = f.readline()
  # Convert labels to one-hot vector (numpy array)
  labels = np_utils.to_categorical(labels, class_num)
  return filepaths, labels


def generateArrays(filepaths, labels, batch_size, channel_num=1):
  batch_imgs = [None] * batch_size
  batch_labels = [None] * batch_size
  indices = [i for i in range(len(filepaths))]
  while True:
    random_indices = random.sample(indices, batch_size)
    for i, index in enumerate(random_indices):
      img = loadSingleImage(filepaths[index], channel_num=channel_num) / 255.
      batch_imgs[i] = img
      batch_labels[i] = labels[index]
    batch_imgs = np.array(batch_imgs)
    batch_labels = np.array(batch_labels)
    yield (batch_imgs, batch_labels)


def loadImagesThenAugmentation(filepaths, labels, channel_num=1):
  output_imgs, output_labels = [], []
  counter = 0
  for filepath, label in zip(filepaths, labels):
    img = loadSingleImage(filepath, channel_num=channel_num) / 255.
    augmented_imgs = augmentImage(img, augmentation_num=100)
    for j, augmented_img in enumerate(augmented_imgs):
      output_imgs.append(augmented_img)
      output_labels.append(label)
    counter += 1
    if counter % 100 == 0:
      print('.. {0}/{1} images are loaded and augmented'.format(counter, len(filepaths)))
  output_imgs = np.array(output_imgs)
  output_labels = np.array(output_labels)
  return output_imgs, output_labels


def loadAlreadyAugmentedImages(filepaths, channel_num=1):
  output_imgs = [None] * len(filepaths)
  for i, filepath in enumerate(filepaths):
    img = loadSingleImage(filepath, channel_num=channel_num) / 255.
    output_imgs[i] = img
    if (i + 1) % 100 == 0:
      print('.. {0}/{1} images are loaded'.format(i + 1, len(filepaths)))
  return np.array(output_imgs) 


def main():
  usage = ('Usage: python {} TARGET_DIRPATH CLASS_NUM '.format(__file__)
           + '[--height] [--width] [--channel_num] [--batch_size] [--epochs] [--generator] '
           + '[--augmentation] [--help]')
  parser = ArgumentParser(usage)
  parser.add_argument('target_dirpath', type=str, help='path of target directory')
  parser.add_argument('class_num', type=int, help='number of classes')
  parser.add_argument('--height', type=int, default=100, help='image height (default: 100)')
  parser.add_argument('--width', type=int, default=100, help='image width (default: 100)')
  parser.add_argument('--channel_num', type=int, default=1, 
                      help='number of image\'s color channels (default: 1)')
  parser.add_argument('--batch_size', type=int, default=256, 
                      help='batch size (default: 256)')
  parser.add_argument('--epochs', type=int, default=50, 
                      help='number of training epoch (default: 50)')
  parser.add_argument('-g', '--generator', action='store_true', default=False, 
                      help='mini-batch training using generator (default: False)')
  parser.add_argument('-a', '--augmentation', action='store_true', default=False, 
                      help='data augmentation (default: False)')
  args = parser.parse_args()

  start = time.time()

  # Build and compile Lenet
  model_filepath = mergeFilepaths(args.target_dirpath, 'model.hdf5')
  model = buildLenet(class_num=args.class_num, height=args.height, width=args.width,
                     channel_num=args.channel_num)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Draw model structure
  struct_img_filepath = mergeFilepaths(args.target_dirpath, 'ModelStructure.png')
  plot_model(model, to_file=struct_img_filepath, show_shapes=True, show_layer_names=True)

  # Load image filepaths and these label from train.txt and validation.txt respectively
  train_txt_filepath = mergeFilepaths(args.target_dirpath, 'train.txt')
  train_filepaths, train_labels = storeImageFilepathsAndLabels(txt_filepath=train_txt_filepath, 
                                                               class_num=args.class_num)
  val_txt_filepath = mergeFilepaths(args.target_dirpath, 'validation.txt')
  val_filepaths, val_labels = storeImageFilepathsAndLabels(txt_filepath=val_txt_filepath, 
                                                           class_num=args.class_num)

  # Set callbacks
  checkpointer = ModelCheckpoint(filepath=model_filepath, monitor='val_acc', verbose=1, 
                                 save_best_only=True, save_weights_only=False)
  early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
  tensor_board = TensorBoard(log_dir=args.target_dirpath+'graph', histogram_freq=0, 
                             write_graph=False, write_images=False)
  
  # Training
  if args.generator:
    # Mini-batch training using fit_generator
    steps_per_epoch = int(len(train_filepaths) / args.batch_size)
    validation_steps = int(len(val_filepaths) / args.batch_size)
    history = model.fit_generator(
      generateArrays(filepaths=train_filepaths, labels=train_labels, batch_size=args.batch_size, 
                     channel_num=args.channel_num),
      steps_per_epoch=steps_per_epoch,
      epochs=args.epochs, 
      validation_data=generateArrays(filepaths=val_filepaths, labels=val_labels, 
                                     batch_size=args.batch_size, channel_num=args.channel_num),
      validation_steps=validation_steps,
      callbacks=[checkpointer, early_stopping, tensor_board]
    )
  else:
    if args.augmentation:
      print('\n[Load and augment training images]')
      train_imgs, train_labels = loadImagesThenAugmentation(train_filepaths, train_labels,
                                                            channel_num=args.channel_num)
      print('\n[Load and augment validation images]')
      val_imgs, val_labels = loadImagesThenAugmentation(val_filepaths, val_labels, 
                                                        channel_num=args.channel_num)
    else:
      print('\n[Load training images which are already augmented]')
      train_imgs = loadAlreadyAugmentedImages(train_filepaths, channel_num=args.channel_num)
      print('\n[Load validation images which are already augmented]')
      val_imgs = loadAlreadyAugmentedImages(val_filepaths, channel_num=args.channel_num)
    history = model.fit(
      train_imgs, train_labels, batch_size=args.batch_size, 
      epochs=args.epochs, validation_data=(val_imgs, val_labels), 
      shuffle=True, callbacks=[checkpointer, early_stopping, tensor_board]
    )
    
  # Save train history
  history_filepath = mergeFilepaths(args.target_dirpath, 'history.pickle')
  with open(history_filepath, mode='wb') as f:
    pickle.dump(history.history, f)

  processing_time = time.time() - start
  showProcessingTime(processing_time)


if __name__ == "__main__":
  main()