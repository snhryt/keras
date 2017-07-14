#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import random
import pickle
import cv2
from argparse import ArgumentParser
from keras.models import Sequential
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model
import numpy as np

from CommonFunc import mergeFilepaths
from augmentation import augmentImage

HEIGHT, WIDTH, CHANNEL_NUM = 100, 100, 1

def buildCaffenet(class_num, height=224, width=224, channel_num=3):
  from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
  from keras.layers.core import Dense, Dropout, Activation, Flatten
  from keras.layers.normalization import BatchNormalization
  from keras.regularizers import l2

  weight_decay = 0.0005

  model = Sequential()
  # Conv1
  model.add(
    Convolution2D(filters=96, kernel_size=(5,5), border_mode='valid', 
                  input_shape=(height, width, channel_num), 
                  subsample=(2, 2), W_regularizer=l2(weight_decay), name='conv1')
  )
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(BatchNormalization())
  # Conv2
  model.add(Convolution2D(256, 4, 4, border_mode='same', W_regularizer=l2(weight_decay), 
                          name='conv2'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(BatchNormalization())
  # Conv3
  model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay), 
                          name='conv3'))
  model.add(Activation('relu'))
  # Conv4
  model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay),
                          name='conv4'))
  model.add(Activation('relu'))
  # Conv5
  model.add(Convolution2D(256, 3, 3, border_mode='same', W_regularizer=l2(weight_decay),
                          name='conv5'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  
  model.add(Flatten())
  # Fc6
  model.add(Dense(4096, W_regularizer=l2(weight_decay), name='fc6'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc7
  model.add(Dense(4096, W_regularizer=l2(weight_decay), name='fc7'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc8
  model.add(Dense(class_num, W_regularizer=l2(weight_decay), name='fc8'))
  model.add(Activation('softmax'))

  model.summary()
  return model


def storeImageFilepathsAndLabels(txt_filepath, class_num):
  print('\n[Loading "{}"]'.format(txt_filepath))
  #total = countLineNum(txt_filepath)
  #print('.. total: {} files'.format(total))
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
  labels = np_utils.to_categorical(labels, class_num)
  return filepaths, labels


def generateArrays(filepaths, labels, batch_size, class_num):
  batch_img_arrays = np.empty((batch_size, WIDTH, HEIGHT, CHANNEL_NUM))
  batch_labels = np.empty((batch_size, class_num))
  img_ext = os.path.splitext(filepaths[0])[1]
  file_num = len(filepaths)
  while True:
    for i in range(batch_size):
      index = random.choice(range(file_num))
      img = image.load_img(filepaths[index], grayscale=True, target_size=(WIDTH, HEIGHT))
      array = image.img_to_array(img) / 255.
      batch_img_arrays[i] = array.reshape(1, WIDTH, HEIGHT, CHANNEL_NUM)
      batch_labels[i] = labels[index]
    yield (batch_img_arrays, batch_labels)

  # augmentation_num = augmented_imgs.shape[0]
  # data_num = augmentation_num * batch_size
  # font_indices = [i for i in range(len(filepaths))]
  # data_indices = [i for i in range(data_num)]
  # tmp_batch_imgs = np.empty((data_num, HEIGHT, WIDTH, CHANNEL_NUM))
  # tmp_batch_labels = np.empty((data_num, class_num))
  # batch_imgs = np.empty((data_num, HEIGHT, WIDTH, CHANNEL_NUM))
  # batch_labels = np.empty((data_num, class_num))

  # while True:
  #   random_font_indices = random.sample(font_indices, batch_size)
  #   for i, index in enumerate(random_font_indices):
  #     img = cv2.imread(filepaths[index], cv2.IMREAD_GRAYSCALE)
  #     bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  #     augmented_imgs = augmentImage(bin_img)
  #     augmented_imgs = augmented_imgs.reshape(augmentation_num, HEIGHT, WIDTH, CHANNEL_NUM)
  #     tmp_batch_imgs[i:i + augmentation_num] = augmented_imgs[0:augmentation_num]
  #     tmp_batch_labels[i:i + augmentation_num] = labels[index]

  #   random_data_indices = random.sample(data_indices, data_num)
  #   for i, index in enumerate(random_data_indices):
  #     batch_imgs[i] = tmp_batch_imgs[index]
  #     batch_labels[i] = tmp_batch_labels[index]
  #   yield (batch_imgs, batch_labels)



def loadImages(filepaths):
  img_ext = os.path.splitext(filepaths[0])[1]
  print ('\n[Loading {0} {1} images]'.format(len(filepaths), img_ext))
  img_arrays = np.zeros((len(filepaths), WIDTH, HEIGHT, CHANNEL_NUM))
  for i, path in enumerate(filepaths):
    if img_ext == '.png' or img_ext == '.jpg':
      img = image.load_img(path, grayscale=True, target_size=(WIDTH, HEIGHT))
      array = image.img_to_array(img) / 255.
    elif img_ext == '.npy':
      array = np.load(path) / 255.
    img_arrays[i] = array.reshape(1, WIDTH, HEIGHT, CHANNEL_NUM)
    if (i + 1) % 1000 == 0:
      print ('.. {} images are loaded'.format(i + 1))
  print ('.. {} images are loaded'.format(len(filepaths)))    
  return img_arrays

def storeAugmentedImages(imgs, labels, class_num, augmentation_num=None):
  all_imgs = np.empty((0, HEIGHT, WIDTH, 1))
  all_labels = np.empty((0, class_num))
  for img, label in zip(imgs, labels):
    augmented_imgs = augmentImage(img, augmentation_num=augmentation_num)
    augmented_imgs /= 255.
    augmented_imgs = augmented_imgs.reshape(augmented_imgs.shape[0], HEIGHT, WIDTH, 1)
    augmented_labels = np.empty((augmented_imgs.shape[0], class_num))
    augmented_labels[:,:] = label
    all_imgs = np.append(all_imgs, augmented_imgs, axis=0)
    all_labels = np.append(all_labels, augmented_labels, axis=0)
  return all_imgs, all_labels


def main():
  # コマンドラインから引数を与える
  usage = ('Usage: python {} TARGET_DIRPATH CLASS_NUM [--generator_training]'.format(__file__)
           + '[--batch_size] [--epochs] [--help]')
  parser = ArgumentParser(usage)
  parser.add_argument('target_dirpath', type=str, help='path of target directory')
  parser.add_argument('class_num', type=int, help='number of classes')
  parser.add_argument('-g', '--generator_training', action='store_true', default=False, 
                      help='mini-batch training using generator (default: False)')  
  parser.add_argument('-b', '--batch_size', type=int, default=256, 
                      help='batch size (default: 256)')
  parser.add_argument('-e', '--epochs', type=int, default=50, 
                      help='number of training epoch (default: 50)')  
  args = parser.parse_args()

  start = time.time()

  # caffenetの構築およびコンパイル
  model_filepath = mergeFilepaths(args.target_dirpath, 'model.hdf5')
  model = buildCaffenet(class_num=args.class_num, height=HEIGHT, width=WIDTH,
                        channel_num=CHANNEL_NUM)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # モデル構造の書き出し
  png_filepath = mergeFilepaths(args.target_dirpath, 'ModelStructure.png')
  plot_model(model, to_file=png_filepath, show_shapes=True, show_layer_names=True)

  # train, validation に使うデータのファイルパスおよびそのラベルを格納
  train_txt_filepath = mergeFilepaths(args.target_dirpath, 'train.txt')
  train_filepaths, train_labels = storeImageFilepathsAndLabels(txt_filepath=train_txt_filepath, 
                                                               class_num=args.class_num)
  val_txt_filepath = mergeFilepaths(args.target_dirpath, 'validation.txt')
  val_filepaths, val_labels = storeImageFilepathsAndLabels(txt_filepath=val_txt_filepath, 
                                                           class_num=args.class_num)

  # callbacksの設定
  checkpointer = ModelCheckpoint(filepath=model_filepath, monitor='loss', verbose=1, 
                                  save_best_only=True, save_weights_only=False)
  # patientce： 何回連続で損失の最小値が更新されなかったらループを止めるか
  early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
  tensor_board = TensorBoard(log_dir=args.target_dirpath+'graph', histogram_freq=0, 
                             write_graph=False, write_images=False)

  # generator を生成して batch_size 枚の画像(.npy)ごとに学習を行う
  if args.generator_training:
    steps_per_epoch = int(len(train_filepaths) / args.batch_size)
    validation_steps = int(len(val_filepaths) / args.batch_size)
    history = model.fit_generator(
      generateArrays(filepaths=train_filepaths, labels=train_labels, batch_size=args.batch_size, 
                     class_num=args.class_num),
      steps_per_epoch=steps_per_epoch,
      epochs=args.epochs, 
      validation_data=generateArrays(filepaths=val_filepaths, labels=val_labels, 
                                     batch_size=args.batch_size, class_num=args.class_num),
      validation_steps=validation_steps,
      callbacks=[checkpointer, early_stopping, tensor_board]
    )
  else:
    # train, validationの画像をすべて読み込んで学習を行う
    # train_img_arrays = loadImages(train_filepaths)
    # val_img_arrays = loadImages(val_filepaths)
    # history = model.fit(
    #   train_img_arrays, train_labels, batch_size=args.batch_size, epochs=args.epochs, 
    #   validation_data=(val_img_arrays,val_labels),
    #   callbacks=[checkpointer, early_stopping, tensor_board]
    # )
    train_imgs = np.empty((0, HEIGHT, WIDTH))
    print('\n[Augment train data]')
    for i, filepath in enumerate(train_filepaths):
      img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      train_imgs = np.append(train_imgs, bin_img[np.newaxis,:,:], axis=0)
      if (i + 1) % 100 == 0:
        print('.. {} images are augmented'.format(i + 1))
    val_imgs = np.empty((0, HEIGHT, WIDTH))
    print('\n[Augment validation data]')    
    for i, filepath in enumerate(val_filepaths):
      img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      val_imgs = np.append(val_imgs, bin_img[np.newaxis,:,:], axis=0)
      if (i + 1) % 100 == 0:
        print('.. {} images are augmented'.format(i + 1))

    augmented_train_imgs, augmented_train_labels = storeAugmentedImages(
        train_imgs, train_labels, args.class_num, augmentation_num=100
    )
    augmented_val_imgs, augmented_val_labels = storeAugmentedImages(
        val_imgs, val_labels, args.class_num, augmentation_num=100
    )
    history = model.fit(
      augmented_train_imgs, augmented_train_labels, batch_size=args.batch_size, epochs=args.epochs, 
      validation_data=(augmented_val_imgs, augmented_val_labels), shuffle=True, 
      callbacks=[checkpointer, early_stopping, tensor_board]
    )
  
  # 学習履歴の保存
  history_filepath = mergeFilepaths(args.target_dirpath, 'history.pickle')
  with open(history_filepath, mode='wb') as f:
    pickle.dump(history.history, f)

  end = time.time() - start
  print('processing time: {}[sec]'.format(end))

if __name__ == "__main__":
  main() 
