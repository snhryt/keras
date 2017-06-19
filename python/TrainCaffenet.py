#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from argparse import ArgumentParser
from keras.models import Sequential, load_model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model


def buildCaffenet(class_num, channel_num=3, img_width=224, img_height=224):
  from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
  from keras.layers.core import Dense, Dropout, Activation, Flatten
  from keras.layers.normalization import BatchNormalization
  from keras.regularizers import l2

  weight_decay = 0.0005

  model = Sequential()
  # Conv1
  model.add(Convolution2D(nb_filter=96, nb_row=5, nb_col=5, border_mode='valid', 
                          input_shape=(img_width, img_height, channel_num), subsample=(2, 2), 
                          W_regularizer=l2(weight_decay), name='conv1'))
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
  model.add(Dense(4096,  W_regularizer=l2(weight_decay), name='fc6'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc7
  model.add(Dense(4096,  W_regularizer=l2(weight_decay), name='fc7'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc8
  model.add(Dense(class_num, W_regularizer=l2(weight_decay), name='fc8'))
  model.add(Activation('softmax'))

  model.summary()
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


def countLineNum(txt_filepath):
  line_num = 0
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      line_num += 1
      line = f.readline()
  return line_num


def storeNpyFilepathsAndLabels(txt_filepath, class_num):
  print('\n[Loading %s]' % txt_filepath)
  total = countLineNum(txt_filepath)
  print('.. total: %d' % total)
  
  npy_filepaths, labels = [], []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      npy_filepath = line.split(' ')[0]
      label = line.split(' ')[1]
      label = label.split('\n')[0]
      npy_filepaths.append(npy_filepath)
      labels.append(label)
      line = f.readline()
  return npy_filepaths, labels


def main():
  usage = 'Usage: python {} TARGET_DIRPATH CLASS_NUM [--batch_size] [--epochs] [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument('target_dirpath', type=str, 
                      help='The path of target directory (must end "/")')
  parser.add_argument('class_num', type=int, help='The number of classes')
  parser.add_argument('-b', '--batch_size', type=int, default=256, help='Batch size')
  parser.add_argument('-e', '--epochs', type=int, default=50, help='The number of training epoch')  
  args = parser.parse_args()

  channel_num, img_width, img_height = 1, 100, 100
  model_filepath = args.target_dirpath + 'model.hdf5'
  model = buildCaffenet(class_num=args.class_num, channel_num=channel_num, img_width=img_width,
                        img_height=img_height)
  # モデル構造の書き出し
  png_filepath = args.target_dirpath + 'ModelStructure.png'
  plot_model(model, to_file=png_filepath, show_shapes=True, show_layer_names=True)
  
  train_txt_filepath = args.target_dirpath + 'train.txt'
  train_filepaths, train_labels = storeNpyFilepathsAndLabels(txt_filepath=train_txt_filepath, 
                                                              class_num=args.class_num)
  '''
  val_txt_filepath = args.target_dirpath + 'validation.txt'
  val_filepaths, val_labels = storeNpyFilepathsAndLabels(txt_filepath=val_txt_filepath, 
                                                          class_num=args.class_num)
  
  # callbacksの設定
  checkpointer = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, 
                                  save_best_only=True, save_weights_only=False)
  # patientce： 何回連続で損失の最小値が更新されなかったらループを止めるか
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
  tensor_board = TensorBoard(log_dir=args.target_dirpath+'graph', histogram_freq=0, write_graph=True)
  
  # generator を生成して batch_size 枚の画像ごとに学習を行う
  def generateArrays(filepaths, labels):
    batch_features = np.zeros((args.batch_size, img_width, img_height, channel_num))
    batch_labels = np.zeros((args.batch_size, args.class_num))
    for i in range(len(filepaths)):
      for j in range(args.batch_size):
        array = np.load(filepaths[i])
        array /= 255.
        batch_features[j] = array.reshape(1, img_width, img_height, channel_num)        
        batch_labels[j] = np_utils.to_categorical(labels[i], args.class_num)
      yield (batch_features, batch_labels)

  steps_per_epoch = int(len(train_filepaths) / args.batch_size)
  validation_steps = int(len(val_filepaths) / args.batch_size)
  history = model.fit_generator(generateArrays(filepaths=train_filepaths, labels=train_labels),
                                steps_per_epoch=steps_per_epoch, epochs=args.epochs, 
                                validation_data=generateArrays(filepaths=val_filepaths,  
                                                                labels=val_labels),
                                validation_steps=validation_steps,
                                callbacks=[checkpointer, early_stopping, tensor_board])
  '''

  checkpointer = ModelCheckpoint(model_filepath, monitor='acc', verbose=1, 
                                  save_best_only=True, save_weights_only=False)
  early_stopping = EarlyStopping(monitor='acc', patience=10, verbose=1)
  tensor_board = TensorBoard(log_dir=args.target_dirpath+'graph', histogram_freq=0, write_graph=True)
  def loadNpys(filepaths):
    features = np.zeros((len(filepaths), img_width, img_height, channel_num))
    for i in range(len(filepaths)):
      array = np.load(filepaths[i])
      array /= 255.
      features[i] = array
    return features

  train_features = loadNpys(train_filepaths)
  train_labels = np_utils.to_categorical(train_labels, args.class_num)
  history = model.fit(train_features, train_labels, batch_size=args.batch_size, epochs=args.epochs,
                      callbacks=[checkpointer, early_stopping, tensor_board])
  
  # 学習履歴の保存
  history_filepath = args.target_dirpath + 'history.pickle'
  with open(history_filepath, mode='wb') as f:
    pickle.dump(history.history, f)

  

if __name__ == "__main__":
  main() 
