#! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
import random
import pickle
from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model
from CommonFunc import mergeFilepaths, countLineNum, showProcessingTime, loadSingleImage
from augmentation import augmentImage
import numpy as np
import matplotlib.pyplot as plt

class Training:
  def __init__(self, target_dirpath, class_num, height=100, width=100, channel_num=1, 
               batch_size=256, epochs=10, need_generator=False, need_augmentation=False):
    self.target_dirpath = target_dirpath
    self.class_num = class_num
    self.height = height
    self.width = width
    self.channel_num = channel_num
    self.batch_size = batch_size
    self.epochs = epochs
    self.need_generator = need_generator
    self.need_augmentation = need_augmentation


  def buildLenet(self):
    from keras.models import Sequential
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.core import Dense, Flatten, Activation

    model = Sequential()
    model.add(Convolution2D(filters=20, kernel_size=(8, 8), strides=(3, 3), 
                            input_shape=(self.height, self.width, self.channel_num), name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(filters=50, kernel_size=(5, 5), strides=(1, 1), name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(self.class_num))
    model.add(Activation('softmax'))
    model.summary()
    return model


  def buildCaffenet(self):
    from keras.models import Sequential
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2

    weight_decay = 0.0005

    model = Sequential()
    # conv1
    model.add(
      Convolution2D(filters=96, kernel_size=(5,5), border_mode='valid', 
                    input_shape=(self.height, self.width, self.channel_num), 
                    subsample=(2, 2), W_regularizer=l2(weight_decay), name='conv1')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    # conv2
    model.add(Convolution2D(256, 4, 4, border_mode='same', W_regularizer=l2(weight_decay), 
                            name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    # conv3
    model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay), 
                            name='conv3'))
    model.add(Activation('relu'))
    # conv4
    model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay),
                            name='conv4'))
    model.add(Activation('relu'))
    # conv5
    model.add(Convolution2D(256, 3, 3, border_mode='same', W_regularizer=l2(weight_decay),
                            name='conv5'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(Flatten())
    # fc6
    model.add(Dense(4096, W_regularizer=l2(weight_decay), name='fc6'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # fc7
    model.add(Dense(4096, W_regularizer=l2(weight_decay), name='fc7'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # fc8
    model.add(Dense(self.class_num, W_regularizer=l2(weight_decay), name='fc8'))
    model.add(Activation('softmax'))
    model.summary()
    return model


  def storeImageFilepathsAndLabels(self, txt_filepath):
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
    labels = np_utils.to_categorical(labels, self.class_num)
    return filepaths, labels


  def generateArrays(self, filepaths, labels):
    batch_imgs = np.empty((self.batch_size, self.height, self.width, self.channel_num))
    batch_labels = np.empty((self.batch_size, self.class_num))
    indices = [i for i in range(len(filepaths))]
    while True:
      random_indices = random.sample(indices, self.batch_size)
      for i, index in enumerate(random_indices):
        img = loadSingleImage(filepaths[index], channel_num=self.channel_num)
        img = img.astype('float32') / 255.
        batch_imgs[i] = img
        batch_labels[i] = labels[index]
      yield batch_imgs, batch_labels


  def loadImagesThenAugmentation(self, filepaths, labels, augmentation_num=100):
    output_imgs = output_labels = []
    counter = 0
    for filepath, label in zip(filepaths, labels):
      img = loadSingleImage(filepath, channel_num=self.channel_num)
      augmented_imgs = augmentImage(img, augmentation_num=augmentation_num)
      for j, augmented_img in enumerate(augmented_imgs):
        f_augmented_img = augmented_img.astype('float32') / 255.
        output_imgs.append(f_augmented_img)
        output_labels.append(label)
      counter += 1
      if counter % 100 == 0:
        print('.. {0}/{1} images are loaded and augmented'.format(counter, len(filepaths)))
    output_imgs = np.array(output_imgs)
    output_labels = np.array(output_labels)
    return output_imgs, output_labels


  def loadAlreadyAugmentedImages(self, filepaths):
    output_imgs = [None] * len(filepaths)
    for i, filepath in enumerate(filepaths):
      img = loadSingleImage(filepath, channel_num=self.channel_num)
      output_imgs[i] = img.astype('float32') / 255.
      if (i + 1) % 100 == 0:
        print('.. {0}/{1} images are loaded'.format(i + 1, len(filepaths)))
    return np.array(output_imgs)


  def saveLossAndAccGraph(self, history_filepath):
    with open(history_filepath, mode='rb') as f:
      history = pickle.load(f)
    highest_train_acc = max(history['acc'])
    highest_val_acc = max(history['val_acc'])

    # Arrange ylim
    img_filepath1 = mergeFilepaths(self.target_dirpath, 'history.png')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(history['loss'], 'o-', label='train-loss')
    ax1.plot(history['val_loss'], 'o-', label='val-loss')
    ax1.plot(history['acc'], 'o-', label='train-accuracy')
    ax1.plot(history['val_acc'], 'o-', label='val-accuracy')
    title = ('train history (highest accuracy -> train:{0:0.3f}, '.format(highest_train_acc) + 
             'val:{0:0.3f})'.format(highest_val_acc))
    ax1.set_title(title)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_xlim(0, len(history['acc']))
    ax1.set_ylim(0.0, 1.2)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
    fig1.savefig(img_filepath1)
    plt.pause(5.0)
    plt.close()
    
    # Not arrange ylim
    img_filepath2 = mergeFilepaths(self.target_dirpath, 'history_NoYlimit.png')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(2, 1, 1)
    ax2.plot(history['loss'], 'o-', label='loss')
    ax2.plot(history['val_loss'], 'o-', label='val-loss')
    ax2.plot(history['acc'], 'o-', label='accuracy')
    ax2.plot(history['val_acc'], 'o-', label='val-accuracy')
    ax2.set_title(title)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_xlim(0, len(history['acc']))    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
    fig2.savefig(img_filepath2)
    plt.close()


  def allFunc(self):
    # Build and compile Lenet
    model = self.buildLenet()
    # model = self.buildCaffenet()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Draw model structure
    struct_img_filepath = mergeFilepaths(self.target_dirpath, 'ModelStructure.png')
    plot_model(model, to_file=struct_img_filepath, show_shapes=True, show_layer_names=True)

    # Load image filepaths and these label from train.txt and validation.txt respectively
    train_txt_filepath = mergeFilepaths(self.target_dirpath, 'train.txt')
    train_filepaths, train_labels = (
      self.storeImageFilepathsAndLabels(txt_filepath=train_txt_filepath)
    )
    val_txt_filepath = mergeFilepaths(self.target_dirpath, 'validation.txt')    
    val_filepaths, val_labels = self.storeImageFilepathsAndLabels(txt_filepath=val_txt_filepath)

    # Set callbacks
    model_filepath = mergeFilepaths(self.target_dirpath, 'lenet.hdf5')
    # model_filepath = mergeFilepaths(self.target_dirpath, 'caffenet.hdf5')            
    monitor = 'val_acc'
    checkpointer = ModelCheckpoint(filepath=model_filepath, monitor=monitor, verbose=1, 
                                   save_best_only=True, save_weights_only=False)
    early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=1)
    tensor_board = TensorBoard(log_dir=self.target_dirpath+'/TensorBoardData', histogram_freq=0, 
                               write_graph=False, write_images=False)

    # Training
    if self.need_generator:
      # Mini-batch training using fit_generator
      steps_per_epoch = int(len(train_filepaths) / self.batch_size)
      validation_steps = int(len(val_filepaths) / self.batch_size)
      history = model.fit_generator(
        self.generateArrays(filepaths=train_filepaths, labels=train_labels),
        steps_per_epoch=steps_per_epoch, epochs=self.epochs, 
        validation_data=self.generateArrays(filepaths=val_filepaths, labels=val_labels),
        validation_steps=validation_steps, callbacks=[checkpointer, early_stopping, tensor_board]
      )
    else:
      if self.need_augmentation:
        print('\n[Load and augment training images]')
        train_imgs, train_labels = self.loadImagesThenAugmentation(train_filepaths, train_labels)
        print('\n[Load and augment validation images]')
        val_imgs, val_labels = self.loadImagesThenAugmentation(val_filepaths, val_labels)
      else:
        print('\n[Load training images which are already augmented]')
        train_imgs = self.loadAlreadyAugmentedImages(train_filepaths)
        print('\n[Load validation images which are already augmented]')
        val_imgs = self.loadAlreadyAugmentedImages(val_filepaths)
      history = model.fit(
        train_imgs, train_labels, batch_size=self.batch_size, epochs=self.epochs, 
        validation_data=(val_imgs, val_labels), shuffle=True, 
        callbacks=[checkpointer, early_stopping, tensor_board]
      )
    
    # Show and save epochs-loss & epochs-accuracy graph
    history_filepath = mergeFilepaths(self.target_dirpath, 'history.pickle')
    with open(history_filepath, mode='wb') as f:
      pickle.dump(history.history, f)
    self.saveLossAndAccGraph(history_filepath)
    

def main():
  parser = ArgumentParser()
  parser.add_argument('target_dirpath', type=str, help='path of target directory')
  parser.add_argument('class_num', type=int, help='number of classes')
  parser.add_argument('--height', type=int, default=100, help='image height (default: 100)')
  parser.add_argument('--width', type=int, default=100, help='image width (default: 100)')
  parser.add_argument('--channel_num', type=int, default=1, 
                      help='number of color channels of image (default: 1)')
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
  x = Training(args.target_dirpath, args.class_num, args.height, args.width, args.channel_num, 
               args.batch_size, args.epochs, args.generator, args.augmentation)
  x.allFunc()
  processing_time = time.time() - start
  showProcessingTime(processing_time)


if __name__ == "__main__":
  main()