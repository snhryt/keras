#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import random
import pickle
import cv2
from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils, plot_model
from keras.preprocessing import image
from CommonFunc import (isImage, countLineNum, showProcessingTime, storeSingleImageShaped4Keras, 
                        storeImagesShaped4Keras)
from augmentation import augmentImage
import numpy as np
import matplotlib.pyplot as plt

class Training:
  '''
  Class for CNN training.
  '''
  def __init__(self, target_dirpath, class_num, height, width, channel_num, batch_size, epochs, 
               network, need_augmentation, need_generator):
    '''
    Initialize Training class.
    '''
    self.target_dirpath = target_dirpath
    self.class_num = class_num
    self.height = height
    self.width = width
    self.channel_num = channel_num
    if channel_num == 1:
      self.color = 'bin'
    elif channel_num == 3:
      self.color = 'color'
    self.batch_size = batch_size
    self.epochs = epochs
    self.network = network
    self.need_augmentation = need_augmentation
    self.need_generator = need_generator


  def buildLenet(self):
    '''
    Return lenet model. Lenet has 2 conv. layers and 2 f.c. layers.
    '''
    from keras.models import Sequential
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.core import Dense, Flatten, Activation

    model = Sequential()
    # model.add(Convolution2D(filters=20, kernel_size=(5, 5), strides=(1, 1), 
    #                         input_shape=(self.height, self.width, self.channel_num), name='conv1'))
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
    '''
    Return caffenet model. Caffenet has 5 conv. layers and 3 f.c. layers.
    '''
    from keras.models import Sequential
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2

    weight_decay = 0.0005

    model = Sequential()
    # conv1
    model.add(
      Convolution2D(filters=96, kernel_size=(5, 5), border_mode='valid', 
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


  def storeFilepathsAndLabels(self, txt_filepath):
    '''
    Return 2 lists: filepaths and labels, which are stored from a text file.
    Each line of the text file must be "<full filepath of a image> <class label>" style.
    '''
    filepaths = []
    labels = []
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

  def storeImagesFromFilepaths(self, filepaths):
    '''
    Return a numpy array of images stored from a list of filepaths.
    '''
    imgs = np.empty((len(filepaths), self.height, self.width, self.channel_num))
    for i, filepath in enumerate(filepaths):
      imgs[i] = storeSingleImageShaped4Keras(filepath, color=self.color)
      if (i + 1) % 1000 == 0:
        print('.. {0}/{1} images are stored'.format(i + 1, len(filepaths)))
    return imgs


  def storeAugmentedBinImages(self, filepaths, labels, augmentation_num=20):
    '''
    Return 2 numpy arrays, which are augmented images and these (augmented) one-hot labels.
    '''
    output_imgs = np.empty((augmentation_num * len(filepaths), self.height, self.width, 1))
    output_labels = np.empty((augmentation_num * len(filepaths), self.class_num))
    counter = 0
    for i, (filepath, label) in enumerate(zip(filepaths, labels)):
      gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      augmented_imgs = augmentImage(
        bin_img, 
        augmentation_num=augmentation_num,
        scale_down=True,
        rotation=False,
        stretch=False,
        jaggy=True,
        noise=False
      )
      for augmented_img in augmented_imgs:
        augmented_img = augmented_img[:, :, np.newaxis]
        output_imgs[counter] = augmented_img.astype('float32') / 255.
        output_labels[counter] = label
        counter += 1
      # if i % 100 == 0:
      #   print('.. {0}/{1} images are loaded and augmented'.format(i, len(imgs)))
    return output_imgs, output_labels


  def generateTrainData(self, filepaths, labels):
    '''
    Generate mini batch of images and these labels for input to fit_generator.
    '''
    indices = [i for i in range(len(filepaths))]
    if self.need_augmentation:
      while True:
        random_indices = random.sample(indices, self.batch_size)
        random_filepaths = [filepaths[index] for index in random_indices]
        random_labels = [labels[index] for index in random_indices]
        batch_imgs, batch_labels = self.storeAugmentedBinImages(random_filepaths, random_labels)
        yield batch_imgs, batch_labels
    else:
      batch_imgs = np.empty((self.batch_size, self.height, self.width, self.channel_num))
      batch_labels = np.empty((self.batch_size, self.class_num))
      while True:
        random_indices = random.sample(indices, self.batch_size)
        for i, index in enumerate(random_indices):
          batch_imgs[i] = storeSingleImageShaped4Keras(filepaths[index], color=self.color)
          batch_labels[i] = labels[index]
        yield batch_imgs, batch_labels


  def saveLossAndAccGraph(self, history_filepath):
    '''
    Draw and save epoch-loss graph and epoch-accuracy graph.
    '''
    with open(history_filepath, mode='rb') as f:
      history = pickle.load(f)

    stopped_epoch = len(history['acc']) - 1
    output_img_filepath = os.path.join(self.target_dirpath, 'history.png')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Left: epoch-accuracy graph
    axes[0].plot(history['acc'], 'o-', label='train')
    axes[0].plot(history['val_acc'], 'o-', label='validation')
    highest_train_acc = max(history['acc'])
    highest_val_acc = max(history['val_acc'])
    title = ('highest -> train:{:.3f}, '.format(highest_train_acc) + 
             'validation:{:.3f}'.format(highest_val_acc))
    axes[0].set_title(title)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlim(0, stopped_epoch)
    # axes[0].set_ylim(0.0, 1.0)
    # axes[0].legend()
    # Right: epoch-loss graph
    axes[1].plot(history['loss'], 'o-', label='train')
    axes[1].plot(history['val_loss'], 'o-', label='validation')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].set_xlim(0, stopped_epoch)
    axes[1].legend()
    fig.tight_layout()
    plt.pause(5.0)
    plt.close()
    fig.savefig(output_img_filepath)


  def train(self):
    '''
    Main function of Training class.
    '''
    # Build and compile network
    if self.network == 'lenet':
      model = self.buildLenet()
    elif self.network == 'caffenet':
      model = self.buildCaffenet()
    else:
      raise NameError('"self.network" must be "lenet" or "caffenet"')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Draw model structure
    struct_img_filepath = os.path.join(self.target_dirpath, 'ModelStructure.png')
    plot_model(model, to_file=struct_img_filepath, show_shapes=True, show_layer_names=True)

    # Set callbacks
    model_filepath = os.path.join(self.target_dirpath, 'model.hdf5')
    monitor = 'val_acc'
    checkpointer = ModelCheckpoint(filepath=model_filepath, monitor=monitor, verbose=1, 
                                   save_best_only=True, save_weights_only=False)
    early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=1)
    tensor_board = TensorBoard(log_dir=self.target_dirpath+'/TensorBoardData', histogram_freq=0, 
                               write_graph=False, write_images=False)
    

    # Store filepaths and class labels, then store images
    train_txt_filepath = os.path.join(self.target_dirpath, 'train.txt')
    val_txt_filepath = os.path.join(self.target_dirpath, 'validation.txt')
    train_filepaths, train_labels = self.storeFilepathsAndLabels(train_txt_filepath)
    val_filepaths, val_labels = self.storeFilepathsAndLabels(val_txt_filepath)
    
    print('\n[Store validation images]')
    val_imgs = self.storeImagesFromFilepaths(val_filepaths)

    # Training
    if self.need_generator:
      # steps_per_epoch = 10000
      steps_per_epoch = len(train_filepaths) / self.batch_size
      # validation_steps = len(val_filepaths) / self.batch_size
      history = model.fit_generator(
        self.generateTrainData(train_filepaths, train_labels),
        steps_per_epoch=steps_per_epoch,
        epochs=self.epochs,
        validation_data=(val_imgs, val_labels),
        callbacks=[checkpointer, early_stopping, tensor_board]
      )
    else:
      print('\n[Store training images]')
      train_imgs = self.storeImagesFromFilepaths(train_filepaths)
      for i, filepath in enumerate(train_filepaths):
        train_imgs[i] = storeSingleImageShaped4Keras(filepath, color=self.color)
      if train_imgs.shape[0] != train_labels.shape[0]:
        raise ValueError()

      if self.need_augmentation:
        train_imgs, train_labels = self.storeAugmentedBinImages(train_imgs, train_labels)
      history = model.fit(
        train_imgs, 
        train_labels, 
        batch_size=self.batch_size, 
        epochs=self.epochs, 
        validation_data=(val_imgs, val_labels), 
        shuffle=True, 
        callbacks=[checkpointer, early_stopping, tensor_board] 
      )
    
    # Show and save epochs-loss & epochs-accuracy graph
    history_filepath = os.path.join(self.target_dirpath, 'history.pickle')
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
  parser.add_argument('--network', type=str, default='lenet', 
                      help='network name ("lenet" or "caffenet", default: "lenet")')
  parser.add_argument('-a', '--augmentation', action='store_true', default=False, 
                      help='data augmentation (default: False)')
  parser.add_argument('-g', '--generator', action='store_true', default=False, 
                      help='mini-batch training using generator (default: False)')
  args = parser.parse_args()

  start = time.time()
  x = Training(
    target_dirpath=args.target_dirpath, 
    class_num=args.class_num, 
    height=args.height, 
    width=args.width, 
    channel_num=args.channel_num, 
    batch_size=args.batch_size, 
    epochs=args.epochs, 
    network=args.network, 
    need_augmentation=args.augmentation,
    need_generator=args.generator
  )
  x.train()
  processing_time = time.time() - start
  showProcessingTime(processing_time)


if __name__ == "__main__":
  main()