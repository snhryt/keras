#! /usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential, load_model, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils, plot_model
import os
import numpy as np
import matplotlib.pyplot as plt


def storeImagesAndLabels(txt_filepath, class_num):
  from PIL import Image

  print("\n[Loading %s]" % txt_filepath)
  img_arrays, labels = [], []
  img_num = 0
  f = open(txt_filepath)
  line = f.readline()
  while line:
    img = load_img(line.split(" ")[0], grayscale=True, target_size=(100,100))
    img_array = img_to_array(img)
    img_array /= 255
    img_arrays.append(img_array)
    index = line.split(" ")[1]
    index = index.split("\n")[0]
    labels.append(int(index))
    img_num += 1
    line = f.readline()
    if img_num % 100 == 0:
      print(".. %d images(indices) are loaded" % img_num)
  f.close()

  # kerasに渡すためにnumpy配列に変換
  img_arrays = np.array(img_arrays)
  # convert class vectors to binary class matrices
  labels = np_utils.to_categorical(labels, class_num)
  return (img_arrays, labels)


def buildLenet(class_num, channel_num=1, img_width=28, img_height=28):
  model = Sequential()
  model.add(Convolution2D(20, 5, 5, strides=(1, 1), input_shape=(channel_num,img_width,img_height),
            name="conv1"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Convolution2D(50, 5, 5, strides=(1, 1), name="conv2"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(500))
  model.add(Activation("relu"))
  model.add(Dense(class_num))
  model.add(Activation("softmax"))
  return model


def buildVggModel(class_num, channel_num=3, img_width=224, img_height=224):
  model = Sequential()
  model.add(ZeroPadding2D((1,1),input_shape=(channel_num,img_width,img_height)))
  model.add(Convolution2D(64, 3, 3, activation="relu", name="conv1_1"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation="relu", name="conv1_2"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation="relu", name="conv2_1"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation="relu", name="conv2_2"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_1"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_2"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_3"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_1"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_2"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_3"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_1"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_2"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_3"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(4096))
  model.add(Dropout(0.5))
  model.add(Dense(4096))
  model.add(Dropout(0.5))
  model.add(Dense(class_num))
  model.add(Activation("softmax"))

  return model


def Caffenet_initialization(shape, name=None):
  """
  Custom weights initialization
  From Convolution2D:
  weights: list of Numpy arrays to set as initial weights.
          The list should have 2 elements, of shape `(input_dim, output_dim)`
          and (output_dim,) for weights and biases respectively.
  From train_val.prototxt
  weight_filler
  {
    type: "gaussian"
    std: 0.01
  }
  bias_filler
  {
    type: "constant"
    value: 0
  }
  Si pasamos esta funcion en el parametro init, pone este peso a las W y las b las deja a 0 (comprobado leyendo el codigo de Keras)
  """

  from keras import backend as K
  mu, sigma = 0, 0.01
  return K.variable(np.random.normal(mu, sigma, shape), name=name)


def buildCaffenet(class_num, channel_num=3, img_width=224, img_height=224):
  from keras.layers.normalization import BatchNormalization
  from keras.regularizers import l2

  weight_decay = 0.0005

  model = Sequential()
  # Conv1
  model.add(Convolution2D(nb_filter=96, nb_row=11, nb_col=11, border_mode='valid', 
            input_shape=(img_width, img_height, channel_num), init=Caffenet_initialization, 
            subsample=(4, 4), W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(BatchNormalization())
  # Conv2
  model.add(Convolution2D(256, 5, 5, border_mode='same', init=Caffenet_initialization, 
            W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(BatchNormalization())
  # Conv3
  model.add(Convolution2D(384, 3, 3, border_mode='same', init=Caffenet_initialization,
            W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  # Conv4
  model.add(Convolution2D(384, 3, 3, border_mode='same', init=Caffenet_initialization, 
            W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  # Conv5
  model.add(Convolution2D(256, 3, 3, border_mode='same', init=Caffenet_initialization, 
            W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  
  model.add(Flatten())
  # Fc6
  model.add(Dense(4096, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc7
  model.add(Dense(4096, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Fc8
  model.add(Dense(class_num, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
  model.add(Activation('softmax'))

  return model


def main():
  #from keras.optimizers import RMSprop
  class_num = 2
  batch_size = 64
  epoch_num = 5
  parent_dirpath = "/media/snhryt/Data/Research_Master/keras/MyWork/2fonts_2class/"
  model_filepath = parent_dirpath + "caffenet_weights.h5"
  model = buildCaffenet(class_num=class_num, channel_num=1, img_width=100, img_height=100)

  if (os.path.exists(model_filepath)):
    print("\n[Loading model weights from %s]" % model_filepath)
    model.load_weights(model_filepath)
  else:
    train_txt_filepath = parent_dirpath + "train.txt"
    val_txt_filepath = parent_dirpath + "validation.txt"
    train_img_arrays, train_labels = storeImagesAndLabels(train_txt_filepath, class_num) 
    val_img_arrays, val_labels = storeImagesAndLabels(val_txt_filepath, class_num) 
      
    # モデルのコンパイル
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # モデルの学習
    # patientce： 何回連続で損失の最小値が更新されなかったらループを止めるか
    # verbose： コマンドラインにコメントを出力する場合は"1"と設定
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    model.fit(train_img_arrays, train_labels, batch_size=batch_size, epochs=epoch_num,
              verbose=1, validation_data=(val_img_arrays, val_labels), 
              callbacks=[early_stopping])
    
    # 学習
    score = model.evaluate(val_img_arrays, val_labels, verbose=1)
    print("\n**Validation score: %f" % score[0])
    print("**Validation accuracy: %f" % score[1])
    
    # モデルの保存
    model.save_weights(model_filepath)
    json_string = model.to_json()
    json_filepath = parent_dirpath + "caffenet.json"
    f = open(json_filepath, "w")
    f.write(json_string)
    f.close()


  """
  # モデルの評価
  classes = model.predict(X_test, batch_size=batch_size, verbose=True)
  print(classes[0])
  """



if __name__ == "__main__":
  main()