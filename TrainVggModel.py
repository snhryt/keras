#! /usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential, load_model, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils import np_utils, plot_model
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def storeImagesAndLabels(txt_filepath, class_num):
  print('\n[Loading %s]' % txt_filepath)
  img_arrays, labels = [], []
  img_num = 0
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      img = load_img(line.split(' ')[0], grayscale=True, target_size=(100,100))
      img_array = img_to_array(img)
      img_array /= 255
      img_arrays.append(img_array)
      index = line.split(' ')[1]
      index = index.split('\n')[0]
      labels.append(int(index))
      img_num += 1
      line = f.readline()
      if img_num % 100 == 0:
        print('.. %d images(indices) are loaded' % img_num)

  # kerasに渡すためにnumpy配列に変換
  img_arrays = np.array(img_arrays)
  # convert class vectors to binary class matrices
  labels = np_utils.to_categorical(labels, class_num)
  return (img_arrays, labels)

def storeImages(dirpath):
  print('\n[Loading images from %s]' % dirpath)
  img_arrays, filenames = [], []
  counter = 0
  for filename in os.listdir(dirpath):
    if os.path.splitext(filename)[1] != '.png':
      continue
    filenames.append(filename)
    filepath = dirpath + '/' + filename
    img = load_img(filepath, grayscale=True, target_size=(100,100))
    img_array = img_to_array(img) / 255
    img_arrays.append(img_array)
    counter += 1

  img_arrays = np.array(img_arrays)
  return (img_arrays, filenames)


def buildLenet(class_num, channel_num=1, img_width=28, img_height=28):
  model = Sequential()
  model.add(Convolution2D(20, 5, 5, strides=(1, 1), input_shape=(channel_num,img_width,img_height),
                          name='conv1'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Convolution2D(50, 5, 5, strides=(1, 1), name='conv2'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(500))
  model.add(Activation('relu'))
  model.add(Dense(class_num))
  model.add(Activation('softmax'))
  return model


def buildCaffenet(class_num, channel_num=3, img_width=224, img_height=224):
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
  model.add(Convolution2D(256, 5, 5, border_mode='same', W_regularizer=l2(weight_decay), 
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

  return model


def main():
  class_num = 2
  parent_dirpath = '/media/snhryt/Data/Research_Master/keras/MyWork/2fonts_2class/'
  weights_filepath = parent_dirpath + 'weights.hdf5'
  json_filepath = parent_dirpath + 'model.json'

  if (os.path.exists(weights_filepath) and os.path.exists(json_filepath)):
    print('\n[Loading the model and its weights]')
    with open(json_filepath) as f:
      model = model_from_json(f.read())
    model.load_weights(weights_filepath)
  else:
    # 画像とラベルの読み込み
    train_txt_filepath = parent_dirpath + 'train.txt'
    val_txt_filepath = parent_dirpath + 'validation.txt'
    train_img_arrays, train_labels = storeImagesAndLabels(train_txt_filepath, class_num) 
    val_img_arrays, val_labels = storeImagesAndLabels(val_txt_filepath, class_num) 
      
    # モデルのコンパイル
    model = buildCaffenet(class_num=class_num, channel_num=1, img_width=100, img_height=100)
    model.summary()
    #from keras.optimizers import RMSprop
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # callbacksの設定
    tbcb = TensorBoard(log_dir=parent_dirpath+'graph', histogram_freq=0, write_graph=True)
    checkpointer = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, 
                                   save_best_only=True)
    # patientce： 何回連続で損失の最小値が更新されなかったらループを止めるか
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # モデルの学習
    batch_size = 256
    epoch_num = 100
    history = model.fit(train_img_arrays, train_labels, batch_size=batch_size, epochs=epoch_num,
                        verbose=1, validation_data=(val_img_arrays, val_labels), 
                        callbacks=[tbcb, early_stopping, checkpointer])
    
    # モデル構造を.json&.png形式で保存
    png_filepath = parent_dirpath + 'model.png'
    model_json = model.to_json()
    with open(json_filepath, mode='w') as f:
      f.write(model_json)
    plot_model(model, to_file=png_filepath)

    # 学習済の重みの保存
    model.save_weights(weights_filepath)
    
    # 学習履歴の保存
    history_filepath = parent_dirpath + 'history.pickle'
    with open(history_filepath, mode='wb') as f:
      pickle.dump(history.history, f)
    
    # lossのグラフ表示&保存
    img_filepath1 = parent_dirpath + 'history.png'
    img_filepath2 = parent_dirpath + "history_NoEdit.png"
    history = None
    with open(history_filepath, mode='rb') as f:
      history = pickle.load(f)

    # y軸の範囲限定版
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(history['loss'], 'o-', label='loss')
    ax1.plot(history['val_loss'], 'o-', label='val-loss')
    ax1.plot(history['acc'], 'o-', label='accuracy')
    ax1.plot(history['val_acc'], 'o-', label='val-accuracy')
    ax1.set_title('train history')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim(0.0, 1.2)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
    fig1.savefig(img_filepath1)
    plt.pause(2.0)
    plt.close()
    # y軸の範囲限定しない版
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(2, 1, 1)
    ax2.plot(history['loss'], 'o-', label='loss')
    ax2.plot(history['val_loss'], 'o-', label='val-loss')
    ax2.plot(history['acc'], 'o-', label='accuracy')
    ax2.plot(history['val_acc'], 'o-', label='val-accuracy')
    ax2.set_title('train history')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
    fig2.savefig(img_filepath2)
    plt.close()
  
  '''
  # 中間層の出力の可視化
  from sklearn.preprocessing import MinMaxScaler
  weights = model.get_layer(name='conv1').get_weights()[0].transpose(3, 2, 0, 1)
  filter_num, channel_num, row, cols = weights.shape
  plt.figure()
  for i in range(filter_num):
    img = weights[i, 0]
    scaler = MinMaxScaler(feature_range=(0, 255))
    img = scaler.fit_transform(img)
    plt.subplot(1, filter_num, i + 1)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
  plt.show()
  '''

  # 学習済のモデルを使って、trainとvalidationとは別の画像でテスト
  #font = 'Aerolinea'
  #font = 'AccoladeSerial-Regular'
  #font = 'A750-Sans-Medium-Regular'
  font = 'A850-Roman-Regular'
  test_img_dirpath = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages/font/' + font
  output_dirpath = parent_dirpath + font
  test_img_arrays, filenames = storeImages(test_img_dirpath)

  if not os.path.isdir(output_dirpath):
    os.mkdir(output_dirpath)
  classes = model.predict(test_img_arrays, batch_size=64, verbose=1)
  
  # テスト画像およびその識別結果のグラフを表示&保存
  for i in range(0, len(classes)): 
    fig = plt.figure(figsize=(4,3))
    
    ax1 = fig.add_subplot(1, 2, 1)
    img = array_to_img(test_img_arrays[i], scale=True)
    ax1.imshow(img, cmap='Greys_r')

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.array([0, 1])
    y = classes[i]
    labels = ['sans-serif', 'serif']
    ax2.bar(left=x, height=y, tick_label=labels, align='center', width=0.5)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel('probability')
    ax2.grid(True)

    fig.tight_layout() # タイトルとラベルが被らないようにする
    #plt.pause(0.7)
    plt.close()
    
    output_img_filepath = output_dirpath + '/' + filenames[i]
    fig.savefig(output_img_filepath)
    if i % 10 == 0 and i != 0:
      print('.. Output %d/%d images' % (i, len(classes)))


if __name__ == '__main__':
  main()