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
  line_num = 0
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      line_num += 1
      line = f.readline()

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
        print('.. %d/%d images(indices) are loaded' % (img_num, line_num))

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

  return model


def main():
  class_num = 100
  parent_dirpath = '/media/snhryt/Data/Research_Master/keras/MyWork/100fonts_100class/'
  model_filepath = parent_dirpath + 'model.hdf5'

  if (os.path.exists(model_filepath)):
    print('\n[Loading and the model and its weights]')
    model = load_model(model_filepath)
  else:
    channel_num, img_width, img_height = 1, 100, 100
    # 画像とラベルの読み込み
    train_txt_filepath = parent_dirpath + 'train.txt'
    val_txt_filepath = parent_dirpath + 'validation.txt'
      
    # モデルのコンパイル
    model = buildCaffenet(class_num=class_num, channel_num=channel_num, img_width=img_width, 
                          img_height=img_height)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # callbacksの設定
    tbcb = TensorBoard(log_dir=parent_dirpath+'graph', histogram_freq=0, write_graph=True)
    checkpointer = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, 
                                   save_best_only=True, save_weights_only=False)
    # patientce： 何回連続で損失の最小値が更新されなかったらループを止めるか
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # モデルの学習
    batch_size = 256
    epoch_num = 50
    def generator(txt_filepath, batch_size):
      print('\n[Loading %s]\n' % txt_filepath)
      line_num = 0
      with open(txt_filepath) as f:
        line = f.readline()
        while line:
          line_num += 1
          line = f.readline()
      print('.. #Data = %d' % line_num)
      
      img_num = 0
      #with open(txt_filepath) as f:
      f = open(txt_filepath)
      line = f.readline()
      while line:
        batch_features = np.zeros((batch_size, img_width, img_height, channel_num))
        batch_labels = np.zeros((batch_size, class_num))
        for i in range(batch_size):
          img = load_img(line.split(' ')[0], grayscale=True, target_size=(img_width,img_height))
          img_array = img_to_array(img)
          img_array /= 255
          batch_features[i] = np.array(img_array).reshape(1, img_width, img_height, channel_num)
          index = line.split(' ')[1]
          index = index.split('\n')[0]
          label = int(index)
          batch_labels[i] = np_utils.to_categorical(label, class_num)
          assert img_num < line_num, '***ERROR! img_num < line_num'
          img_num += 1
          line = f.readline()
        yield (batch_features, batch_labels)
      f.close()
    
    history = model.fit_generator(generator(train_txt_filepath, batch_size), epochs=epoch_num, 
                                  steps_per_epoch=1000)
    '''
    history = model.fit(train_img_arrays, train_labels, batch_size=batch_size, epochs=epoch_num,
                        verbose=1, validation_data=(val_img_arrays, val_labels), 
                        callbacks=[tbcb, checkpointer, early_stopping])
    '''
    
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

    # モデル構造の書き出し
    png_filepath = parent_dirpath + 'model.png'
    plot_model(model, to_file=png_filepath, show_shapes=True, show_layer_names=True)
    
    # checkpointer で保存した中で最もベストだった重みファイルを読み込み、 model を再定義
    model = None
    model = load_model(model_filepath)
    # （後々使うかもしれないので）学習済の重み単体でも保存
    weights_filepath = parent_dirpath + 'weights.hdf5'
    model.save_weights(weights_filepath)

    # 一番よかったvalidation accuracyの書き出し
    val_result_filepath = parent_dirpath + 'ValidationResult.txt'
    score = model.evaluate(val_img_arrays, val_labels, verbose=0)
    with open(val_result_filepath, 'w') as f:img = load_img(line.split(' ')[0], grayscale=True, target_size=(100,100))
      f.writelines('Best validation loss: ' + str(score[0]) + '\n')
      f.writelines('Best validation accuracy: ' + str(score[1]) + '\n')
  
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

  # テスト画像およびその識別結果のグラフを表示&保存
  def outputImages(font):
    test_img_dirpath = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages/font/' + font
    output_dirpath = parent_dirpath + font
    test_img_arrays, filenames = storeImages(test_img_dirpath)

    if not os.path.isdir(output_dirpath):
      os.mkdir(output_dirpath)
    classes = model.predict(test_img_arrays, batch_size=64, verbose=1)

    if class_num == 2:
      x = np.array([])
      for j in range(class_num):
        x = np.append(x, j) 
      labels = ['sans-serif', 'serif']

      for i in range(0, len(classes)): 
        fig = plt.figure(figsize=(4,3))
        
        ax1 = fig.add_subplot(1, 2, 1)
        img = array_to_img(test_img_arrays[i], scale=True)
        ax1.imshow(img, cmap='Greys_r')

        ax2 = fig.add_subplot(1, 2, 2)
        y = classes[i]
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
    else:
      top3_x, top3_y = [0, 0, 0], [0.0, 0.0, 0.0]
      for i in range(class_num):
        y = classes[i]
        if y >= top3_y[0]:
          tmp_y1, tmp_y2, tmp_x1, tmp_x2 = top3_y[0], top3_y[1], top3_x[0], top3_x[1]
          top3_y[0], top3_y[1], top3_y[2] = y, tmp_y1, tmp_y2
          top3_x[0], top3_x[1], top3_x[2] = i, tmp_x1, tmp_x2          
        elif y >= top3_y[1]:
          tmp_y, tmp_x = top3_y[1], top3_x[1]
          top3_y[1], top3_y[2] = y, tmp_y
          top3_x[1], top3_x[2] = x, tmp_x          
        elif y >= top3_y[2]:
          top3_y[2] = y
          top3_x[2] = x


    
    
    

  # 学習済のモデルを使って、trainとvalidationとは別の画像でテスト
  outputImages(font='Aerolinea')
  outputImages(font='AccoladeSerial-Regular')
  outputImages(font='A750-Sans-Medium-Regular')
  outputImages(font='A850-Roman-Regular')


if __name__ == '__main__':
  main()