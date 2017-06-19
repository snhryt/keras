#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import np_utils


def showAndSaveLossGraph(output_dirpath):
  history_filepath = output_dirpath + 'history.pickle'
  with open(history_filepath, mode='rb') as f:
    history = pickle.load(f)
  
  # y軸の範囲限定版
  img_filepath1 = output_dirpath + 'history.png'
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(2, 1, 1)
  ax1.plot(history['loss'], 'o-', label='train-loss')
  #ax1.plot(history['val_loss'], 'o-', label='val-loss')
  ax1.plot(history['acc'], 'o-', label='train-accuracy')
  #ax1.plot(history['val_acc'], 'o-', label='val-accuracy')
  ax1.set_title('train history')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')
  ax1.set_ylim(0.0, 1.2)
  ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
  fig1.savefig(img_filepath1)
  plt.pause(2.0)
  plt.close()
  
  # y軸の範囲限定しない版
  img_filepath2 = output_dirpath + "history_NoEdit.png"
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(2, 1, 1)
  ax2.plot(history['loss'], 'o-', label='loss')
  #ax2.plot(history['val_loss'], 'o-', label='val-loss')
  ax2.plot(history['acc'], 'o-', label='accuracy')
  #ax2.plot(history['val_acc'], 'o-', label='val-accuracy')
  ax2.set_title('train history')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('loss')
  ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
  fig2.savefig(img_filepath2)
  plt.close()

  val_result_filepath = output_dirpath + 'ValidationResult.txt'

def storeTestImages(dirpath):
  print('\n[Loading images from %s]' % dirpath)
  img_arrays, filenames = [], []
  counter = 0
  for filename in os.listdir(dirpath):
    if os.path.splitext(filename)[1] != '.png':
      continue
    filenames.append(filename)
    filepath = dirpath + filename
    img = image.load_img(filepath, grayscale=True, target_size=(100,100))
    img_array = image.img_to_array(img) / 255
    img_arrays.append(img_array)
    counter += 1

  img_arrays = np.array(img_arrays)
  return img_arrays, filenames

def storeFontIndex(txt_filepath):
  dict = {}
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      index = int(line.split(' ')[0])
      font = line.split(' ')[1]
      font = font.split('\n')[0]
      dict[index] = font
      line = f.readline()
  return dict

def getFontImage(target_font, dirpath):
  for filename in os.listdir(dirpath):
    stem, ext = os.path.splitext(filename)
    if ext == '.png':
      if '_' in stem:
        font = stem.split('_')[1]
      else:
        font = stem      
      
      if font == target_font:
        _img = image.load_img(dirpath+filename, grayscale=True, target_size=(100, 100))
        return _img
  


def main():
  usage = 'Usage: python {} TARGET_DIRPATH CLASS_NUM [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument('target_dirpath', type=str, 
                      help='The path of target directory (must end "/")')
  parser.add_argument('font_list_filepath', type=str, 
                      help='List of selected fonts and these indices')  
  parser.add_argument('class_num', type=int, help='Number of classes')
  parser.add_argument('alphabet', type=str, help='Target alphabet(e.g. "capA", "smallA") or "all"')  
  args = parser.parse_args()

  model_filepath = args.target_dirpath + 'model.hdf5'
  model = load_model(model_filepath)
  
  # lossのグラフ表示&保存
  showAndSaveLossGraph(output_dirpath=args.target_dirpath)

  # Validation accuracyの書き出し
  '''
  val_filepath = args.target_dirpath + 'validation.txt'
  val_features, val_labels = [], []
  with open(val_filepath) as f:
    line = f.readline()
    while line:
      array = np.load(line.split(' ')[0])
      array /= 255.
      val_features.append(array)
      label = line.split(' ')[1]
      label = label.split('\n')[0]
      val_labels.append(label)
      line = f.readline()
  val_features = np.array(val_features)
  val_labels = np_utils.to_categorical(val_labels, args.class_num)

  score = model.evaluate(val_features, val_labels)
  val_result_filepath = args.target_dirpath + 'ValidationResult.txt'
  with open(val_result_filepath, 'w') as f:
    f.writelines('Best validation loss: ' + str('%03.3f' % score[0]) + '\n')
    f.writelines('Best validation accuracy: ' + str('%03.3f' % score[1]))
  '''

  # テスト画像およびその識別結果のグラフを表示&保存
  def outputImages(target_font=None):
    IMG_DIRPATH = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages/'
    if args.alphabet == 'all':
      test_img_dirpath = IMG_DIRPATH + target_font + '/'
      output_dirpath = args.target_dirpath + target_font + '/'
      font_img_dirpath = IMG_DIRPATH + 'FontMontage/all/'
    else:
      output_dirpath = args.target_dirpath + 'OutputImages/'      
      if 'cap' in args.alphabet:
        test_img_dirpath = IMG_DIRPATH + 'capital/' + args.alphabet[-1] + '/'
      elif 'small' in args.alphabet:
        test_img_dirpath = IMG_DIRPATH + 'small/' + args.alphabet[-1] + '/'
      font_img_dirpath = test_img_dirpath      
    test_img_arrays, filenames = storeTestImages(test_img_dirpath)
    results = model.predict(test_img_arrays, batch_size=64, verbose=1)
    
    if not os.path.isdir(output_dirpath):
      os.mkdir(output_dirpath)
    
    # key:フォントのindex、 value:フォント
    index_font_dict = storeFontIndex(args.font_list_filepath)

    print('\n[Saving result images to %s]' % output_dirpath)
    TOP_NUM = 3
    for i in range(len(results)):
      test_img = image.array_to_img(test_img_arrays[i], scale=True)
      font_prob_dict = {}
      for j in range(args.class_num):
        font_prob_dict[index_font_dict[j]] = results[i][j]
      font_prob_dict = sorted(font_prob_dict.items(), key=lambda x:x[1], reverse=True)
      
      fig, axes = plt.subplots(nrows=2,ncols=TOP_NUM, figsize=(8,6))
      axes[0, 0].imshow(test_img, cmap='Greys_r')
      for j in range(1, TOP_NUM):
        axes[0, j].axis('off')    
      for j in range(TOP_NUM):
        if j >= args.class_num:
          axes[1, j].axis('off')
          break
        font = font_prob_dict[j][0]
        prob = font_prob_dict[j][1]
        if prob < 0.1:
          axes[1, j].axis('off')          
        else:
          font_img = getFontImage(font, font_img_dirpath)
          axes[1, j].imshow(font_img, cmap='Greys_r')
          title = 'top' + str(j + 1) + ': ' + str('%03.1f' % (prob*100)) + '%'
          axes[1, j].set_title(title, fontsize=16)

      fig.tight_layout() # タイトルとラベルが被らないようにする
      # plt.pause(0.7)
      plt.close()
      output_img_filepath = output_dirpath + filenames[i]
      fig.savefig(output_img_filepath)
      if (i + 1) % 10 == 0:
        print('.. %d/%d result images are saved' % (i + 1, len(results)))
    
  outputImages()
    
      

if __name__ == "__main__":
  main() 