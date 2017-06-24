#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from argparse import ArgumentParser
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

def showAndSaveLossGraph(output_dirpath):
  history_filepath = mergeFilepaths(output_dirpath, 'history.pickle')
  with open(history_filepath, mode='rb') as f:
    history = pickle.load(f)
  
  # y軸の範囲限定版
  img_filepath1 = mergeFilepaths(output_dirpath, 'history.png')
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(2, 1, 1)
  ax1.plot(history['loss'], 'o-', label='train-loss')
  ax1.plot(history['val_loss'], 'o-', label='val-loss')
  ax1.plot(history['acc'], 'o-', label='train-accuracy')
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
  img_filepath2 = mergeFilepaths(output_dirpath, 'history_NoEdit.png')
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

def mergeFilepaths(dirpath, filename):
  if dirpath[-1] == '/':
    filepath = dirpath + filename
  else:
    filepath = dirpath + '/' + filename
  return filepath

def countLineNum(txt_filepath):
  line_num = 0
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      line_num += 1
      line = f.readline()
  return line_num

def isImage(filepath):
  img_exts = ['.png', '.jpg']
  ext = os.path.splitext(filepath)[1]
  for i in range(len(img_exts)):
    if ext == img_exts[i]:
      return True
  return False

def storeImagesAndLabels(txt_filepath, class_num):
  print('\n[Loading "{}"]'.format(txt_filepath))
  total = countLineNum(txt_filepath)
  print('.. total: {} files'.format(total))

  img_arrays, labels = [], []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      filepath = line.split(' ')[0]
      if isImage(filepath):
        img = image.load_img(filepath, grayscale=True)
        array = image.img_to_array(img) / 255.
      elif os.path.splitext(filepath)[1] == '.npy':
        array = np.load(filepath) / 255.
      img_arrays.append(array)
      label = line.split(' ')[1]
      label = label.split('\n')[0]
      labels.append(label)
      line = f.readline()
  img_arrays = np.array(img_arrays)
  labels = np_utils.to_categorical(labels, class_num)
  return img_arrays, labels


def storeImagesAndFilenamesFromDirpath(path):
  print('\n[Loading images from "{}"]'.format(path))  
  img_arrays, filenames = [], []
  if os.path.isdir(path):
    for filename in os.listdir(path):
      if isImage(filename):
        filenames.append(filename)
        img = image.load_img(path + filename, grayscale=True, target_size=(100, 100))
        img_array = image.img_to_array(img) / 255.
        img_arrays.append(img_array)
  elif os.path.isfile(path):
    filenames.append(os.path.basename(path))
    img = image.load_img(path, grayscale=True, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.
    img_arrays.append(img_array)
  img_arrays = np.array(img_arrays)
  return img_arrays, filenames


def storeFontIndex(txt_filepath):
  index_font_dict = {}
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      index = int(line.split(' ')[0])
      font = line.split(' ')[1]
      font = font.split('\n')[0]
      index_font_dict[index] = font
      line = f.readline()
  return index_font_dict


def getFontImage(target_font, dirpath):
  for filename in os.listdir(dirpath):
    stem, ext = os.path.splitext(filename)
    if isImage(filename):
      font = stem.split('_')[1] if ('_' in stem) else stem
      if font == target_font:
        img = image.load_img(mergeFilepaths(dirpath, filename), grayscale=True)
        return img


def main():
  usage = 'Usage: python {} TARGET_DIRPATH CLASS_NUM [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument('target_dirpath', type=str, help='path of target directory')
  parser.add_argument('font_list_filepath', type=str, 
                      help='list of selected fonts and these indices')  
  parser.add_argument('class_num', type=int, help='number of classes')
  parser.add_argument('-a', '--alphabet', type=str, default=None,
                      help='target alphabet; e.g. "capA", "smallA" (default: None)')
  # parser.add_argument('-f', '--font', type=str, default=None, help='target font (default: None)') 
  args = parser.parse_args()

  model_filepath = mergeFilepaths(args.target_dirpath, 'model.hdf5')
  model = load_model(model_filepath)

  # lossのグラフ表示&保存
  if not os.path.isfile(mergeFilepaths(args.target_dirpath, 'history.png')):
    showAndSaveLossGraph(output_dirpath=args.target_dirpath)

  # Validation accuracyの書き出し
  result_filepath = mergeFilepaths(args.target_dirpath, 'ValidationResult.txt')
  if not os.path.isfile(result_filepath):
    val_filepath = mergeFilepaths(args.target_dirpath, 'validation.txt')
    val_img_arrays, val_labels = storeImagesAndLabels(val_filepath, args.class_num)
    score = model.evaluate(val_img_arrays, val_labels)
    with open(result_filepath, 'w') as f:
      f.writelines('Best validation loss: ' + str('%03.3f' % score[0]) + '\n')
      f.writelines('Best validation accuracy: ' + str('%03.3f' % score[1]))
  
  # key:フォントのindex、 value:フォント
  index_font_dict = storeFontIndex(args.font_list_filepath)

  # テスト画像およびその識別結果を描画&保存
  def drawRecogResultImages(font=None):
    PARENT_DIRPATH = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages/'
    output_dirpath = mergeFilepaths(args.target_dirpath, 'OutputImages')
    if font and args.alphabet:
      if 'cap' in args.alphabet:
        test_img_dirpath = PARENT_DIRPATH + 'capital/' + args.alphabet[-1]
      elif 'small' in args.alphabet:
        test_img_dirpath = PARENT_DIRPATH + 'small/' + args.alphabet[-1]        
      test_img_filename = args.alphabet + '_' + font + '.png'
      test_img_filepath = mergeFilepaths(test_img_dirpath, test_img_filename)
      font_img_dirpath = test_img_dirpath
      test_img_arrays, filenames = storeImagesAndFilenamesFromDirpath(path=test_img_filepath)      
    else:
      if font:
        test_img_dirpath = PARENT_DIRPATH + 'font/' + font
        output_dirpath = mergeFilepaths(output_dirpath, font)
        font_img_dirpath = PARENT_DIRPATH + 'FontMontage/all'
      elif args.alphabet:
        if 'cap' in args.alphabet:
          test_img_dirpath = PARENT_DIRPATH + 'capital/' + args.alphabet[-1]
        elif 'small' in args.alphabet:
          test_img_dirpath = PARENT_DIRPATH + 'small/' + args.alphabet[-1]
        font_img_dirpath = test_img_dirpath 
      test_img_arrays, filenames = storeImagesAndFilenamesFromDirpath(path=test_img_dirpath)
    results = model.predict(test_img_arrays, batch_size=256, verbose=1)
    
    if not os.path.isdir(output_dirpath):
      os.mkdir(output_dirpath)
  
    TOP_N = 3
    if args.class_num < TOP_N:
      TOP_N = args.class_num
    print('\n[Saving recognition result images to {}]'.format(output_dirpath))
    for i in range(len(results)):
      font_prob_dict = {}
      for j in range(args.class_num):
        font_prob_dict[index_font_dict[j]] = results[i][j]
      # probability が高い順にソート
      font_prob_dict = sorted(font_prob_dict.items(), key=lambda x:x[1], reverse=True)

      ncols = 1
      # もし TOP_N にprobability=0.0のものが含まれていれば描画しない
      for j in range(TOP_N):
        if font_prob_dict[j][1] * 1000 <= 1:
          break
        ncols += 1
      fig, axes = plt.subplots(nrows=1, ncols=ncols)

      test_img = image.array_to_img(test_img_arrays[i], scale=True)
      axes[0].imshow(test_img, cmap='Greys_r')
      axes[0].set_title('input image', fontsize=16)

      for j in range(1, ncols):
        font = font_prob_dict[j - 1][0]
        str_prob = '{0:3.1f}'.format(font_prob_dict[j - 1][1] * 100)
        font_img = getFontImage(font, font_img_dirpath)
        axes[j].imshow(font_img, cmap='Greys_r')
        title = 'prob=' + str_prob + '[%]'
        axes[j].set_title(title, fontsize=14)

      fig.tight_layout() # タイトルとラベルが被らないようにする
      # plt.pause(0.7)
      plt.close()
      output_img_filepath = mergeFilepaths(output_dirpath, filenames[i])
      fig.savefig(output_img_filepath)
      if (i + 1) % 10 == 0:
        print('.. {0}/{1} result images are saved'.format(i + 1, len(results)))
    
  drawRecogResultImages(font='A750-Sans-Medium-Regular')
  drawRecogResultImages(font='A850-Roman-Regular')
  drawRecogResultImages(font='Accidental-Presidency')
  drawRecogResultImages(font='Action-Man')
  drawRecogResultImages(font='AdelonSerial-Xbold-Regular')
  drawRecogResultImages(font='Aesop-Regular')
  drawRecogResultImages(font='Airmole')
  drawRecogResultImages(font='AkazanLt-Regular')
  drawRecogResultImages(font='Amplitude-BRK')
  drawRecogResultImages(font='Avondale-SC')
  

if __name__ == "__main__":
  main() 