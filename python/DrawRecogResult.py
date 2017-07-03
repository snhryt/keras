#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from keras.models import load_model
from keras.preprocessing import image
from CommonFunc import loadImages, mergeFilepaths
import numpy as np
import matplotlib.pyplot as plt

usage = ('Usage: python {} IMG_PATH TARGET_DIRPATH'.format(__file__)
          + '[--height] [--width] [--gray] [--help]')
parser = ArgumentParser(usage)
parser.add_argument('img_path', type=str, 
                    help='single test image file path OR path of directory including test images')
parser.add_argument('target_dirpath', type=str, 
                    help='path of directory including "model.hdf5"')
# parser.add_argument('font_list_filepath', type=str, 
#                     help='path of list including selected fonts and these indices')
parser.add_argument('--height', type=int, default=100, help='images height (default: 100)')
parser.add_argument('--width', type=int, default=100, help='images width (default: 100)')
parser.add_argument('-g', '--gray', action='store_true', default=False, help='load images as gray')
args = parser.parse_args()

# load model
model_filepath = mergeFilepaths(args.target_dirpath, 'model.hdf5')
model = load_model(model_filepath)

# load test image(s) and filename(s)
test_imgs, test_filenames = loadImages(args.img_path, need_filename=True, height=args.height, 
                                       width=args.width, gray=args.gray)
test_imgs /= 255.
results = model.predict(test_imgs, batch_size=256, verbose=1)
# results.shape = (#data, #class)

# make output directory ifnot exist
output_dirpath = mergeFilepaths(args.target_dirpath, 'OutputImages')
if not os.path.isdir(output_dirpath):
  os.mkdir(output_dirpath)

# draw test image itself and a bar graph of classified result
x = [i for i in range(results.shape[1])]
for img, filename, probs in zip(test_imgs, test_filenames, results):
  if args.gray:
    img = img.reshape(args.height, args.width, 1)
  else:
    img = img.reshape(args.height, args.width, 3)    
  img = image.array_to_img(img, scale=True)
  output_img_filepath = mergeFilepaths(output_dirpath, filename)

  # get class index that has highest probability
  highest_prob_index = probs.argsort()[::-1][0]

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4,3))
  # axes[0](left side): test image
  axes[0].imshow(img, cmap='Greys_r')
  axes[0].set_title('input image')
  # axes[1](right side): bar graph
  axes[1].bar(left=x, height=probs, width=1.0, color='blue', align='center')
  axes[1].set_xlim(0, results.shape[1])
  axes[1].set_ylim(0.0, 1.0)
  xlabel = 'class label (top:' + str(highest_prob_index) + ')'
  ylabel = 'probability (top:' + '{0:0.3f}'.format(probs[highest_prob_index]) + ')'
  axes[1].set_xlabel(xlabel)
  axes[1].set_ylabel(ylabel)
  axes[1].grid(True)
  fig.tight_layout()
  plt.close()
  fig.savefig(output_img_filepath)
  print('kado desu ensyu ganbatte!!') # thank you!!!!!
    

'''
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
  usage = 'Usage: python {} IMG_PATH TARGET_DIRPATH FOFNT_LIST_FILEPATH [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument('img_path', type=str, help='test image file path')
  parser.add_argument('target_dirpath', type=str, help='target directory path')
  parser.add_argument('font_list_filepath', type=str, 
                      help='path of list including selected fonts and these indices')   
  args = parser.parse_args()

  model_filepath = mergeFilepaths(args.target_dirpath, 'model.hdf5')
  model = load_model(model_filepath)
  
  # key:フォントのindex、 value:フォント
  index_font_dict = storeFontIndex(args.font_list_filepath)
  font_index_dict = {}
  for i in range(len(index_font_dict)):
    font_index_dict[index_font_dict[i]] = i
  

  # テスト画像およびその識別結果を描画&保存
  output_dirpath = mergeFilepaths(args.target_dirpath, 'OutputImages')
  if not os.path.isdir(output_dirpath):
    os.mkdir(output_dirpath)

  test_img, filename = storeImagesAndFilenamesFromDirpath(path=args.img_filepath)
  test_img = test_img.reshape(1, 100, 100, 1)
  result = model.predict(test_img, batch_size=256, verbose=1)

  output_img_filepath = mergeFilepaths(output_dirpath, filename)

  TOP_N = 3
  if len(index_font_dict) < TOP_N:
    TOP_N = len(index_font_dict)

  font_prob_dict = {}
  for j in range(len(index_font_dict)):
    font_prob_dict[index_font_dict[j]] = result[0][j]
  # probability が高い順にソート
  font_prob_dict = sorted(font_prob_dict.items(), key=lambda x:x[1], reverse=True)

  ncols = 1
  # もし TOP_N にprobability=0.0のものが含まれていれば描画しない
  for j in range(TOP_N):
    if font_prob_dict[j][1] * 1000 <= 1:
      break
    ncols += 1
  fig, axes = plt.subplots(nrows=1, ncols=ncols)

  test_img = test_img.reshape(100, 100, 1)
  test_img = image.array_to_img(test_img, scale=True)
  axes[0].imshow(test_img, cmap='Greys_r')
  axes[0].set_title('input image', fontsize=16)

  font_img_dirpath = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages/capital/A/'
  for j in range(1, ncols):
    font = font_prob_dict[j - 1][0]
    index = str(font_index_dict[font])
    str_prob = '{0:3.1f}'.format(font_prob_dict[j - 1][1] * 100)
    font_img = getFontImage(font, font_img_dirpath)
    axes[j].imshow(font_img, cmap='Greys_r')
    title = index + ':prob=' + str_prob + '[%]'
    axes[j].set_title(title, fontsize=14)

  fig.tight_layout() # タイトルとラベルが被らないようにする
  # plt.pause(0.7)
  plt.close()
  fig.savefig(output_img_filepath)
  
    

if __name__ == "__main__":
  main() 
'''