#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def countLineNum(txt_filepath):
  '''
  Count number of lines of a text file.
  '''
  line_num = 0
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      line_num += 1
      line = f.readline()
  return line_num

def isImage(filepath):
  '''
  If the extension of file is ".png" or ".jpg" or ".bmp", return True.
  '''
  img_exts = ['.png', '.jpg', '.jpeg', '.bmp']
  ext = os.path.splitext(filepath)[1]
  for i in range(len(img_exts)):
    if ext == img_exts[i]:
      return True
  return False

def showProcessingTime(processing_time):
  '''
  Show processing time.
  '''
  if processing_time < 60 * 2:
    print('processing time: {0:.5f} s'.format(processing_time))
  elif processing_time < 60 * 60:
    print('processing time: {} m'.format(int(processing_time / 60)))
  else:
    hours = int(processing_time / (60 * 60))
    minutes = int(processing_time - (60 * 60) * hours) / 60
    print('processing time: {0} h, {1} m'.format(hours, minutes))

def storeIndexFontDict(txt_filepath):
  '''
  Return a dictionary (key:font index, value:font name) got from a text file.
  Each line of a text file must be "<font index> <font name>" style.
  '''
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

def storeSingleImage(filepath, color='gray'):
  '''
  Store a image from filepath.
  The image is read as color or grayscale or binarized (default: grayscale).
  '''
  if color == 'color':
    img = cv2.imread(filepath)
  elif color == 'gray':
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  elif color == 'bin':
    gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  else:
    raise NameError('"color" must be one of them: "color", "gray", "bin"')
  return img

def storeImagesFromDirectory(dirpath, color='gray', need_filenames=False):
  '''
  Return a list of images stored from directory.
  The images are read as color or grayscale or binarized (default: grayscale).
  This function also returns a list of filenames of the images, if you need.
  '''
  imgs = filenames = []
  if color == 'color':
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = os.path.join(dirpath, filename)
      img = cv2.imread(filepath)
      imgs.append(img)
      filenames.append(filename)
  elif color == 'gray':
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = os.path.join(dirpath, filename)
      gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      imgs.append(gray_img)
      filenames.append(filename)
  elif color == 'bin':
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = os.path.join(dirpath, filename)
      gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      imgs.append(bin_img)
      filenames.append(filename)
  else:
    raise NameError('"color" must be one of them: "color", "gray", "bin"')

  if need_filenames:
    return np.array(imgs), filenames
  else:
    return np.array(imgs)