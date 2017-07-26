#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

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

def showProcessingTime(processing_time):
  if processing_time < 60 * 2:
    print('processing time: {0:.5f} s'.format(processing_time))
  elif processing_time < 60 * 60:
    print('processing time: {} m'.format(int(processing_time / 60)))
  else:
    hours = int(processing_time / (60 * 60))
    minutes = int(processing_time - (60 * 60) * hours) / 60
    print('processing time: {0} h, {1} m'.format(hours, minutes))

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

def loadSingleImage(filepath, channel_num=1):
  if channel_num == 1:
    gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    img = img.reshape(img.shape[0], img.shape[1], 1)
  else:
    img = cv2.imread(filepath)
  # img = img.reshape(1, img.shape[0], img.shape[1], channel_num)   
  return img

def loadImages(dirpath, channel_num=1):
  imgs = []
  if channel_num == 1:
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = mergeFilepaths(dirpath, filename)
      gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      img = img.reshape(img.shape[0], img.shape[1], 1)
      imgs.append(img)
  else:
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = mergeFilepaths(dirpath, filename)
      img = cv2.imread(filepath)
      imgs.append(img)
  return np.array(imgs)

def loadImagesAndFilenames(dirpath, channel_num=1):
  imgs, filenames = [], []
  if channel_num == 1:
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = mergeFilepaths(dirpath, filename)
      gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      img = img.reshape(img.shape[0], img.shape[1], 1)
      imgs.append(img)
      filenames.append(filename)
  else:
    for filename in os.listdir(dirpath):
      if not isImage(filename):
        continue
      filepath = mergeFilepaths(dirpath, filename)
      img = cv2.imread(filepath)
      imgs.append(img)
      filenames.append(filename)
  return np.array(imgs), filenames

