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

def loadImages(path, need_filename=False, height=100, width=100, gray=True):
  channel_num = 1 if gray else 3

  if need_filename:
    imgs = np.empty((0, height, width, channel_num))
    filenames = []
    if os.path.isdir(path):
      for filename in os.listdir(path):
        if not isImage(filename):
          continue
        img_filepath = mergeFilepaths(path, filename)
        if gray:
          gray_img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
          img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
          img = img.reshape(height, width, channel_num)
        else:
          img = cv2.imread(img_filepath)
        imgs = np.append(imgs, img[np.newaxis,:,:,:], axis=0)
        filenames.append(filename)
      return imgs, filenames
    else:
      if gray:
        gray_img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        img = img.reshape(height, width, channel_num)
      else:
        img = cv2.imread(img_filepath)
      img = img.reshape(1, height, width, channel_num)
      filename = os.path.basename(path)
      return img, filename

  else:
    if os.path.isdir(path):
      imgs = np.empty((0, height, width, channel_num))
      for filename in os.path.listdir(path):
        if not isImage(filename):
          continue
        img_filepath = mergeFilepaths(path, filename)
        if gray:
          gray_img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
          img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
          img = img.reshape(height, width, channel_num)
        else:
          img = cv2.imread(img_filepath)
        imgs = np.append(imgs, img[np.newaxis,:,:,:], axis=0)
      return imgs
    else:
      if gray:
        gray_img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        img = img.reshape(height, width, channel_num)
      else:
        img = cv2.imread(img_filepath)
      return img
