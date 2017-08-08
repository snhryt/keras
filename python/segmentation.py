#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from CommonFunc import loadSingleImage
import numpy as np
import matplotlib.pyplot as plt

def getRectImage(img, rect):
  img_height, img_width = img.shape
  x, y, width, height = rect
  x_min = x - 1 if (x > 0) else x 
  y_min = y - 1 if (y > 0) else y
  x_max = x_min + width + 1 if (x_min + width <= img_width) else x_min + width
  y_max = y_min + height + 1 if (y_min + height <= img_height) else y_min + height
  segmented_img = img[y_min:y_max, x_min:x_max]
  return segmented_img

def isBlackBackground(bin_img, judge_thresh=0.5):
  height, width = bin_img.shape
  white_pixel_num = np.count_nonzero(bin_img)
  if ( (bin_img[0][0] == 0 and bin_img[height - 1][width - 1] == 0) 
        or white_pixel_num < (height * width * judge_thresh) ):
    return True
  else:
    return False

def getSquareImage(bin_img):
  height, width = bin_img.shape
  if height == width:
    return bin_img
  else:
    if height > width:
      square_img = np.empty((height, height), dtype=np.uint8)
      square_img[:,:] = 255
      half_length = int((height - width) / 2)
      if half_length % 2 == 0:
        square_img[0:height, half_length:half_length + width] = bin_img[0:height, 0:width]
      else:
        square_img[0:height, half_length + 1:(half_length + 1) + width] = bin_img[0:height, 0:width]
    else:
      square_img = np.empty((width, width), dtype=np.uint8)
      square_img[:,:] = 255    
      half_length = int((width - height) / 2)
      if half_length % 2 == 0:
        square_img[half_length:half_length + height, 0:width] = bin_img[0:height, 0:width]
      else:
        square_img[half_length + 1:(half_length + 1) + height, 0:width] = bin_img[0:height, 0:width]
    return square_img

def isNoisy(bin_img, value_change_thresh=30):
  height, width = bin_img.shape
  vertical_value_change_num = horizontal_value_change_num = 0
  for x in range(width):
    value_change_counter = 0
    first_pixel_value = bin_img[0][x]
    for y in range(height):
      if abs(bin_img[y][x] - first_pixel_value) == 255:
        value_change_counter += 1
        first_pixel_value = bin_img[y][x]
    vertical_value_change_num += value_change_counter
  vertical_value_change_mean = int(vertical_value_change_num / height)
  if vertical_value_change_mean > value_change_thresh:
    return True
  
  for y in range(height):
    value_change_counter = 0
    first_pixel_value = bin_img[y][0]
    for x in range(width):
      if abs(bin_img[y][x] - first_pixel_value) == 255:
        value_change_counter += 1
        first_pixel_value = bin_img[y][x]
    horizontal_value_change_num += value_change_counter
  horizontal_value_change_mean = int(horizontal_value_change_num / width)
  if horizontal_value_change_mean > value_change_thresh:
    return True
  
  return False

class Segmentation:
  def __init__(self, img, resized_width=100, resized_height=100):
    self.img = img
    self.height, self.width = self.img.shape[:2]
    self.resized_width = resized_width
    self.resized_height = resized_height
    self.text_region_rects = []
    self.text_region_imgs = []
    self.char_imgs = []
    self.char_img_indices = []
  
  def getSameTextRegionIndex(self, rect, diff_thresh=20):
    x_min1 = rect[0]
    y_min1 = rect[1]
    x_max1 = x_min1 + rect[2]
    y_max1 = y_min1 + rect[3]
    if len(self.text_region_rects) > 0:
      for i, text_region_rect in enumerate(self.text_region_rects):
        x_min2 = text_region_rect[0]
        y_min2 = text_region_rect[1]
        x_max2 = x_min2 + text_region_rect[2]
        y_max2 = y_min2 + text_region_rect[3]
        x_min_diff = abs(x_min1 - x_min2)
        y_min_diff = abs(y_min1 - y_min2)
        x_max_diff = abs(x_max1 - x_max2)
        y_max_diff = abs(y_max1 - y_max2)
        if ( (x_min_diff < diff_thresh and y_min_diff < diff_thresh) 
              or (x_max_diff < diff_thresh and y_max_diff < diff_thresh) ):
          return i
    return -1

  def segmentTextRegions(self, height_thresh=30):
    # Noise reduction using filter that is able to preserve the edges of character 
    filtered_img = cv2.edgePreservingFilter(self.img)

    # Store images of 5 channels (RGB + lightness + gradient magnitude)
    channel_imgs = cv2.text.computeNMChannels(filtered_img, _mode=cv2.text.ERFILTER_NM_RGBLGrad)
    for i in range(len(channel_imgs) - 1):
      channel_imgs.append(255 - channel_imgs[i])

    for channel_img in channel_imgs:
      classifier1_path = '../trained_classifierNM1.xml'
      classifier2_path = '../trained_classifierNM2.xml'
      # Select Extremal Regions(ERs) based on two ERFilters proposed by Neuman and Matas
      erfilter1 = cv2.text.createERFilterNM1(cv2.text.loadClassifierNM1(classifier1_path), 
                           40, 0.00015, 0.13, 0.2, True, 0.1)
      erfilter2 = cv2.text.createERFilterNM2(cv2.text.loadClassifierNM2(classifier2_path), 0.5)
      text_regions = cv2.text.detectRegions(channel_img, erfilter1, erfilter2)
      rects = cv2.text.erGrouping(self.img, channel_img, [r.tolist() for r in text_regions])
      
      bin_img = cv2.threshold(channel_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      for rect in rects:
        x, y, width, height = rect
        text_region_img = getRectImage(bin_img, rect)
        if height < height_thresh or isBlackBackground(text_region_img) or isNoisy(text_region_img):
          continue
        same_region_index = self.getSameTextRegionIndex(rect)
        if same_region_index >= 0:
          if width >= self.text_region_rects[same_region_index][2]:
            self.text_region_rects[same_region_index] = rect
            self.text_region_imgs[same_region_index] = text_region_img
        else:
          self.text_region_rects.append(rect)
          self.text_region_imgs.append(text_region_img)

  def drawTextRegions(self):
    for rect in self.text_region_rects:
      x_min = rect[0]
      y_min = rect[1]
      x_max = x_min + rect[2]
      y_max = y_min + rect[3]
      cv2.rectangle(img=self.img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), 
                    thickness=5)
  
  def isGoodCharImage(self, bin_img, white_pixel_ratio=0.95, width_thresh=5):
    height, width = bin_img.shape
    if np.sum(bin_img) <= (255 * height * width * white_pixel_ratio):
      return True
    else:
      return False
  
  def segmentChars(self, labeling=True):
    if labeling:
      for i, img in enumerate(self.text_region_imgs):
        height, width = img.shape
        labeled_img = cv2.connectedComponents(255 - img)[1]
  #       # Noise reduction by erotion and dilation
  #       img_copy = img.copy()
  #       img_copy = 255 - img_copy
  #       kernel = np.ones((3, 3), np.uint8)
  #       eroded_img = cv2.erode(img_copy, kernel, iterations=1)
  #       dilated_img = cv2.dilate(eroded_img, kernel, iterations=2)
  #       masked_img = cv2.bitwise_and(img_copy, img_copy, mask=dilated_img)
  #       # Labeling
  #       labeled_img = cv2.connectedComponents(masked_img)[1]

        x_img_dict = {}
        for label in range(1, labeled_img.max() + 1):
          char_img = np.empty(img.shape)
          char_img[:] = 255
          char_img[labeled_img == label] = 0

          # 'white_col_x' is a list of x whose columns' pixel value is 255 (i.e., white)
          white_col_x = np.where(np.sum(char_img, axis=0) == height * 255)[0]
          if len(white_col_x) == 0:
            continue
          x_min = x_max = 0
          for j in range(len(white_col_x) - 1):
            if (white_col_x[j] + 1) != white_col_x[j + 1]:
              x_min = white_col_x[j]
              x_max = white_col_x[j + 1]
              break
          segmented_img = char_img[0:height, x_min:x_max+1]

          square_img = getSquareImage(segmented_img)
          resized_img = cv2.resize(square_img, (self.resized_width, self.resized_height))
          if self.isGoodCharImage(resized_img):
            x_img_dict[x_min] = resized_img

        for x, char_img in sorted(x_img_dict.items()):
          self.char_imgs.append(char_img)
          self.char_img_indices.append(i)
          
    else: # Simple cutting algorithm
      for i, img in enumerate(self.text_region_imgs):
        height, width = img.shape
        
        # If (almost) all pixel of a column is white, x of that column becomes cutting candidates
        ratio = 0.98
        white_col_x = np.where(np.sum(img, axis=0) >= height * 255 * ratio)[0]
        if len(white_col_x) == 0:
          continue
        
        # In below example, we only need 8 coordinates(x) of '|'(cutting line)
        # [    |h| |o| |g|    |e|   ]
        cutting_x = []
        index = 0
        if white_col_x[index] != 0: # In case like [h| |o| |g|    |e|   ]
          cutting_x.append(0)
        cutting_x.append(white_col_x[index])
        index += 1
        while index < len(white_col_x) - 1:
          if (white_col_x[index] - 1 == white_col_x[index - 1] 
              and white_col_x[index] + 1 == white_col_x[index + 1]):
            index += 1
          else:
            cutting_x.append(white_col_x[index])
            index += 1
        cutting_x.append(white_col_x[index])
        if white_col_x[-1] != width - 1: # In case like [    |h| |o| |g|    |e] 
          cutting_x.append(width - 1)
        
        for j in range(len(cutting_x) - 1):
          segmented_img = img[ 0:height, cutting_x[j]:cutting_x[j + 1] ]
          if self.isGoodCharImage(segmented_img):
            square_img = getSquareImage(segmented_img)
            resized_img = cv2.resize(square_img, (self.resized_width, self.resized_height))
            self.char_imgs.append(resized_img)
            self.char_img_indices.append(i)
    self.char_imgs = np.array(self.char_imgs)
    
  def detectAndSegmentChars(self):
    self.segmentTextRegions()
    self.drawTextRegions()
    self.segmentChars(labeling=True)