#! /usr/bin/env python
# -*- coding: utf-8 -*-
import random
import cv2
import numpy as np

class Margin:
  left = 0
  right = 0
  top = 0
  bottom = 0
  def __init__(self, bin_img):
    # bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    height, width = bin_img.shape[:2]
    reshaped = bin_img.reshape(height * width)
    self.top = reshaped.argmin() / width
    self.bottom = reshaped[::-1].argmin() / width

    transposed = bin_img.transpose()
    reshaped_transposed = transposed.reshape(height * width)
    self.left = reshaped_transposed.argmin() / height
    self.right = reshaped_transposed[::-1].argmin() / height


def embedImage(fimg, x_tl=0, y_tl=0, angle=0.0):
  fimg_height, fimg_width = fimg.shape[:2]
  bimg_height, bimg_width = fimg_height * 2, fimg_width * 2
  bimg = np.empty((bimg_height, bimg_width), dtype=np.uint8)
  bimg[:,:] = 255

  y_start = ((bimg_width - fimg_width) / 2) + y_tl
  x_start = ((bimg_height - fimg_height) / 2) + x_tl
  if y_start >= fimg_height or x_start >= fimg_width:
    raise ValueError('x_tl and y_tl must be in ({0}, {1})'.format(-((bimg_width - fimg_width) / 2), 
                     (bimg_width - fimg_width) / 2))
  bimg[y_start:y_start + fimg_height, x_start:x_start + fimg_width] = fimg[0:fimg_height, 0:fimg_height]

  if angle != 0.0:
    affine_matrix = cv2.getRotationMatrix2D((bimg_height / 2, bimg_width / 2), angle, 1.0)
    bimg = cv2.warpAffine(bimg, affine_matrix, bimg.shape, borderValue=255)
  return bimg


def getRoi(img):
  # bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  height, width = img.shape[:2]
  reshaped_img = img.reshape(height * width)
  transposed_img = img.transpose()
  reshaped_transposed_img = transposed_img.reshape(height * width)

  top = reshaped_img.argmin() / width 
  bottom = (height - 1) - (reshaped_img[::-1].argmin() / width)
  left = reshaped_transposed_img.argmin() / height
  right = (width - 1) - (reshaped_transposed_img[::-1].argmin() / height)
  # must be list (because tuple is immutable)
  return [left, top, right - left + 1, bottom - top + 1]


def getSquareImage(roi_img):
  height, width = roi_img.shape[:2]
  if height == width:
    return roi_img
  else:
    if height > width:
      square_img = np.empty((height, height), dtype=np.uint8)
      square_img[:,:] = 255
      half_length = (height - width) / 2
      if half_length % 2 == 0:
        square_img[0:height, half_length:half_length + width] = roi_img[0:height, 0:width]
      else:
        square_img[0:height, half_length + 1:(half_length + 1) + width] = roi_img[0:height, 0:width]
    else:
      square_img = np.empty((width, width), dtype=np.uint8)
      square_img[:,:] = 255      
      half_length = (width - height) / 2
      if half_length % 2 == 0:
        square_img[half_length:half_length + height, 0:width] = roi_img[0:height, 0:width]
      else:
        square_img[half_length + 1:(half_length + 1) + height, 0:width] = roi_img[0:height, 0:width]
    return square_img
  

def storeScaleDownImages(imgs):
  output_imgs = np.empty((0, imgs[0].shape[0], imgs[0].shape[1]))
  for i,img in enumerate(imgs):
    large_img = embedImage(img)
    margin = Margin(img)
    roi = getRoi(large_img)
    roi_copy = list(roi) # 参照渡しの浅いコピーにならないように深いコピー
    for delta in [6, 12, 18, 24]:
      roi[0] -= (margin.left + delta)
      roi[1] -= (margin.top + delta)
      roi[2] += margin.left + margin.right + delta * 2
      roi[3] += margin.top + margin.bottom + delta * 2
      roi_img = large_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
      square_img = getSquareImage(roi_img)
      resized_img = cv2.resize(square_img, img.shape[:2])
      output_imgs = np.append(output_imgs, resized_img[np.newaxis,:,:], axis=0)
      roi = list(roi_copy)
  return output_imgs

def storeRotatedImages(imgs):
  output_imgs = np.empty((0, imgs[0].shape[0], imgs[0].shape[1]))
  for i, img in enumerate(imgs):
    margin = Margin(img)
    for angle in [-8.0, -4.0, 4.0, 8.0]:
      large_rotated_img = embedImage(img, x_tl=0, y_tl=0, angle=angle)
      bin_large_rotated_img = cv2.threshold(large_rotated_img, 0, 255, 
                                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      roi = getRoi(bin_large_rotated_img)
      roi[0] -= margin.left
      roi[1] -= margin.top
      roi[2] += margin.left + margin.right
      roi[3] += margin.top + margin.bottom
      roi_img = bin_large_rotated_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
      square_img = getSquareImage(roi_img)
      resized_img = cv2.resize(square_img, img.shape[:2])
      output_imgs = np.append(output_imgs, resized_img[np.newaxis,:,:], axis=0)
  return output_imgs


def storeStretchedImages(imgs):
  output_imgs = np.empty((0, imgs[0].shape[0], imgs[0].shape[1]))
  for i, img in enumerate(imgs):
    large_img = embedImage(img)
    quarter_length = (large_img.shape[0] - img.shape[0]) / 2
    roi = [quarter_length, quarter_length, img.shape[0], img.shape[0]]
    roi_copy = list(roi)
    for delta in [5, 10, 15]:
      roi[0] -= delta
      roi[1] += delta
      roi[2] += delta * 2
      roi[3] -= delta * 2
      roi_img = large_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
      resized_img = cv2.resize(roi_img, img.shape[:2])
      output_imgs = np.append(output_imgs, resized_img[np.newaxis,:,:], axis=0)
      roi = list(roi_copy)
    for delta in [5, 10, 15]:
      roi[0] += delta
      roi[1] -= delta
      roi[2] -= delta * 2
      roi[3] += delta * 2
      roi_img = large_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
      resized_img = cv2.resize(roi_img, img.shape[:2])
      output_imgs = np.append(output_imgs, resized_img[np.newaxis,:,:], axis=0)
      roi = list(roi_copy)
  return output_imgs


def storeJaggyImages(imgs):
  output_imgs = np.empty((0, imgs[0].shape[0], imgs[0].shape[1]))  
  for i, img in enumerate(imgs):
    small_img = cv2.resize(img, (30, 30))
    resized_img = cv2.resize(small_img, img.shape[:2])
    output_imgs = np.append(output_imgs, resized_img[np.newaxis,:,:], axis=0)
  return output_imgs

def storeNoisyImages(imgs):
  indices = [i for i in range(1, len(imgs))]
  random_indices = random.sample(indices, 100)
  height, width = imgs[0].shape[:2]
  output_imgs = np.empty((0, height, width))
  pixels = [i for i in range(height * width)]
  for i, index in enumerate(random_indices):
    img_copy = imgs[index].copy()
    reshaped_img = img_copy.reshape(height * width)
    noise_pixels = random.sample(pixels, 20)
    for j, pixel in enumerate(noise_pixels):
      if reshaped_img[pixel] == 0:
        reshaped_img[pixel] = 255
      else:
        reshaped_img[pixel] = 0
    noisy_img = reshaped_img.reshape(height, width)
    output_imgs = np.append(output_imgs, noisy_img[np.newaxis,:,:], axis=0)
  return output_imgs

# def storeTranslatedImages(imgs, output_imgs):

def showImage(img):
  cv2.imshow('', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def augmentImage(img):
  augmented_imgs = np.empty((0, img.shape[0], img.shape[1]))
  # input (= no change)
  augmented_imgs = np.append(augmented_imgs, img[np.newaxis,:,:], axis=0)
  # scale down
  scale_down_imgs = storeScaleDownImages(augmented_imgs)
  augmented_imgs = np.append(augmented_imgs, scale_down_imgs, axis=0)
  del scale_down_imgs
  # rotation
  rotated_imgs = storeRotatedImages(augmented_imgs)
  augmented_imgs = np.append(augmented_imgs, rotated_imgs, axis=0)
  del rotated_imgs
  # stretch
  stretched_imgs = storeStretchedImages(augmented_imgs)
  augmented_imgs = np.append(augmented_imgs, stretched_imgs, axis=0)
  del stretched_imgs
  # generate jagginess
  jaggy_imgs = storeJaggyImages(augmented_imgs)
  augmented_imgs = np.append(augmented_imgs, jaggy_imgs, axis=0)
  del jaggy_imgs
  # generate papper noise
  noisy_imgs = storeNoisyImages(augmented_imgs)
  augmented_imgs = np.append(augmented_imgs, noisy_imgs, axis=0)
  del noisy_imgs

  return augmented_imgs
  # print augmented_imgs.shape
  # output_dirpath = '/home/snhryt/Desktop/augmentation/'
  # for i, img in enumerate(augmented_imgs):
  #   output_filepath = output_dirpath + '{0:03d}'.format(i) + '.png'
  #   cv2.imwrite(output_filepath, img)


# def main():
#   img_filepath = ('/media/snhryt/Data/Research_Master/Syn_AlphabetImages/font/' + 
#                   '00-Starmap-Truetype/capA_00-Starmap-Truetype.png')
#   img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
#   bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#   augmentImage(bin_img)


# main()