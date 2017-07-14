# !/usr/bin/env python
# *-coding utf-8 -*
import time
import cv2
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def storeImageImread(filepath):
  img = cv2.imread(filepath)
  #img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  return img

def storeImageLoadimg(filepath):
  img = image.load_img(filepath)
  #img = image.load_img(filepath, grayscale=True)
  img_array = image.img_to_array(img)
  return img_array

def appendListThenConvertNparray(img, N=100):
  img_list = []
  for i in range(N):
    img_list.append(img)
  img_list = np.array(img_list)
  return img_list

def appendNparray(img, N=100):
  img_list = np.empty((0, img.shape[0], img.shape[1], img.shape[2]))
  for i in range(N):
    img_list = np.append(img_list, img[np.newaxis,:,:], axis=0)
  return img_list  

def main():
  img_filepath = '/media/snhryt/Data/Research_Master/Real_Images/Input/0-top100/006202406X.jpg'

  start = time.time()
  img1 = storeImageImread(img_filepath)
  processing_time = time.time() - start
  print('cv2.imread -> process time: {0}[s], img1.shape = {1}'.format(processing_time, img1.shape))

  start = time.time()
  img2 = storeImageLoadimg(img_filepath)
  processing_time = time.time() - start
  print('image.load_img -> process time: {0}[s], img2.shape = {1}'.format(processing_time, img2.shape))

  load_img_num = 200

  ltime_list, ntime_list, x = [], [], []
  for i in range(1, load_img_num + 1):
    x.append(i)

    start = time.time()
    img_array = appendListThenConvertNparray(img1, i)
    end = time.time()
    ltime_list.append(end - start)

    start = time.time()
    img_array = appendNparray(img1, i)
    end = time.time()
    ntime_list.append(end - start)
  '''
  output_img_filepath = '/home/snhryt/Desktop/SpeedTest.png'
  plt.plot(x, ltime_list, label='list append')
  plt.plot(x, ntime_list, label='nparray append')
  plt.xlabel('#appending')
  plt.ylabel('processing time[s]')
  plt.xlim(1, load_img_num)
  title = 'image.shape = {}'.format(img1.shape)
  plt.title(title)
  plt.legend()
  plt.pause(5.0)
  plt.close()
  plt.savefig(output_img_filepath)
  '''

  output_img_filepath1 = '/home/snhryt/Desktop/list.png'
  output_img_filepath2 = '/home/snhryt/Desktop/nparray.png'
  plt.plot(x, ltime_list, label='list append')
  plt.xlabel('#appending')
  plt.ylabel('processing time[s]')
  plt.xlim(1, load_img_num)
  title = 'image.shape = {}'.format(img1.shape)
  plt.title(title)
  plt.savefig(output_img_filepath1)
  plt.close()


  plt.plot(x, ntime_list, label='nparray append')
  plt.xlabel('#appending')
  plt.ylabel('processing time[s]')
  plt.xlim(1, load_img_num)
  title = 'image.shape = {}'.format(img1.shape)
  plt.title(title)
  plt.savefig(output_img_filepath2)
  plt.close()
  


if __name__ == '__main__':
  main()