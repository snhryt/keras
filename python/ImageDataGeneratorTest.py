#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
from keras.preprocessing import image

def drawImages(datagen, img_array, output_img_num=100, batch_size=1):
  output_dirpath = "/home/snhryt/Desktop/output/"
  if os.path.isdir(output_dirpath):
    shutil.rmtree(output_dirpath)
    os.mkdir(output_dirpath)
  else:
    os.mkdir(output_dirpath)

  g = datagen.flow(
    img_array, batch_size=batch_size, save_to_dir=output_dirpath, save_prefix='img', 
    save_format='jpg'
  )
  for i in range(output_img_num):
    g.next()

def main():
  img_filepath = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages_Augmented/numpy/' + \
                 'WithoutTranslation/00-Starmap-Truetype/capA_00-Starmap-Truetype_312.npy'
  array = np.load(img_filepath)
  # img = image.array_to_img(array)
  # img.show()
  array = np.expand_dims(array, axis=0)
  datagen = image.ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.78, 
    zoom_range=0.3
  )
  drawImages(datagen, array)



if __name__ == '__main__':
  main()