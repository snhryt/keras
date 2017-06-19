#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from argparse import ArgumentParser
from keras.preprocessing.image import load_img, img_to_array

def isImage(filename):
  ext = os.path.splitext(filename)[1]
  if ext == '.png' or ext == '.jpg':
    return True
  else:
    return False

def getImageArray(img_filepath, is_gray=True):
  img = load_img(img_filepath, grayscale=is_gray)
  array = np.array(img_to_array(img) / 255)
  return array

def saveNpyArray(array, output_dirpath, filename):
  output_filename = os.path.splitext(filename)[0] + '.npy'
  if filename[-1] == '/':
    output_filepath = output_dirpath + output_filename    
  else:
    output_filepath = output_dirpath + '/' + output_filename
  np.save(output_filepath, array)

def readTxt(txt_filepath):
  fonts = []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      font = line.split(' ')[1]
      font = font.split('\n')[0]
      fonts.append(font)
      line = f.readline()
  return fonts
    
def main():
  usage = 'Usage: python {} INPUT_PATH OUTPUT_DIRPATH [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument('input_path', type=str, 
                      help='Single image filepath (the extension must be .png or .jpg) OR '
                           +'Path of directory including images OR '
                           +'List.txt (each line must consist "{FONT NAME}<space>{INDEX}" style)')
  parser.add_argument('output_dirpath', type=str, help='Output directory path')
  parser.add_argument('-g', '--gray', action='store_true', help='load image(s) as gray scale')
  args = parser.parse_args()

  if not os.path.exists(args.output_dirpath):
    os.makedirs(args.output_dirpath)
    print('Make %s', args.output_dirpath)

  if isImage(args.input_path):
    print('[Convert %s to .npy]' % args.input_path)
    array = getImageArray(args.input_path, is_gray=args.gray)
    saveNpyArray(array, args.output_dirpath, os.path.basename(args.input_path))
  elif os.path.isdir(args.input_path):
    print('[Convert images in %s .npy]' % args.input_path)  
    total = 0
    for filename in os.listdir(args.input_path):
      if isImage(filename):
        total += 1

    counter = 0
    for filename in os.listdir(args.input_path):
      img_filepath = args.input_path + filename
      if isImage(img_filepath):
        array = getImageArray(img_filepath, is_gray=args.gray)
        saveNpyArray(array, args.output_dirpath, filename)
        counter += 1
        if counter % 100 == 0:
          print('.. %d/%d images are converted to numpy' % (counter, total))
  elif os.path.splitext(args.input_path)[1] == '.txt':
    print('[Convert augmented images of fonts in %s .npy]' % args.input_path)      
    fonts = readTxt(args.input_path)
    for i in range(len(fonts)):
      dirpath = args.output_dirpath + fonts[i]
      if not os.path.exists(dirpath):
        os.mkdir(dirpath)

      img_dirpath = '/media/snhryt/Data/Research_Master/Syn_AlphabetImages_Augmented/image'
      img_dirpath = os.path.join(img_dirpath, 'WithoutTranslation/' + fonts[i])
      for filename in os.listdir(img_dirpath):
        img_filepath = img_dirpath + '/' + filename
        if isImage(img_filepath):
          array = getImageArray(img_filepath, is_gray=args.gray)
          saveNpyArray(array, dirpath, filename)
      print('.. %d/%d font images are converted to numpy' % (i + 1, len(fonts)))
      

  print 'Done!'


if __name__ == "__main__":
  main() 
