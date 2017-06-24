#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from argparse import ArgumentParser
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def isImage(filepath):
  img_exts = ['.png', '.jpg']
  ext = os.path.splitext(filepath)[1]
  for i, x in enumerate(img_exts):
    if ext == x:
      return True
  return False

def countFiles(dirpath):
  counter = 0
  for filename in os.listdir(dirpath):
    counter += 1
  return counter


def getImageArray(img_filepath, is_gray=True):
  img = load_img(img_filepath, grayscale=is_gray)
  array = np.array(img_to_array(img) / 255)
  return array

def mergeFilepaths(dirpath, filename):
  if dirpath[-1] == '/':
    filepath = dirpath + filename
  else:
    filepath = dirpath + '/' + filename
  return filepath

def storeFontsFromTxt(txt_filepath):
  fonts = []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      font = line.split(' ')[1]
      font = font.split('\n')[0]
      if font[-1] == '\r':
        font = font.split('\r')[0]
      fonts.append(font)
      line = f.readline()
  return fonts
    
def main():
  usage = 'Usage: python {} INPUT_PATH OUTPUT_DIRPATH [--gray] [--help]'.format(__file__)
  parser = ArgumentParser(usage)
  parser.add_argument(
    'input_path', type=str, 
    help='single image filepath (the extension must be .png or .jpg) '
          + 'OR path of directory including images '
          + 'OR list.txt (each line must be consisted of "{FONT_NAME}<space>{INDEX}" style)'
  )
  parser.add_argument(
    'output_dirpath', type=str, help='output directory path'
  )
  parser.add_argument(
    '-g', '--gray', action='store_true', default=True, help='load image(s) as gray scale'
  )
  args = parser.parse_args()

  if not os.path.exists(args.output_dirpath):
    os.makedirs(args.output_dirpath)
    print('Make {}'.format(args.output_dirpath))

  if isImage(args.input_path):
    print('[Convert "{}" to .npy]'.format(args.input_path))
    array = getImageArray(args.input_path, is_gray=args.gray)
    output_filename = os.path.splitext(args.input_path)[0] + '.npy'
    output_filepath = mergeFilepaths(args.output_dirpath, output_filename)
    np.save(output_filepath, array)

  elif os.path.isdir(args.input_path):
    print('[Convert images in "{}" to .npy]'.format(args.input_path))  
    total = 0
    for filename in os.listdir(args.input_path):
      if isImage(filename):
        total += 1
    counter = 0
    for filename in os.listdir(args.input_path):
      img_filepath = args.input_path + filename
      output_filename = os.path.splitext(filename)[0] + '.npy'
      output_filepath = mergeFilepaths(args.output_dirpath, output_filename)
      if not os.path.isfile(output_filepath):
        array = getImageArray(img_filepath, is_gray=args.gray)
        np.save(output_filepath, array)
        counter += 1
        if counter % 100 == 0:
          print('.. {0}/{1} images are converted to numpy'.format(counter, total))

  elif os.path.splitext(args.input_path)[1] == '.txt':
    print('[Convert augmented images of fonts in "{}" to .npy]'.format(args.input_path))      
    fonts = storeFontsFromTxt(args.input_path)
    for i, font in enumerate(fonts):
      img_dirpath = ('/media/snhryt/Data/Research_Master/Syn_AlphabetImages_Augmented/image/'
                     + 'WithoutTranslation/' + font + '/')
      if not os.path.isdir(img_dirpath):
        print('.. "{}" directory is not found!'.format(font))
        continue

      new_output_dirpath = args.output_dirpath + font
      if not os.path.exists(new_output_dirpath):
        os.mkdir(new_output_dirpath)
        
      if countFiles(new_output_dirpath) == 18200:
        shutil.rmtree(img_dirpath)
        print('.. {0}/{1} font({2})'.format(i+1, len(fonts), font) + 
              ' images are already converted to numpy -> delete .png directory')
      else:
        for filename in os.listdir(img_dirpath):
          img_filepath = img_dirpath + filename
          output_filepath = mergeFilepaths(new_output_dirpath, 
                                           os.path.splitext(filename)[0] + '.npy')
          if not os.path.isfile(output_filepath):
            array = getImageArray(img_filepath, is_gray=args.gray)
            np.save(output_filepath, array)
        shutil.rmtree(img_dirpath)            
        print('.. {0}/{1} font({2}) images are converted to numpy'.format(i+1, len(fonts), font))


  print '[Done!]'


if __name__ == "__main__":
  main() 
