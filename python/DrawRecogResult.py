#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from argparse import ArgumentParser
from keras.models import load_model
from CommonFunc import (mergeFilepaths, isImage, showProcessingTime, loadSingleImage, 
                        loadImagesAndFilenames)
import numpy as np
import matplotlib.pyplot as plt

def loadImagesFromList(txt_filepath, channel_num=1):
  imgs = filenames = []
  with open(txt_filepath) as f:
    line = f.readline()
    while line:
      filepath = line.split('\n')[0]
      img = loadSingleImage(filepath, channel_num=channel_num)
      img = img.astype('float32') / 255.
      imgs.append(img)
      filenames.append(os.path.basename(filepath))
      line = f.readline()
  return np.array(imgs), filenames


def main():
  parser = ArgumentParser()
  parser.add_argument('path', type=str, 
                      help=('single test image filepath OR path of directory including test images '
                            + 'OR txt filepath that is list of filepaths of test images'))
  parser.add_argument('target_dirpath', type=str, 
                      help='path of directory including "model.hdf5"')
  parser.add_argument('--height', type=int, default=100, help='images height (default: 100)')
  parser.add_argument('--width', type=int, default=100, help='images width (default: 100)')
  parser.add_argument('--channel_num', type=int, default=1, 
                      help='number of color channels of image (default: 1)')
  args = parser.parse_args()

  start = time.time()

  # Load test images
  channel_num = 1
  ext = os.path.splitext(args.path)[1]
  if isImage(args.path):
    img = loadSingleImage(args.path, channel_num=channel_num)
    test_imgs = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    test_filenames = os.path.basename(args.path)
  elif ext == '':
    test_imgs, test_filenames = loadImagesAndFilenames(args.path, channel_num=channel_num)
  elif ext == '.txt':
    test_imgs, test_filenames = loadImagesFromList(args.path, channel_num=channel_num)
  
  # Load model
  model_filepath = mergeFilepaths(args.target_dirpath, 'lenet.hdf5')
  model = load_model(model_filepath)
  
  # Get classifying results (softmax outputs = probabilities) 
  # results.shape = (#data, #class)
  results = model.predict(test_imgs, batch_size=256, verbose=1)

  # Make output directory if it does not exist
  output_dirpath = mergeFilepaths(args.target_dirpath, 'OutputImages')
  if not os.path.isdir(output_dirpath):
    os.mkdir(output_dirpath)

  # Draw test image itself and a bar graph of classifying result
  x = [chr(i) for i in range(65, 65 + 26)]
  # x = [i for i in range(results.shape[1])]
  for img, filename, probs in zip(test_imgs, test_filenames, results):
    output_img_filepath = mergeFilepaths(output_dirpath, filename)

    # Get the class index of highest probability
    highest_prob_index = probs.argsort()[::-1][0]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4,3))
    # axes[0](left side): test image
    axes[0].imshow(img, cmap='Greys_r')
    axes[0].set_title('input image')
    # axes[1](right side): bar graph
    axes[1].bar(left=x, height=probs, width=1.0, color='b', align='center')
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
  
  processing_time = time.time() - start
  showProcessingTime(processing_time)


if __name__ == "__main__":
  main()