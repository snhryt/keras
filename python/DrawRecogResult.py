#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import cv2
from argparse import ArgumentParser
from keras.utils import np_utils
from keras.models import load_model
from CommonFunc import (isImage, countLineNum, showProcessingTime, storeSingleImageShaped4Keras, 
                        storeImagesShaped4Keras)
import numpy as np
import matplotlib.pyplot as plt

COLOR = 'bin'

def storeImagesLabelsFilenames(txt_filepath):
    """
    Return a numpy array of images and 2 lists: class labels list and filenames of the images list,
    which are stored from a text file.
    Each line of the text file must be "<full filepath of a image> <class label>" style.
    """
    print('\n[Loading "{}"]'.format(txt_filepath))
    total = countLineNum(txt_filepath)
    
    imgs = [None] * total
    labels = [None] * total
    filenames = [None] * total
    counter = 0
    with open(txt_filepath) as f:
        line = f.readline()
        while line:
            filepath = line.split(' ')[0]
            imgs[counter] = storeSingleImageShaped4Keras(filepath, color=COLOR)
            label = line.split(' ')[1]
            label = label.split('\n')[0]
            labels[counter] = int(label)
            filenames[counter] = os.path.basename(filepath)
            counter += 1
            line = f.readline()
    imgs = np.array(imgs)
    return imgs, labels, filenames


def storeMedoidFonts(list_filepath):
    """
    Return the names of center (called as medoid) font in clusters. 
    """
    medoid_fonts = []
    with open(list_filepath) as f:
        line = f.readline()
        while line:
            cluster_index = line.split(' ')[0]
            medoid_font = line.split(' ')[1]
            medoid_font = medoid_font.split('\n')[0]
            medoid_fonts.append(medoid_font)
            line = f.readline()
    return medoid_fonts


def storeMedoidImage(dirpath, alphabet, font_name):
    """
    Return a numpy array of a image of character which is the medoid of cluster.
    """
    font_img_dirpath = os.path.join(dirpath, font_name)
    for filename in os.listdir(font_img_dirpath):
        if ('cap' + alphabet) in filename:
            filepath = os.path.join(font_img_dirpath, filename)
            medoid_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            break
    return medoid_img


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'path', 
        type=str, 
        help=('filepath of single test image OR path of directory including test images ' 
              + 'OR txt filepath that is list of test image filepaths')
    )
    parser.add_argument(
        'target_dirpath', 
        type=str, 
        help='path of directory including "model.hdf5"'
    )
    parser.add_argument(
        '--classification_target', 
        type=str, 
        default='character', 
        help='classification target ("character" or "font")'
    )
    parser.add_argument(
        '--medoid_fonts_list_filepath', 
        type=str,
        default=None, 
        help='.txt filepath of a list of medoid fonts'
    )
    parser.add_argument(
        '--char_imgs_dirpath', 
        type=str, 
        default=None, 
        help=('path of directory whose sub-directories are named same as font name ' 
              + 'and iclude character images')
    )
    parser.add_argument(
        '--alphabet',
        type=str,
        default='A',
        help='classified alphabet (default: "A")'
    )
    args = parser.parse_args()

    start = time.time()

    if not args.classification_target in {'character', 'font'}:
        raise NameError('classification_target must be "character" or "font"')

    # Load test image(s)
    if isImage(args.path):
        img = storeSingleImageShaped4Keras(args.path, color=COLOR)
        test_imgs = img[np.newaxis, :, :, :]
        test_filenames = os.path.basename(args.path)
    else:
        if args.classification_target == 'character':
            if os.path.splitext(args.path)[1] == '.txt':
                test_imgs, test_labels, test_filenames = storeImagesLabelsFilenames(args.path)
        elif args.classification_target == 'font':
            test_imgs, test_filenames = storeImagesShaped4Keras(args.path, color=COLOR, 
                                                                need_filename=True) 
    test_imgs = test_imgs.astype('float32') / 255.

    # Load model
    model_filepath = os.path.join(args.target_dirpath, 'model.hdf5')
    model = load_model(model_filepath)
    
    # Get classifying results (softmax outputs = probabilities) 
    # results.shape = (#data, #class)
    results = model.predict(test_imgs, batch_size=256, verbose=1)

    # Make output directory if it does not exist
    output_dirpath = os.path.join(args.target_dirpath, 'OutputImages')
    if not os.path.isdir(output_dirpath):
        os.mkdir(output_dirpath)

    # Draw test image itself and a bar graph of classifying result
    x = [i for i in range(results.shape[1])]
    if args.classification_target == 'character':
        x_label = [chr(i) for i in range(65, 65 + 26)]
        corretct_num = 0
        for img, label, filename, probs in zip(test_imgs, test_labels, test_filenames, results):
            img = img.reshape(img.shape[0], img.shape[1])
            output_img_filepath = os.path.join(output_dirpath, filename)

            # Get the class index of highest probability
            highest_prob_index = probs.argsort()[::-1][0]
            if highest_prob_index == label:
                corretct_num += 1

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            # Left: test image
            axes[0].imshow(img, cmap='Greys_r')
            axes[0].set_title('input image')
            # Right: bar graph
            axes[1].bar(left=x, height=probs, width=1.0, tick_label=x_label, align='center')
            axes[1].set_xlim(0, results.shape[1] - 1)
            axes[1].set_ylim(0.0, 1.0)
            ylabel = 'probability (top:' + '{:.3f}'.format(probs[highest_prob_index]) + ')'
            axes[1].set_ylabel(ylabel)
            axes[1].grid(True)
            fig.tight_layout()
            plt.close()
            fig.savefig(output_img_filepath)
        print('\n#correctly classified: {0}/{1} images'.format(corretct_num, len(test_imgs)))

        # Make a file whose filename has information about how many images are correctly classified
        log_filename = ('{0} of {1} '.format(corretct_num, len(test_imgs))
                        + 'images are correctly classified')
        log_filepath = os.path.join(args.target_dirpath, log_filename)
        f = open(log_filepath, mode='w')
        f.close()

    elif args.medoid_fonts_list_filepath and args.char_imgs_dirpath:
        medoid_fonts = storeMedoidFonts(args.medoid_fonts_list_filepath)
        for img, filename, probs in zip(test_imgs, test_filenames, results):
            img = img.reshape(img.shape[0], img.shape[1])
            highest_prob_index = probs.argsort()[::-1][0]
            medoid_img = storeMedoidImage(args.char_imgs_dirpath, args.alphabet, 
                                          medoid_fonts[highest_prob_index])
            output_img_filepath = os.path.join(output_dirpath, filename)

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
            # Left: test image
            axes[0].imshow(img, cmap='Greys_r')
            axes[0].set_title('input')
            # Middle: cluster medoid image
            axes[1].imshow(medoid_img, cmap='Greys_r')
            axes[1].set_title('medoid of top prob. cluster', fontsize=12)
            # Right: bar graph
            axes[2].bar(left=x, height=probs, width=1.0, align='center')
            axes[2].set_xlim(0, results.shape[1] - 1)
            axes[2].set_ylim(0.0, 1.0)
            xlabel = 'cluster index (top:{})'.format(highest_prob_index)
            ylabel = 'probability (top:{:.3f})'.format(probs[highest_prob_index])
            axes[2].set_xlabel(xlabel)
            axes[2].set_ylabel(ylabel)
            axes[2].grid(True)
            fig.tight_layout()
            plt.close()
            fig.savefig(output_img_filepath)

    processing_time = time.time() - start
    showProcessingTime(processing_time)


if __name__ == "__main__":
    main()