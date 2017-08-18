#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import collections
import time
from argparse import ArgumentParser
from keras.models import load_model
from CommonFunc import (isImage, showProcessingTime, storeSingleImageShaped4Keras, 
                        storeImagesShaped4Keras, convertColorImageChannelOrder)
from segmentation import Segmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BookInfo:
    """
    BookInfo consists of 3 strings: filename, book title, and book genre.
    """
    def __init__(self):
        self.filename = ''
        self.title = ''
        self.genre = ''
    

def storeBookInfo(csv_filepath, book_filenames):
    """
    Return a list of BookInfo objects.
    First, read a .csv file which consists of 3 columns: <Filename>,<Title>,<Genre>, 
    and then store book_filenames[i]'s information.
    So, the length of list that this function returns is the same as len(book_filenames).
    """
    csv = pd.read_csv(csv_filepath)
    book_info = []
    for filename in book_filenames:
        df = csv[csv['Filename'].isin([filename])]
        if len(df) == 0:
            continue
        x = BookInfo()
        x.filename = filename
        x.title = df['Title'].values[0]
        x.genre = df['Genre'].values[0]
        book_info.append(x)
    return book_info


def storeClassifiedAlphabets(results):
    """
    Return a list of alphabets which have the highest classification probability.
    """
    classified_alphabets = []
    for i, probs in enumerate(results):
        highest_prob_index = probs.argsort()[::-1][0]
        classified_alphabets[i] = chr(highest_prob_index + 65)
    return classified_alphabets


def storeBeginAndEndIndicesOfSequentialNumbers(sequential_nums):
    """
    Store begin_indices and end_indices of a sequential numbers list.
    E.g. where [1, 1, 2, 3, 3, 3] is given, this function returns [0, 2, 3] and [1, 2, 5].
    """
    begin_indices = [0]
    for i in range(1, len(sequential_nums)):
        if sequential_nums[i - 1] != sequential_nums[i]:
            begin_indices.append(i)
            
    end_indices = [None] * len(begin_indices)
    for i in range(len(begin_indices) - 1):
        end_indices[i] = begin_indices[i + 1] - 1
    if begin_indices[-1] == len(sequential_nums) - 1:
        end_indices[len(begin_indices) - 1] = begin_indices[-1]
    else:
        end_indices[len(begin_indices) - 1] = len(sequential_nums) - 1
    return begin_indices, end_indices


def drawResult(input_img, text_region_img, char_imgs, classfied_word, classified_fonts,
               output_filepath):
    """
    Draw classification result. Output image is like below: 
                     col 0                col 1        ,...,        col n 
        row 0:    [input image]    [text region image]
        row 1: [character image 0] [character image 1]       [charcter image n]
    """
    ncols = 2 if (len(char_imgs) < 2) else len(char_imgs)
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    # Top-left: input image
    axes[0][0].imshow(convertColorImageChannelOrder(input_img))
    axes[0][0].set_title('input')
    # Top-right: text region image
    axes[0][1].imshow(text_region_img, cmap='Greys_r')
    axes[0][1].set_title('segmented and binarized text region')
    # Bottom
    for i, img in enumerate(char_imgs):
        title = ('classified -> alphabet:{}, '.format(classfied_word[i]) 
                 + 'font(index):{}'.format(classified_fonts[i]))
        axes[1][i].imshow(img, cmap='Greys_r')
        axes[1][i].set_title(title)
    fig.tight_layout()
    plt.close()
    fig.savefig(output_filepath)


def main(args):
    """
    Main function.
    """
    start = time.time()

    # Load test images
    if isImage(args.path):
        img = storeSingleImageShaped4Keras(args.path, color='color')
        filename = os.path.basename(args.path)
        test_imgs = img[np.newaxis, :, :, :]
        test_filenames = [filename]
    elif os.path.isdir(args.path):
        test_imgs, test_filenames = storeImagesShaped4Keras(args.path, color='color', 
                                                            need_filename=True)

    # Load book information
    book_info = storeBookInfo(args.csv_filepath, test_filenames)

    # Load model for character classification
    model1 = load_model(args.char_classification_model_filepath)
    model2 = load_model(args.font_classification_model_dirpath)

    # Make output directory if it does not exist
    if not os.path.isdir(args.output_dirpath):
        os.mkdir(args.output_dirpath)
    
    # Step1: Text region segmentation by MSER-like method
    # Step2: Character segmentation by simple method
    # Step3: Character classification by a CNN 
    # Step4: Font (shape) classification by a CNN selected from 26 CNNs for classifying font of A-Z
    #        CNNs are selected based on Step3's result
    # Step5: Decide the font of a image
    book_fonts = [None] * len(test_imgs)
    for i, (img, filename) in enumerate(zip(test_imgs, test_filenames)):
        # Step1 & Step2
        x = Segmentation(img)
        x.detectAndSegmentChars()

        if len(x.char_imgs) < 3:
            book_fonts[i] = ''
            continue

        # Step3
        normalized_char_imgs = x.char_imgs.astype('float32') / 255.
        char_classification_results = model1.predict(normalized_char_imgs, batch_size=8)
        classified_alphabets = storeClassifiedAlphabets(char_classification_results)

        # Step4
        classified_font_indices = [None] * len(x.char_imgs)
        for j, (char_img, alphabet) in enumerate(zip(normalized_char_imgs, classified_alphabets)):
            # model2_filepath = os.path.join(args.font_classification_model_dirpath, 
            #                                alphabet + '.hdf5')
            # model2 = load_model(model2_filepath)
            char_img = char_img[np.newaxis, :, :, :]
            font_classification_result = model2.predict(char_img)
            classified_font_indices[j] = font_classification_result[0].argsort()[::-1][0]

        # Step5
        count_dict = collections.Counter(classified_font_indices)
        book_fonts[i] = str(count_dict.most_common()[0][0])

        # Draw result
        height, width, channel_num = x.char_imgs[0].shape
        begin_indices, end_indices = storeBeginAndEndIndicesOfSequentialNumbers(x.char_img_indices)
        for j, (begin_index, end_index) in zip(begin_indices, end_indices):
            counter = 0
            tmp_char_imgs = np.empty((end_index - begin_index + 1, height, width, channel_num))
            tmp_classified_word = ''
            tmp_classified_font_indices = [None] * (end_index - begin_index + 1)
            for index in range(begin_index, end_index + 1):
                tmp_char_imgs[counter] = x.char_imgs[index]
                tmp_classified_word += classified_alphabets[index]
                tmp_classified_font_indices[counter] = classified_font_indices[index]
                counter += 1
            stem, ext = os.path.splitext(filename)
            output_img_filepath = os.path.join(args.output_dirpath, (stem + '_' + str(j) + ext))
            drawResult(img, x.text_region_imgs[j], tmp_char_imgs, tmp_classified_word, 
                       tmp_classified_font_indices, output_img_filepath)

    output_txt_filepath = os.path.join(args.output_dirpath, 'FilenameFontList.txt')
    with open(output_txt_filepath, mode='w') as f:
        for filename, font in zip(test_filenames, book_fonts):
            if font != '':
                f.write(filename + ' ' + font + '\n')

    processing_time = time.time() - start
    showProcessingTime(processing_time)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'path', 
        type=str, 
        help='test image filepath OR path of directory including test images'
    )
    parser.add_argument(
        'csv_filepath', 
        type=str, 
        help='filepath of book information csv'
    )
    parser.add_argument(
        'char_classification_model_filepath', 
        type=str,
        help='filepath of trained CNN model for character classification'
    )
    parser.add_argument(
        'font_classification_model_dirpath', 
        type=str,
        help='path of directory including trained CNN model for font (shape) classification'
    )
    parser.add_argument(
        'output_dirpath', 
        type=str, 
        help='output directory path'
    )
    args = parser.parse_args()
    main(args)