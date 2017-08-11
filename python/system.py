#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import segmentation
from argparse import ArgumentParser
from keras.models import load_model
from CommonFunc import mergeFilepaths, isImage, loadSingleImage, loadImagesAndFilenames
import numpy as np

class BookInfo:
    def __init__(self):
        self.filename = self.title = self.genre = ''
    
def storeBookInfo(csv_filepath, book_filenames):
    '''
    Read a csv file which is consisted of 3 columns: <Filename>,<Title>,<Genre>, 
    and then store book_filenames[i]'s information.
    So, this function returns a book information list whose len() is same as len(book_filenames).
    '''
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

def storeStartAndEndIndicesOfSequentialNumbers(sequential_nums):
    '''
    Store start_indices and end_indices of a sequential numbers list, 
    e.g. where [1, 1, 2, 3, 3, 3] is given, this function returns [0, 2, 3] and [1, 2, 5].
    '''
    start_indices = [0]
    for i in range(1, len(sequential_nums)):
        if sequential_nums[i - 1] != sequential_nums[i]:
            start_indices.append(i)
            
    end_indices = [None] * len(start_indices)
    for i in range(len(start_indices) - 1):
        end_indices[i] = start_indices[i + 1] - 1
    if start_indices[-1] == len(sequential_nums) - 1:
        end_indices[len(start_indices) - 1] = start_indices[-1]
    else:
        end_indices[len(start_indices) - 1] = len(sequential_nums) - 1
    return start_indices, end_indices

def drawResult(text_region_img, char_imgs, classfied_word, classified_fonts):
    hoge = 1

def decideFontOfWord(classified_fonts):
    font = 'hoge'
    return font

def decideFontOfImage(voted_font):
    font = 'hoge'
    return font


def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=str, 
                        help='test image filepath OR path of directory including test images')
    parser.add_argument('csv_filepath', type=str, help='filepath of book information csv')
    parser.add_argument('char_classification_model_dirpath', type=str,
                        help='filepath of trained CNN model for character classification')
    parser.add_argument('font_classification_model_dirpath', type=str,
                        help=('path of directory including trained CNN model for font (shape) '
                              + 'classification'))
    parser.add_argument('output_dirpath', type=str, help='output directory path')
    args = parser.parse_args()

    # Load test images
    if isImage(args.path):
        img = loadSingleImage(args.path, channel_num=1)
        filename = os.path.basename(args.path)
        test_imgs = test_filenames = []
        test_imgs.append(img)
        test_filenames.append(filename)
    elif os.path.isdir(args.path):
        test_imgs, test_filenames = loadImagesAndFilenames(args.path, channel_num=1)

    # Load book information
    book_info = storeBookInfo(args.csv_filepath, test_filenames)

    # Load model for character classification
    model1 = load_model(args.char_classification_model_filepath)

    # Make output directory if it does not exist
    if not os.path.isdir(args.output_dirpath):
        os.mkdir(args.output_dirpath)
    
    output_txt_filepath = mergeFilepaths(args.output_dirpath, 'FilenameFontList.txt')
    f = open(output_txt_filepath, mode='w')
    
    # Step1: text region segmentation by MSER-like method
    # Step2: character segmentation by simple method
    # Step3: character classification by a CNN 
    # Step4: font (shape) classification by a CNN selected from 26 CNNs based on Step3's result
    # Step5: decide the font of a image
    for img, filename in zip(test_imgs, test_filenames):
        # Step1 & Step2
        x = Segmentation(img)
        x.detectAndSegmentChars()

        start_indices, end_indices = storeStartAndEndIndicesOfSequentialNumbers(x.char_img_indices)
        for start, end in zip(start_indices, end_indices):
            if 

        # Step3
        char_classification_results = model1.predict(x.char_imgs, batch_size=8)
        classified_char_indices = getHighProbClassIndices(char_classification_results)
        classified_word = ''
        for index in classified_char_indices:
            classified_word += chr(index + 65)
        char_num = len(classified_word)

        # Step4
        for i, (char_img, alphabet) in enumerate(zip(x.char_imgs, classified_word)):
            model2_filepath = os.path.join(args.font_classification_model_dirpath, (alphabet+'.h5'))
            model2 = load_model(model2_filepath)
            reshaped = char_img[np.newaxis, :, :]
            font_classification_result = model2.predict(reshaped, batch_size=8)



        # Step1
        voted_fonts = [None] * char_num
        for i, text_region_img in enumerate(text_region_imgs):
            # Step2
            char_imgs = segmentChars(text_region_img)
            char_num = len(char_imgs)
            # Step3
            char_classification_results = model1.predict(char_imgs, batch_size=8)
            classified_char_indices = getHighProbClassIndices(char_classification_results)
            classified_word = ''
            for i, index in enumerate(classified_char_indices):
                classified_word += chr(index + 65)
            # Step4
            classified_fonts = [None] * char_num
            for char_img, alphabet in zip(char_imgs, classified_word):
                model2_filepath = mergeFilepaths(args.font_classification_model_dirpath, (alphabet + '.h5'))
                model2 = load_model(model2_filepath)
                reshaped = char_img.reshape(1, char_img.shape[0], char_img.shape[1], char_img.shape[2])
                font_classification_result = model2.predict(reshaped, batch_size=8)
                classified_fonts[i] = getHighProbClassIndices(font_classification_result, top_N=3)

            drawResult(text_region_img, char_imgs, classified_word, classified_fonts)
            voted_fonts[i] = decideFontOfWord(classified_fonts)
        # Step5
        font = decideFontOfImage(voted_fonts)
        f.write((filename + ' ' + font))
    f.close()


if __name__ == '__main__':
    main()