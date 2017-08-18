#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def countLineNum(txt_filepath):
    '''
    Return number of lines of a text file.
    '''
    line_num = 0
    with open(txt_filepath) as f:
        line = f.readline()
        while line:
            line_num += 1
            line = f.readline()
    return line_num

def isImage(filepath):
    '''
    Return True if the extension of file is ".png" or ".jp(e)g" or ".bmp", otherwise return False.
    '''
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    ext = os.path.splitext(filepath)[1]
    if ext in img_exts:
        return True
    else:
        return False

def showProcessingTime(processing_time):
    '''
    Show processing time.
    '''
    if processing_time < 60 * 2:
        print('processing time: {:.5f} s'.format(processing_time))
    elif processing_time < 60 * 60:
        print('processing time: {} m'.format(int(processing_time / 60)))
    else:
        hours = int(processing_time / (60 * 60))
        minutes = int(processing_time - (60 * 60) * hours) / 60
        print('processing time: {0} h, {1} m'.format(hours, minutes))

def storeIndexFontDict(txt_filepath):
    '''
    Return a dictionary (key:font index, value:font name) stored from a text file.
    Each line of the text file must be "<font index> <font name>" style.
    '''
    index_font_dict = {}
    with open(txt_filepath) as f:
        line = f.readline()
        while line:
            index = int(line.split(' ')[0])
            font = line.split(' ')[1]
            font = font.split('\n')[0]
            index_font_dict[index] = font
            line = f.readline()
    return index_font_dict

def storeSingleImageShaped4Keras(filepath, color='gray'):
    '''
    Return a numpy array of a image stored from filepath.
    You can select the number of color channels of image from 1("gray" or "bin") or 3("color") 
    (default: "gray").
    When color="gray", the output array is reshaped as (height, width, 1) for model fitting 
    on Keras having TensorFlow backend.
    '''
    if not color in {'color', 'gray', 'bin'}:
        raise NameError('color must be "color", "gray", or "bin"')

    if color == 'color':
        img = cv2.imread(filepath)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if color == 'bin':
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        img = img[:, :, np.newaxis]
    # img = img.astype('float32') / 255.
    return img
        
def storeImagesShaped4Keras(path, color='gray', need_filename=False):
    '''
    Return a numpy array of images stored from a directory or a list of image filepaths.
    You can select the number of color channels of images from 1("gray" or "bin") or 3("color")
    (default: "gray").
    When color="gray", the output array is reshaped as (image_num, height, width, 1) for model 
    fitting on Keras having TensorFlow backend.
    And when need_filename=True, this function also returns a list of filenames of the images.
    '''
    if not color in {'color', 'gray', 'bin'}:
        raise NameError('color must be "color", "gray", or "bin"')

    imgs = []
    filenames = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if not isImage(filename):
                continue
            filepath = os.path.join(path, filename)
            img = storeSingleImageShaped4Keras(filepath, color=color)
            imgs.append(img)
            filenames.append(filename)
    elif os.path.splitext(path)[1]:
        with open(path) as f:
            line = f.readline()
            while line:
                filepath = line.split('\n')[0]
                img = storeSingleImageShaped4Keras(filepath, color=color)
                imgs.append(img)
                filenames.append(os.path.basename(filepath))
                line = f.readline()
    else:
        raise NameError('The extension of path must be "" or ".txt"')

    if need_filename:
        return np.array(imgs), filenames
    else:
        return np.array(imgs)

def convertColorImageChannelOrder(color_img):
    """
    Return a color image whose channel order is converted to RGB from BGR for imshow of matplotlib.
    """
    converted_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return converted_img