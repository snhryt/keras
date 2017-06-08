from __future__ import print_function

import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.utils import np_utils

import numpy
from pandas import DataFrame
from pandas import read_csv
data_train = read_csv("/media/snhryt/Data/Research_Master/keras/mnist_sample/short_prac_train.csv")
data_test = read_csv("/media/snhryt/Data/Research_Master/keras/mnist_sample/short_prac_test.csv")

batch_size = 100
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

# the data, shuffled and split between tran and test sets
x_train = numpy.asarray([data_train.ix[i][1:] for i in range(len(data_train))])
x_test = numpy.asarray([data_test.ix[i][1:] for i in range(len(data_test))])
y_train = data_train['label']
y_test = data_test['label']

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode="valid",
                        input_shape=(img_rows, img_cols, 1)))
model.add(Activation("relu"))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adadelta",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])