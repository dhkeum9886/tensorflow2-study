#-*- coding:utf-8 -*-
# https://www.tensorflow.org/tutorials/keras/text_classification?hl=ko
# 영화 리뷰를 사용한 텍스트 분류

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)




if __name__ == '__main__':

    # IMDB 데이터셋 다운로드
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)