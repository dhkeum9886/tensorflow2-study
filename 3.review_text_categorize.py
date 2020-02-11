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

    print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))

    # print(train_data[0])
    # print(train_labels)

    print ('before padding length', len(train_data[0]), len(train_data[1]))

    # 정수를 단어로 다시 변환하기

    # 단어와 정수 인덱스를 매핑한 딕셔너리
    word_index = imdb.get_word_index()

    # 처음 몇 개 인덱스는 사전에 정의되어 있습니다
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # 단어와 정수 인덱스를 매핑한 딕셔너리를 이용해서 정수를 단어로 치환하는 함수.
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    # 해독하기 전의 데이터. 정수만 출력됨.
    print ('before decode\r\n', train_data[0])

    # 해독된 후의 데이터. 문자열이 출력됨.
    print ('after decode\r\n', decode_review(train_data[0]))


    # 데이터 준비
    # 최대 길이를 정하고 남는 공간에 <PAD> 를 채워서 길이를 맞춤.
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    # 최대 길이를 정하고 남는 공간에 <PAD> 를 채워서 길이를 맞춤.
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    print ('after padding length', len(train_data[0]), len(train_data[1]))


    print ('after padding, before decode\r\n', train_data[0])
    print ('after padding, after decode\r\n', decode_review(train_data[0]))