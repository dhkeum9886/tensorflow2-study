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

    # 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기입니다(10,000개의 단어)
    vocab_size = 10000

    model = keras.Sequential()

    # Embedding 층.
    # 이 층은 정수로 인코딩된 단어를 입력 받고 각 단어 인덱스에 해당하는 임베딩 벡터를 찾습니다.
    # 이 벡터는 모델이 훈련되면서 학습됩니다.
    # 이 벡터는 출력 배열에 새로운 차원으로 추가됩니다.
    # 최종 차원은 (batch, sequence, embedding)이 됩니다.
    model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))

    # GlobalAveragePooling1D 층
    # sequence 차원에 대해 평균을 계산하여 각 샘플에 대해 고정된 길이의 출력 벡터를 반환합니다.
    # 이는 길이가 다른 입력을 다루는 가장 간단한 방법입니다.
    model.add(keras.layers.GlobalAveragePooling1D())

    # 이 고정 길이의 출력 벡터는 16개의 은닉 유닛을 가진 완전 연결(fully-connected) 층(Dense)을 거칩니다.
    model.add(keras.layers.Dense(16, activation='relu'))

    # 하나의 출력 노드(node)를 가진 완전 연결 층입니다.
    # sigmoid 활성화 함수를 사용하여 0과 1 사이의 실수를 출력합니다.
    # 이 값은 확률 또는 신뢰도를 나타냅니다.
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()