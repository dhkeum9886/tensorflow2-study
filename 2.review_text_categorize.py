#-*- coding:utf-8 -*-
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub?hl=ko
# 케라스와 텐서플로 허브를 사용한 영화 리뷰 텍스트 분류하기

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds


# print("버전: ", tfds.__file__)

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")



# Unrecognized instruction format: NamedSplit('train')(tfds.percent[0:60])


if __name__ == '__main__':

    # IMDB 데이터셋 다운로드
    train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
    # # 아래 코드에서 에러 발생함.
    # # 원인 파악하기 귀찮아서..안함.
    # (train_data, validation_data), test_data = tfds.load(
    #     name="imdb_reviews",
    #     split=(train_validation_split, tfds.Split.TEST),
    #     as_supervised=True)
