#-*- coding:utf-8 -*-
# https://www.tensorflow.org/tutorials/keras/classification?hl=ko
# 첫 번째 신경망 훈련하기: 기초적인 분류 문제

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
import random

print('tf.__version__', tf.__version__)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    # if predicted_label == true_label:
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == '__main__':
    print('data loading')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 데이터 크기 확인.
    # print('train_images', train_images.shape)
    # print('train_labels', train_labels.shape)
    # print('test_images', test_images.shape)
    # print('test_labels', test_labels.shape)

    # 분류하기 위한 인덱스 정의
    # train_labels[n] 의 값으로 매칭.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # train_images[0] 이미지 보기.
    print('train_images[0]', train_images[0])
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    # 픽셀 값의 범위가 0~255 라는 것을 알 수 있음.

    # 픽셀값의 범위를 0~1로 수정.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # # train_images[0] 이미지 보기.
    # plt.figure()
    # plt.imshow(train_images[1])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    # # 픽셀 값의 범위가 0~1 로 수정됨.

    # # 처음 25개의 이미지와 클래스 이름 출력.
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    # 모델 구성.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), # 2차원 배열(28 x 28 픽셀)의 이미지 포맷을 28 * 28 = 784 픽셀의 1차원 배열로 변환

        # 픽셀을 펼친 후에는 두 개의 tf.keras.layers.Dense 층이 연속되어 연결
        keras.layers.Dense(128, activation='relu'),  # 첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가집니다
        keras.layers.Dense(10, activation='softmax') # 두 번째 (마지막) 층은 10개의 노드의 소프트맥스(softmax) 층
    ])

    # 모델 컴파일.
    model.compile(optimizer='adam', # 옵티마이저(Optimizer)-데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다.
                  loss='sparse_categorical_crossentropy', # 손실 함수(Loss function)-훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다.
                  metrics=['accuracy']) # 지표(Metrics)-훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.

    # 모델 훈련
    model.fit(train_images, train_labels, epochs=5) # epochs 만큼 반복?

    # 정확도 평가.
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\n테스트 정확도:', test_acc)

    # 예측만들기
    predictions = model.predict(test_images)

    # 가장 높은 신뢰도를 가진 레이블 확인.
    # print ('0 index ')
    # print (predictions[0])
    # print (np.argmax(predictions[0]))
    # print ('1 index ')
    # print (predictions[1])
    # print (np.argmax(predictions[1]))

    # # 잘못 예측된 레이블은 빨간색으로 표현.
    # # i 인덱스 번호의 이미지와 가장 높은 신뢰도 레이블 확인.
    # i = 1
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image(i, predictions, test_labels, test_images)
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions, test_labels)
    # plt.show()

    # # 여러 이미지의 예측.
    # num_rows = 5
    # num_cols = 3
    # num_images = num_rows * num_cols
    # plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    # for i in range(num_images):
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    #     plot_image(i, predictions, test_labels, test_images)
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    #     plot_value_array(i, predictions, test_labels)
    # plt.show()

    # 랜덤 넘버 생성. 해당 이미지의 예측.
    randomindex = random.randint(1, 10)
    print('random index : ', randomindex)

    img = test_images[randomindex]
    img = (np.expand_dims(img, 0))
    predictions_single = model.predict(img)
    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    print ('예측 : ', np.argmax(predictions_single[0]), '정답 : ', test_labels[randomindex])