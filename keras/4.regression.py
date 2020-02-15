#-*- coding:utf-8 -*-
# https://www.tensorflow.org/tutorials/keras/regression?hl=ko
# 자동차 연비 예측하기: 회귀

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)



if __name__ == '__main__':

    # 데이터셋 다운로드.
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print (dataset_path)


    # 판다스를 사용하여 데이터를 읽습니다.

                    # MPG = 미국 자동차 연비 , 마일 단위
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()

    print('로우 데이터')
    print(dataset.tail())

    # 데이터 누락이 있는지 확인.
    print(dataset.isna().sum())

    # 데이터 누락이 있는 행을 삭제.
    dataset = dataset.dropna()

    print('누락을 정리한 데이터')
    print(dataset.tail())


    # Origin 컬럼은 값이 각 1,2,3 으로 입력되어 있음.
    # "Origin" 열은 수치형이 아니고 범주형이므로 원-핫 인코딩(one-hot encoding)으로 변환하겠습니다:
    origin = dataset.pop('Origin')

    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0


    print('Origin 가공 데이터')
    print(dataset.tail())


    # 데이터셋을 훈련 세트와 테스트 세트로 분할
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)            # 테스트 세트는 모델을 최종적으로 평가할 때 사용


    print('\r\ntrain_dataset')
    print(train_dataset.tail())

    print('\r\ntest_dataset')
    print(test_dataset.tail())

    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")                   # MPG 레이블 분리하기 # MPG 를 예측하기 위해 모델을 훈련.
    train_stats = train_stats.transpose()
    print('train_stats')
    print(train_stats)

    # 특성과 레이블 분리하기                      # MPG 를 예측하기 위해 모델을 훈련.
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')


    # 데이터 정규화 , normalization
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # print('normed_train_data')
    # print(normed_train_data.tail())
    #
    # print('normed_test_data')
    # print(normed_test_data.tail())


    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()
    print(model.summary())

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)


    # 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    def plot_history(history):

        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    #  1,000번의 에포크(epoch) 동안 훈련
    EPOCHS = 1000

    # 검증 점수가 향상되지 않으면 자동으로 훈련을 멈추도록
    # patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    # plot_history(history)


    # 훈련 과정을 보기 위함.
    #
    # # 훈련 정확도와 검증 정확도는 history 객체에 기록
    # history = model.fit(
    #     normed_train_data, train_labels,
    #     epochs=EPOCHS, validation_split=0.2, verbose=0,
    #     callbacks=[PrintDot()])
    #
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # hist.tail()
    #
    # # history 객체에 저장된 통계치를 사용해 모델의 훈련 과정을 시각화
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print(hist.tail())
    #
    #
    plot_history(history)



    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))


    # 예측
    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()













