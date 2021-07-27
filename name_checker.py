# coding=utf-8

import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from metric import Metrics
from embedding import DataLoader
from model import MyModel


class BadNameModel:
    def __init__(self, my_config):
        self.config = my_config
        self.data_loader = DataLoader(self.config)
        self.model = MyModel(self.config).model
        self.model.summary()
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            metrics=["accuracy"]
        )
        print('init model success')

    def load_model(self, file_path):
        self.model.load_weights(file_path)

    def train(self):
        print('data init ------')
        data_loader = DataLoader(self.config)
        print('data init end ------')

        check_pointer = ModelCheckpoint(
            os.path.join(self.config.MODEL_SAVE_PATH, 'model_' + '_{epoch:03d}.h5'), verbose=1)
        tensor_board = TensorBoard(
            log_dir="./log/" + "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) + "_" + str(self.config.LEARNING_RATE),
            update_freq='epoch')

        train_data, train_label, test_data, test_label = data_loader.load_data()

        self.model.fit(
            x=train_data,
            y=train_label,
            test_data=(test_data, test_label),
            epochs=self.config.EPOCH,
            shuffle=True,
            batch_size=self.config.BATCH_SIZE,
            callbacks=[check_pointer]
        )

        self.predict(test_data, test_label)

    def predict(self, test_data, test_label):
        val_predict = self.model.predict(test_data, batch_size=512)
        val_target = test_label

        predict_result_list = [1 if i[0] > 0.5 else 0 for i in val_predict]
        test_label_list = val_target.tolist()

        predict_consistent_num = 0
        predict_true_consistent_num = 0
        recall_consistent_num = 0
        recall_true_consistent_num = 0

        for i in range(len(test_label_list)):
            if predict_result_list[i] == 1:
                predict_consistent_num += 1
                if predict_result_list[i] == test_label_list[i]:
                    predict_true_consistent_num += 1
            if test_label_list[i] == 1:
                recall_consistent_num += 1
                if predict_result_list[i] == test_label_list[i]:
                    recall_true_consistent_num += 1

        predict_inconsistent_num = 0
        predict_true_inconsistent_num = 0
        recall_inconsistent_num = 0
        recall_true_inconsistent_num = 0
        for i in range(len(test_label_list)):
            if predict_result_list[i] == 0:
                predict_inconsistent_num += 1
                if predict_result_list[i] == test_label_list[i]:
                    predict_true_inconsistent_num += 1
            if test_label_list[i] == 0:
                recall_inconsistent_num += 1
                if predict_result_list[i] == test_label_list[i]:
                    recall_true_inconsistent_num += 1

        if predict_inconsistent_num == 0:
            predict_inconsistent_num = 1
        if predict_consistent_num == 0:
            predict_consistent_num = 1

        in_precision = predict_true_inconsistent_num / predict_inconsistent_num
        in_recall = recall_true_inconsistent_num / recall_inconsistent_num
        precision = predict_true_consistent_num / predict_consistent_num
        recall = recall_true_consistent_num / recall_consistent_num
        print(
            " — val_in_f1: %f — val_in_precision: %f — val_in_recall: %f - val_f1: %f — val_precision: %f — val_recall: %f" % (
                2 * in_precision * in_recall / (in_precision + in_recall), in_precision, in_recall,
                2 * precision * recall / (precision + recall), precision, recall))
        print()


if __name__ == '__main__':
    model = BadNameModel()
    model.model.summary()
