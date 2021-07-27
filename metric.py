# coding=utf-8
import tensorflow as tf
from config import Config as Config
import numpy as np
import time


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        val_predict = self.model.predict(self.validation_data[0], batch_size=512)
        val_target = self.validation_data[1]

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

        return
