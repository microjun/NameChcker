# coding=utf-8

import numpy as np
from config import Config as Config
import random


def load_per_data(data_list):
    method_name_list = []
    interface_list = []
    method_body_list = []
    struct_list = []
    return_list = []
    label_list = []

    for method in data_list:
        old_name, new_name, method_body, struct_token, return_token = \
            method[4].split(' '), method[3].split(' '), method[7].split(' '), method[9].split(' '), method[
                8].split(' ')
        interface = method[5].split(' ') + method[6].split(' ')

        method_name_list.append(new_name)
        method_name_list.append(old_name)
        label_list.append(1)
        label_list.append(0)
        for i in range(2):
            interface_list.append(interface)
            method_body_list.append(method_body)
            struct_list.append(struct_token)
            return_list.append(return_token)

    return [label_list, method_name_list, interface_list, method_body_list, struct_list, return_list]


class DataLoader:

    def __init__(self, my_config) -> None:
        super().__init__()
        self.config = my_config
        self.vocab_index = self.get_vocab_embedding()
        self.unk = len(self.vocab_index) + 2

    def read_data(self, file_name):
        method_list = []
        with open(self.config.DATA_ROOT_PATH + file_name, 'r', encoding='utf8') as f:
            for line in f:
                method = line.strip().split(',')
                method_list.append(method)
        return method_list

    def load_data(self):
        random.seed(1997)
        train_list = []
        test_list = []
        method_list = self.read_data('train_data.csv')

        for i in range(len(method_list)):
            method = method_list[i]
            flag = random.randint(0, 9)
            if flag == 0:
                test_list.append(method)
            else:
                train_list.append(method)

        train_data = load_per_data(train_list)
        test_data = load_per_data(test_list)
        train_data, train_label = self.load_vocab_embedding_data(train_data)
        test_data, test_label = self.load_vocab_embedding_data(test_data)
        return train_data, train_label, test_data, test_label

    def load_vocab_embedding_data(self, embedding_data):
        label_list, method_name_list, interface_list, method_body_list, struct_list, return_list = embedding_data

        method_name_embedding = self.load_per_embedding_data(method_name_list, 5)
        interface_embedding = self.load_per_embedding_data(interface_list, 5)
        method_body_embedding = self.load_per_embedding_data(method_body_list, 150)
        struct_embedding = self.load_per_embedding_data(struct_list, 5)
        return_embedding = self.load_per_embedding_data(return_list, 5)

        return [method_name_embedding, method_body_embedding, interface_embedding, struct_embedding,
                return_embedding], np.asarray(label_list)

    def load_per_embedding_data(self, data_list, max_length):
        embedding_list = np.zeros((len(data_list), max_length), dtype=int)

        for i in range(len(data_list)):
            for j in range(min(max_length, len(data_list[i]))):
                index = self.vocab_index.get(data_list[i][j])
                embedding_list[i][j] = index if index is not None else self.unk

        return embedding_list

    def get_vocab_embedding(self, file_name='vocab.txt'):
        print('DATA VOCAB PATH: ' + self.config.DATA_ROOT_PATH + file_name)
        vocab_index = {}

        with open(self.config.DATA_ROOT_PATH + file_name, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split(' ')
                vocab_index[str(values[0])] = int(values[1])

        return vocab_index


if __name__ == '__main__':
    config = Config()
    data = DataLoader(config)
    data.load_data()
