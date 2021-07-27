# coding=utf-8

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Masking, Bidirectional, Permute, Reshape, \
    Multiply, RepeatVector, Lambda, Attention, Embedding, Flatten
from config import Config


class MyModel:
    def __init__(self, my_config):
        self.model = None
        self.config = my_config
        self.build()
        print('init model success')

    def build(self):
        method_name_token = Input((5,), dtype=tf.int32, name='method_name_input')
        method_body_token = Input((100,), dtype=tf.int32, name='method_body_input')
        interface_token = Input((10,), dtype=tf.int32, name='interface_input')
        struct_token = Input((10,), dtype=tf.int32, name='struct_input')
        return_token = Input((10,), dtype=tf.int32, name='return_input')

        embedding = Embedding(25000, self.config.EMBEDDING_SIZE, mask_zero=True, name='method_name_embedding')

        method_name_embedding = embedding(method_name_token)
        method_body_embedding = embedding(method_body_token)
        interface_embedding = embedding(interface_token)
        struct_embedding = embedding(struct_token)
        return_embedding = embedding(return_token)

        method_name_hidden = LSTM(units=self.config.HIDDEN_SIZE, name='method_name_hidden', return_sequences=True)(
            method_name_embedding)
        method_name_hidden = LSTM(units=self.config.HIDDEN_SIZE, name='method_name_double_hidden')(method_name_hidden)

        method_body_hidden = LSTM(units=self.config.HIDDEN_SIZE, name='context_hidden', return_sequences=True)(
            method_body_embedding)

        method_name_hidden = Reshape((1, self.config.EMBEDDING_SIZE))(method_name_hidden)
        method_body_hidden = Attention(use_scale=True, trainable=self.config.IS_TRAIN)(
            [method_name_hidden, method_body_hidden])

        method_name_hidden = Reshape((self.config.EMBEDDING_SIZE,))(method_name_hidden)
        method_body_hidden = Reshape((self.config.EMBEDDING_SIZE,))(method_body_hidden)

        interface_hidden = Flatten()(interface_embedding)
        interface_hidden = Dense(units=self.config.HIDDEN_SIZE, name='interface_hidden')(interface_hidden)
        interface_hidden = Concatenate()([method_name_hidden, interface_hidden])
        interface_hidden = Dense(units=self.config.HIDDEN_SIZE, name='interface_hidden2')(interface_hidden)

        struct_hidden = Flatten()(struct_embedding)
        struct_hidden = Dense(units=self.config.HIDDEN_SIZE, name='struct_hidden')(struct_hidden)
        struct_hidden = Concatenate()([method_name_hidden, struct_hidden])
        struct_hidden = Dense(units=self.config.HIDDEN_SIZE, name='struct_hidden2')(struct_hidden)

        return_hidden = Flatten()(return_embedding)
        return_hidden = Dense(units=self.config.HIDDEN_SIZE, name='return_hidden')(return_hidden)
        return_hidden = Concatenate()([method_name_hidden, return_hidden])
        return_hidden = Dense(units=self.config.HIDDEN_SIZE, name='return_hidden2')(return_hidden)

        result_hidden = Concatenate()(
            [method_name_hidden, method_body_hidden, interface_hidden, struct_hidden, return_hidden])

        result_hidden = Dense(128, activation='relu')(result_hidden)
        result_hidden = Dense(32, activation='relu')(result_hidden)
        output = Dense(1, activation='sigmoid')(result_hidden)

        inputs = [method_name_token, method_body_token, interface_token, struct_token, return_token]
        self.model = Model(inputs=inputs, outputs=output)
        # self.model.summary()


if __name__ == '__main__':
    model = Model(Config())
    model.model.summary()
    print(model.config.IS_TRAIN)
