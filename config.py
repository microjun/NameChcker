# coding=utf-8

class Config:

    def __init__(self) -> None:
        super().__init__()

        self.DATA_ROOT_PATH = '/home1/likejun/Data/20210528/'
        self.RESULT_DATA_ROOT_PATH = '/home1/likejun/BadNameDetection20210401/result/'

        self.TRAIN_DATA_SET_PATH = self.DATA_ROOT_PATH
        self.TEST_DATA_SET_PATH = '/home1/likejun/BadNameDetection20210401/data/46_noreal/'
        self.MODEL_SAVE_PATH = self.DATA_ROOT_PATH + 'save/'
        self.FINAL_SAVE_PATH = '/home1/likejun/BadNameDetection20210401/final_data/'

        self.MAX_METHOD_NAME_LENGTH = 5
        self.MAX_CONTEXT_LENGTH = 100

        self.BATCH_SIZE = 128
        self.WORD_VECTOR_EMBEDDING_LENGTH = 128
        self.HIDDEN_SIZE = 128
        self.EMBEDDING_SIZE = 128
        self.DROPOUT_RATE = 0.2

        self.LEARNING_RATE = 0.001
        self.EPOCH = 60

        self.IS_TRAIN = True
        self.VOCAB_SIZE = 90000

