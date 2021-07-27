# coding=utf-8

from name_checker import BadNameModel
from config import Config
import os
import numpy as np
import tensorflow as tf
import time

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print(memory_gpu, str(np.argmax(memory_gpu)))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

class Node:
    def __init__(self, label="", parent=None, children=None, num=0):
        if children is None:
            children = []
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def train():
    my_config = Config()
    my_model = BadNameModel(my_config)
    my_model.train()


if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time()
    print(end - start)
