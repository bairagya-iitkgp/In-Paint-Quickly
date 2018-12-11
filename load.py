import os
import numpy as np
from config import *

def load(dir_=args.data_path):
    x_train = np.load(os.path.join(dir_, 'x_train4.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test4.npy'))
    return x_train, x_test


if __name__ == '__main__':
    x_train, x_test = load()
    print(x_train.shape)
    print(x_test.shape)

