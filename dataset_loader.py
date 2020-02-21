import os
import numpy as np
import math
from copy import deepcopy
import gzip
import pickle
import datasets
from sklearn.model_selection import train_test_split


def load(dataset):
    if(not os.path.isdir('./datasets')):
        datasets.run_all()
    if(dataset == 'mnist'):
        with open("./datasets/mnist.pkl",'rb') as f:
            data = pickle.load(f)
        return data["training_images"], data["training_labels"], data["test_images"], data["test_labels"]
    elif(dataset == 'notmnist'):
        with open("./datasets/notmnist.pkl",'rb') as f:
            data = pickle.load(f)
        return data
    elif(dataset == 'fashionmnist'):
        with open("./datasets/fashionMnist.pkl",'rb') as f:
            data = pickle.load(f)
        return data