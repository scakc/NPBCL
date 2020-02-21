### Downloading MNIST
from __future__ import print_function
import os    
import numpy as np
from urllib3 import request
import gzip
import pickle
import os.path
from os import path
import matplotlib.pyplot as plt
from urllib import request
import os
import sys
import tarfile
from scipy import ndimage
from PIL import Image
import re
from sklearn.model_selection import train_test_split as tts
from keras.datasets import fashion_mnist

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist[name[0]] = tmp.reshape(-1,1,28,28).astype(np.float32) / 255
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    if(not path.exists("mnist.pkl")):
        print("File does not exists downloading Mnist Data")
        init()
    else:
        print("File exists, Loading...")
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

### Downloading notMNIST
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

# url = 'https://www.kaggle.com/lubaroli/notmnist/download'

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = request.urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders


def get_data(path):
    all_images_as_array=[]
    for filename in sorted(os.listdir(path)):
        if re.match(r'.ipynb',filename):
            continue
        try:
            img=Image.open(path + filename)
            np_array = np.asarray(img)
            l,b = np_array.shape
            np_array = np_array.reshape([l*b])
            all_images_as_array.append(np_array)
        except:
            continue
    return np.array(all_images_as_array)





num_classes = 10
def run_all():
    cpath = os.getcwd()
    
    try:
        os.chdir(os.getcwd() + '/datasets')
    except:
        os.mkdir('datasets')
        os.chdir(os.getcwd() + '/datasets')

        print('Downloading the MNIST dataset...')
        init()

    
    print('\nDownloading the notMNIST dataset')
    filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    folders = maybe_extract(filename)
    names = os.listdir("notMNIST_small")
    tasks = []
    count = 0
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    for name in sorted(names):
        print("processing / " + name)
        if re.match(r'.ipynb',name):
            continue
        path = "notMNIST_small/" + name
        imgs = get_data(path + "/")
        labels = np.ones(imgs.shape[0])*count
        X_train, X_test, y_train, y_test = tts(imgs, labels, test_size=0.2, random_state=42)
        train_x.append(X_train)
        train_y.append(y_train)
        test_x.append(X_test)
        test_y.append(y_test)
        count += 1
    tasks = (np.concatenate(train_x),np.concatenate(train_y),np.concatenate(test_x),np.concatenate(test_y))
    pickle.dump(tasks, open("notmnist.pkl", "wb"))
    
    
    print('\nDownloading the fashionMNIST dataset')
    ### Downloading fashionMNIST
    (train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
    X = np.concatenate([train_X, test_X], axis = 0)
    Y = np.concatenate([train_Y, test_Y], axis = 0)
    train_X,test_X,train_Y,test_Y = tts(X, Y, test_size=0.28571, random_state=2, stratify = Y)
    dataset = (train_X, train_Y, test_X, test_Y)
    pickle.dump(dataset, open('fashionMnist.pkl', 'wb'))
    
    os.chdir(cpath)