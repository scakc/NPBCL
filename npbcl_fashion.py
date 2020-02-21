import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
try:
    os.mkdir('./saves')
except:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import math
from data_generators import PermutedMnistGenerator, SplitMnistGenerator, NotMnistGenerator, FashionMnistGenerator
from ibpbcl import IBP_BCL
import torch


torch.manual_seed(19)
np.random.seed(1)


hidden_size = [200]
alpha = [30]
no_epochs = 5
no_tasks = 5
coreset_size = 0#200
coreset_method = "kcen"
single_head = False
batch_size = 256

# data_gen = PermutedMnistGenerator(no_tasks)
# data_gen = SplitMnistGenerator()
# data_gen = NotMnistGenerator()
data_gen = FashionMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head, grow = False)

accs, _ = model.batch_train(batch_size)
np.save('./saves/fashionmnist_accuracies.npy', accs)