import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from ibpbnn_vae import IBP_BAE
import copy as cpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributions as tod
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import gzip
import pickle
from visualisation import plot_images
from sklearn.manifold import TSNE as tsne
from data_generators import PermutedMnistGenerator, SplitMnistGenerator, NotMnistGenerator, OneMnistGenerator, OneNotMnistGenerator
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import clear_output as clr

# dataset_ = 'mnist'
# dataset_ = 'notmnist'
dataset_ = str(input())
paths = {
    'mnist' : 'saves',
    'notmnist' : 'saves'
}
no_tasks = 10

def show_images(images):
    images = torchvision.utils.make_grid(images, nrow=10)
    show_image(images.permute(1,2,0))

def show_image(img):
    plt.imshow(img)
#     plt.show()


def get_model(get_till = 1, dict_path = './cache/01'):
    mf_weights, mf_variances = None, None
    prev_masks, alpha, beta = None, None, None
    for i in range(get_till):
        model = IBP_BAE(in_dim, hidden_size, out_dim, 784, max_tasks, prev_means=mf_weights, 
                        prev_log_variances=mf_variances, prev_masks = prev_masks, 
                        alpha=alpha, beta = beta, single_head=True)
        mf_weights, mf_variances = model.get_weights()
        prev_masks, alpha, beta = model.get_IBP()
    
    model.load_state_dict(torch.load(dict_path + '/model_last_' + str(get_till-1)))
    return model

if(dataset_ == 'mnist'):
    data_gen = OneMnistGenerator()
elif(dataset_ == 'notmnist'):
    data_gen = OneNotMnistGenerator()
    
in_dim, out_dim = data_gen.get_dims()
max_tasks = data_gen.max_iter

hidden_size = [500, 500, 100]
alpha = [140.0, 140.0, 40.0, 140.0, 140.0]
no_epochs = 50
coreset_size = 0#50
coreset_method = "rand"
single_head = True
batch_size = 256

x_testsets, y_testsets = [], []
## Training the model sequentially.
for task_id in range(max_tasks):
    ## Loading training and test data for current task
    _, _, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)

print('Loading saved Model.')
for i in range(no_tasks):
    task = i+1
    tmp = np.zeros([10, 784])
    model = get_model(get_till = task, dict_path = './' + paths[dataset_])
    
print('plotting TSNE')
zs = []
ys = []
for i in range(no_tasks): 
    input = torch.tensor(x_testsets[i]).float()
    code = model.encode(input, task_id = i, no_samples=1, const_mask=True, temp = model.min_temp)[0].mean(0)
    zs.append(code.detach().numpy())
    ys.append(torch.ones((code.shape[0]))*i)

Z = np.concatenate(zs, axis = 0)
Y = np.concatenate(ys)
means_emp = np.concatenate([np.mean(d, 0).reshape([1,-1]) for d in zs])
vars_emp = np.concatenate([np.std(d, 0).reshape([1,-1]) for d in zs])

# means_emp = np.concatenate([m.cpu().detach().numpy().reshape([1,-1]) for m in model.z_mus])
# vars_emp = np.concatenate([(m/2).exp().cpu().detach().numpy().reshape([1,-1]) for m in model.z_lvs])

Z_with_means = np.concatenate([Z, means_emp], axis = 0)
manif = tsne(n_components = 2)
Z_embedded = manif.fit_transform(Z_with_means)

prev = 0
figure = plt.figure(figsize = [8,8])
till = no_tasks
tasks = np.arange(no_tasks)[:till]
for t in tasks:
    now = zs[t].shape[0]
    plt.plot(Z_embedded[prev:now+prev,0],Z_embedded[prev:now+prev,1], '.', label = "class_" + str(t))
    prev = now+prev

# prev = Z.shape[0]
# for t in tasks:
#     plt.plot(Z_embedded[prev+t:prev+t+1,0],Z_embedded[prev+t:prev+t+1,1], 'o', markersize=12, 
#              color = 'C'+str(t), markeredgecolor=(0,0,0,1), markeredgewidth=2)
    
plt.legend()
plt.savefig('./Gens/'+dataset_+'_tsne_plot.png')
plt.savefig('./Gens/'+dataset_+'_tsne_plot.eps', format = 'eps')

# Done 
def gen_samples(model, task_id, num_samples):
    with torch.no_grad():
        D = model.size[model.z_index]//2
        model.KL_B = [] # KL Divergence terms for the bernoulli distribution
        model.KL_G = [] # KL Divergence terms for the latent Gaussian distribution
        x = torch.tensor(means_emp[task_id]).unsqueeze(0).unsqueeze(0) + torch.randn(num_samples, 1, D).to(model.device)*(torch.tensor(vars_emp[task_id]).unsqueeze(0).unsqueeze(0))
        ret =  model.decode(x, task_id, model.no_pred_samples , const_mask  = True, temp = model.min_temp).mean(1)
        return ret

    
print('Generating emperical samples.')
with torch.no_grad():
    num_samples = 10
    logliks = []
    num_run = no_tasks
    cache = []
    # fig, ax = plt.subplots(num_samples, num_run, figsize = [8,8])
    for i in range(len(x_testsets[:num_run])):
        
        print("Generating Task " + str(i))
        x_test, y_test = x_testsets[i], y_testsets[i]
        pred = model.prediction_prob(x_test, i)
        pred_mean = np.mean(pred, axis=1) # N x O
        eps = 10e-8
        target = y_test#targets.unsqueeze(1).repeat(1, self.no_train_samples, 1)# Formating desired output : N x O
        loss = np.sum(- target * np.log(pred_mean+eps) - (1.0 - target) * np.log(1.0-pred_mean+eps) , axis = -1)
        log_lik = - (loss).mean()# Binary Crossentropy Loss
        logliks.append(log_lik)

#         samples = pred_mean[:num_samples]
#         samples = model.gen_samples(i, num_samples).cpu().detach().numpy()
        samples = F.sigmoid(gen_samples(model, i, num_samples)).cpu().detach().numpy()
        cache.append(samples)
        # for s in range(num_samples):
        #     ax[s][i].imshow(np.reshape(samples[s], [28,28]), cmap = 'gray')
    
    # plt.savefig('./Gens/'+dataset_+'_generated.png')

fig = plt.figure(figsize = [8,8])
show_images(torch.tensor(cache).permute(1,0,2).reshape(-1,28,28).unsqueeze(1))
plt.savefig('./Gens/'+dataset_+'generated.png')
plt.savefig('./Gens/'+dataset_+'generated.eps', format = 'eps')

with torch.no_grad():
    num_samples = 10
    logliks = []
    num_run = no_tasks
    cache = []
    # fig, ax = plt.subplots(num_samples, num_run, figsize = [8,8])
    for i in range(len(x_testsets[:num_run])):

        print("Reconstructing Task " + str(i))
        x_test, y_test = x_testsets[i], y_testsets[i]
        pred = model.prediction_prob(x_test, i)
        pred_mean = np.mean(pred, axis=1) # N x O
        eps = 10e-8
        target = y_test#targets.unsqueeze(1).repeat(1, self.no_train_samples, 1)# Formating desired output : N x O
        loss = np.sum(- target * np.log(pred_mean+eps) - (1.0 - target) * np.log(1.0-pred_mean+eps) , axis = -1)
        log_lik = - (loss).mean()# Binary Crossentropy Loss
        logliks.append(log_lik)

        samples = pred_mean[:num_samples]
        cache.append(samples)
        
#         samples = model.gen_samples(i, num_samples).cpu().detach().numpy()
#         samples = F.sigmoid(gen_samples(model, i, num_samples)).cpu().detach().numpy()
        # for s in range(num_samples):
        #     ax[s][i].imshow(np.reshape(samples[s], [28,28]), cmap = 'gray')

    # plt.savefig('./Gens/'+dataset_+'_reconstructed.png')

fig = plt.figure(figsize = [8,8])
show_images(torch.tensor(cache).permute(1,0,2).reshape(-1,28,28).unsqueeze(1))
plt.savefig('./Gens/'+dataset_+'recon.png')
plt.savefig('./Gens/'+dataset_+'recon.eps', format = 'eps')      

print('Generating Samples')
for i in range(no_tasks):
    task = i+1
    tmp = np.zeros([10, 784])
    model = get_model(get_till = task, dict_path = './' + paths[dataset_])
    for j in range(task):
        print(i,j)
#         tmp[j] = model.gen_samples(j, 1).cpu().detach().numpy()
        tmp[j] = F.sigmoid(gen_samples(model, j, 1)).cpu().detach().numpy()
    if task == 1:
        x_gen_all = tmp
    else:           
        x_gen_all = np.concatenate([x_gen_all, tmp], 0)
        
plot_images(x_gen_all, [28,28], './Gens/', dataset_+'_gen_all')