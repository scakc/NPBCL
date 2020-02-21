import os
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
matplotlib.use('Agg')
import gzip
import pickle

class IBP_BCL:
    def __init__(self, hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size=0, single_head=True):
        '''
        hidden_size : list of network hidden layer sizes
        alpha : IBP prior concentration parameters
        data_gen : Data Generator
        coreset_size : Size of coreset to be used (0 represents no coreset)
        single_head : To given single head output for all task or multihead output for each task seperately.
        '''
        ## Intializing Hyperparameters for the model.
        self.hidden_size = hidden_size
        self.alpha = alpha#[alpha for i in range(len(hidden_size)*2-1)]
        self.beta = [1.0 for i in range(len(hidden_size)*2-1)]
        self.no_epochs = no_epochs
        self.data_gen = data_gen
        if(coreset_method != "kcen"):
            self.coreset_method = self.rand_from_batch
        else:
            self.coreset_method = self.k_center
        self.coreset_size = coreset_size
        self.single_head = single_head 
        self.cuda = torch.cuda.is_available()
    
    def rand_from_batch(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ Random coreset selection """
        # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        idx = np.random.choice(x_train.shape[0], coreset_size, False)
        x_coreset.append(x_train[idx,:])
        y_coreset.append(y_train[idx,:])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train    

    def k_center(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ K-center coreset selection """
        # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        dists = np.full(x_train.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, x_train, current_id)
        idx = [current_id]
        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = update_distance(dists, x_train, current_id)
            idx.append(current_id)
        x_coreset.append(x_train[idx,:])
        y_coreset.append(y_train[idx,:])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists
    
    def merge_coresets(self, x_coresets, y_coresets):
        ## Merges the current task coreset to rest of the coresets
        merged_x, merged_y = x_coresets[0], y_coresets[0]
        for i in range(1, len(x_coresets)):
            merged_x = np.vstack((merged_x, x_coresets[i]))
            merged_y = np.vstack((merged_y, y_coresets[i]))
        return merged_x, merged_y
    
    def logit(self, x):
        eps = 10e-8
        return (np.log(x+eps) - np.log(1-x+eps))
    
    def get_soft_logit(self, masks, task_id):
        var = []
        for i in range(len(masks)):
            var.append(self.logit(masks[i][task_id]*0.8 + 0.1))
        
        return var
       
    def get_scores(self, model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, 
                   no_epochs, single_head, batch_size=None, kl_mask = None):
        ## Retrieving the current model parameters
        mf_model = model
        mf_weights, mf_variances = model.get_weights()
        prev_masks, self.alpha, self.beta = mf_model.get_IBP()
        logliks = []
        
        ## In case the model is single head or have coresets then we need to test accodingly.
        if single_head:# If model is single headed.
            if len(x_coresets) > 0:# Model has non zero coreset size
                del mf_model
                torch.cuda.empty_cache() 
                x_train, y_train = self.merge_coresets(x_coresets, y_coresets)
                prev_pber = self.get_soft_logit(prev_masks,i)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = IBP_BAE(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], self.max_tasks,
                                   prev_means=mf_weights, prev_log_variances=mf_variances, 
                                   prev_masks = prev_masks, alpha=alpha, beta = beta, prev_pber = prev_pber, 
                                   kl_mask = kl_mask, single_head=single_head)
                final_model.ukm = 1
                final_model.batch_train(x_train, y_train, 0, self.no_epochs, bsize, max(self.no_epochs//5,1))
            else:# Model does not have coreset
                final_model = model

        ## Testing for all previously learned tasks
        num_samples = 10
        fig, ax = plt.subplots(num_samples, len(x_testsets), figsize = [10,10])
        for i in range(len(x_testsets)):
            if not single_head:# If model is multi headed.
                if len(x_coresets) > 0:
                    try:
                        del mf_model
                    except:
                        pass
                    torch.cuda.empty_cache() 
                    x_train, y_train = x_coresets[i], y_coresets[i]# coresets per task
                    prev_pber = self.get_soft_logit(prev_masks,i)
                    bsize = x_train.shape[0] if (batch_size is None) else batch_size
                    final_model = IBP_BAE(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], self.max_tasks, 
                                   prev_means=mf_weights, prev_log_variances=mf_variances, 
                                   prev_masks = prev_masks, alpha=alpha, beta = beta, prev_pber = prev_pber, 
                                   kl_mask = kl_mask, learning_rate = 0.0001, single_head=single_head)
                    final_model.ukm = 1
                    final_model.batch_train(x_train, y_train, i, self.no_epochs, bsize, max(self.no_epochs//5,1), init_temp = 0.25)
                else:
                    final_model = model
            
            
            x_test, y_test = x_testsets[i], y_testsets[i]
            pred = final_model.prediction_prob(x_test, i)
            pred_mean = np.mean(pred, axis=1) # N x O
            eps = 10e-8
            target = y_test#targets.unsqueeze(1).repeat(1, self.no_train_samples, 1)# Formating desired output : N x O
            loss = np.sum(- target * np.log(pred_mean+eps) - (1.0 - target) * np.log(1.0-pred_mean+eps) , axis = -1)
            log_lik = - (loss).mean()# Binary Crossentropy Loss
            logliks.append(log_lik)

            # samples = pred_mean[:num_samples]
            samples = final_model.gen_samples(i, num_samples).cpu().detach().numpy()
            recosn = pred_mean[:num_samples]
            for s in range(num_samples):
                if(len(x_testsets) == 1):
                    ax[s].imshow(np.reshape(recosn[s], [28,28]))
                else:
                    ax[s][i].imshow(np.reshape(recosn[s], [28,28]))

        plt.savefig('./Gens/Task_till_' + str(i) +'.png')

        return logliks

    def concatenate_results(self, score, all_score):
        ## Concats the current accuracies on all task to previous result in form of matrix
        if all_score.size == 0:
            all_score = np.reshape(score, (1,-1))
        else:
            new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
            new_arr[:] = np.nan# Puts nan in place of empty values (tasks that previous model was not trained on)
            new_arr[:,:-1] = all_score
            all_score = np.vstack((new_arr, score))
        return all_score
        
    def batch_train(self, batch_size=None):
        '''
        batch_size : Batch_size for gradient updates
        '''
        np.set_printoptions(linewidth=np.inf)
        ## Intializing coresets and dimensions.
        in_dim, out_dim = self.data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []
        x_trainset, y_trainset = [], []
        all_acc = np.array([])
        self.max_tasks = self.data_gen.max_iter
        # fig1, ax1 = plt.subplots(1,self.max_tasks, figsize = [10,5])
        ## Training the model sequentially.
        for task_id in range(self.max_tasks):
            ## Loading training and test data for current task
            x_train, y_train, x_test, y_test = self.data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)
            ## Initializing the batch size for training 
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            ## If this is the first task we need to initialize few variables.
            if task_id == 0:
                prev_masks = None
                prev_pber = None
                kl_mask = None
                mf_weights = None
                mf_variances = None
            ## Select coreset if coreset size is non zero
            if self.coreset_size > 0:
                x_coresets,y_coresets,x_train,y_train = self.coreset_method(x_coresets,y_coresets,x_train,y_train,self.coreset_size)
            ## Training the network  
            mf_model = IBP_BAE(in_dim, self.hidden_size, out_dim, x_train.shape[0], self.max_tasks, 
                               prev_means=mf_weights, prev_log_variances=mf_variances, 
                               prev_masks = prev_masks, alpha=self.alpha, beta = self.beta, prev_pber = prev_pber, 
                               kl_mask = kl_mask, single_head=self.single_head)
            if(self.cuda):
                mf_model = mf_model.cuda()
                if torch.cuda.device_count() > 1: 
                    mf_model = nn.DataParallel(mf_model) #enabling data parallelism
            mf_model.batch_train(x_train, y_train, task_id, self.no_epochs, bsize,max(self.no_epochs//5,1))
            mf_weights, mf_variances = mf_model.get_weights()
            prev_masks, self.alpha, self.beta = mf_model.get_IBP()

            ## Figure of masks that has been learned for all seen tasks.
            # fig, ax = plt.subplots(1,task_id+1, figsize = [10,5])
            # for i,m in enumerate(prev_masks[0][:task_id+1]):
            #     if(task_id == 0):
            #         ax.imshow(m, vmin = 0, vmax = 1)
            #     else:
            #         ax[i].imshow(m,vmin=0, vmax=1)
            # fig.savefig("all_masks.png")
            
            ## Calculating Union of all task masks and also for visualizing the layer wise network sparsity
            sparsity = []
            kl_mask = []
            M = len(mf_variances[0])
            for j in range(M):
                ## Plotting union mask
                var = (np.sum(prev_masks[j][:task_id+1],0)>0.5)*1.02
                mask = (var > 0.5)*1
                mask2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.1)*1.0
                ## Calculating network sparsity
                var2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.5)
                kl_mask.append(var2)
                filled = np.mean(mask)
                sparsity.append(filled)
            
            # ax1[task_id].imshow(mask2,vmin=0, vmax=1)
            # fig1.savefig("union_mask.png")
            print("Network sparsity : ", sparsity)
            
            acc = self.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, 
                                  self.hidden_size, self.no_epochs, self.single_head, batch_size, kl_mask)

            torch.save(mf_model.state_dict(), "./saves/model_last_" + str(task_id))
            del mf_model
            torch.cuda.empty_cache() 
            all_acc = self.concatenate_results(acc, all_acc); print(all_acc.round(3)); print('*****')

        np.savetxt('./Gens/res.txt', all_acc)
        return [all_acc, prev_masks]