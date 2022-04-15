import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.spatial.distance import cosine
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'simcse')))
from sim_utils import load_examples, Inputexample, CustomTextDataset, freeze_layers, test
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, AutoModel, AutoTokenizer
from loss import Similarity, create_supervised_pair, supervised_contrasive_loss


def train_contrastive_learnig(model,optimizer,loss_fct,label_maps,temp,train_loader,tokenizer,valid_loader,device,epochs:int=30):
    
    
    train_loss_hist = [] 
    valid_loss_hist = []
    
    train_acc_hist = []
    valid_acc_hist = []
    

    test_acc = []

    min_valid_loss = np.inf
    skip_time = 0 # the number of time that yi not equal to yj in supervised contrastive loss equation
    for e in range(epochs):  # loop over the dataset multiple times
 
        model.train()
        correct = 0
        running_loss = 0.0
       

        for (idx, batch) in enumerate(train_loader):
            sentence = batch["Text"]
            inputs = tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")


            #assert len(np.unique(batch["Class"])) == len(batch["Class"])  
            # move parameter to device
            inputs = {k:v.to(device) for k,v in inputs.items()}

            # map string labels to class idex
            labels = [label_maps[stringtoId] for stringtoId in (batch['Class'])]

            # convert list to tensor
            labels = torch.tensor(labels).unsqueeze(0)
            labels = labels.to(device)


             # clear gradients
            optimizer.zero_grad()
            
            
            outputs = model(**inputs,labels=labels,output_hidden_states=True)     
        
            hidden_states = outputs.hidden_states

            last_hidden_states = hidden_states[12]

            # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
            # (batch_size,seq_len,embed_dim)
            h = last_hidden_states[:,0,:]

            # create pair samples
            T, h_i, h_j, idx_yij = create_supervised_pair(h,batch['Class'],debug=False)

            if h_i is None:
                print("skip this batch")
                skip_time +=1 
                continue

            # supervised contrastive loss 
            
            loss_s_cl = supervised_contrasive_loss(device,loss_fct,h_i, h_j, h, T,temp=temp,idx_yij=idx_yij,debug=False)

            # cross entropy loss
            loss_classify, logits = outputs[:2]

            # loss total
            loss = loss_s_cl + (lamda * loss_classify )

            # Calculate gradients
            loss.backward()

            # Update Weights
            optimizer.step()

            # Calculate Loss
            running_loss += loss.item()
            
            #calculate nums of correction 
            correct += (torch.max(logits,dim=-1)[1] == labels).sum()

        
        train_loss_hist.append(running_loss/len(train_data))
        train_acc_hist.append(correct/len(train_data))
        
        
        print(f'======  Epoch {e+1} ====== ')
        print(f' Training Loss: {running_loss/len(train_data)}, \t\t Training acc: {correct/len(train_data)}')
        
        print("train correct : ",correct)
        print("train total :",len(train_data))
        
        
        running_loss = 0.0
        correct = 0
        model.eval()     # Optional when not using Model Specific layer
        log_correct = []
        
        
        for (idx, batch) in enumerate(valid_loader):
            
            sentence = batch["Text"]
            inputs = tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")


            #assert len(np.unique(batch["Class"])) < len(batch["Class"])  
            # move parameter to device
            inputs = {k:v.to(device) for k,v in inputs.items()}

            # map string labels to class idex
            labels = [label_maps[stringtoId] for stringtoId in (batch['Class'])]

            # convert list to tensor
            labels = torch.tensor(labels).unsqueeze(0)
            labels = labels.to(device)


             # clear gradients
            optimizer.zero_grad()
            
            
            outputs = model(**inputs,labels=labels,output_hidden_states=True)     
        
            hidden_states = outputs.hidden_states

            last_hidden_states = hidden_states[12]

            # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
            # (batch_size,seq_len,embed_dim)
            h = last_hidden_states[:,0,:]

            # create pair samples
            T, h_i, h_j, idx_yij = create_supervised_pair(h,batch['Class'],debug=False)

          
            # supervised contrastive loss 
            loss_s_cl = supervised_contrasive_loss(device,h_i, h_j, h, T,temp=temp,idx_yij=idx_yij,debug=False)

            # cross entropy loss
            loss_classify, logits = outputs[:2]

            # loss total
            loss = loss_s_cl + (lamda * loss_classify )
            
            # Calculate Loss
            running_loss += loss.item()
            
            #calculate nums of correction 
            correct += (torch.max(logits,dim=-1)[1] == labels).sum()
            
        # add code to logging
        valid_loss_hist.append(running_loss/len(valid_data))
        valid_acc_hist.append(correct/len(valid_data))
        
        print(f' Validation Loss: {running_loss/len(valid_data)}, \t\t Validation acc: {correct/len(valid_data)}')
        
        print("valid correct : ",correct)
        print("valid total :",len(valid_data))
       
        
        
        # save best current model 
        if min_valid_loss > (running_loss/len(valid_data)):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{running_loss/len(valid_data):.6f}) \t Saving The Model')
            min_valid_loss = running_loss/len(valid_data) 
            torch.save(model.state_dict(), 'saved_model.pth')
            
       
            
    return (train_acc_hist, train_loss_hist), (valid_acc_hist, valid_loss_hist)
