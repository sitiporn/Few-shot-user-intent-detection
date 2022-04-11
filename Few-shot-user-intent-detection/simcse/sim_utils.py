import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer



def load_examples(file_path, do_lower_case=True):
    examples = []
    
    with open('{}/seq.in'.format(file_path),'r',encoding="utf-8") as f_text, open('{}/label'.format(file_path),'r',encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            
            e = Inputexample(text.strip(),label=label.strip())
            examples.append(e)
            
    return examples

# each sentence has a sentence and label format 

class Inputexample(object):
    def __init__(self,text_a,label = None):
        self.text = text_a
        self.label = label

 
class CustomTextDataset(Dataset):
    def __init__(self,labels,text,batch_size,repeated_label:bool=False):
        self.labels = labels
        self.text = text
        self.batch_size = batch_size 
        self.count = 0 
        self.batch_labels = []
        self.repeated_label = repeated_label
        
        if self.repeated_label == True:
            print("Train on Combine between Supervised Contrastive and Cross Entropy loss")
            
        else:
            print("Train on Cross Entropy loss")
            
        
        print("len of dataset :",len(self.labels))
              
     
          

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        
        
        # write code here for 1)
        if self.repeated_label == True:
        
            if len(np.unique(self.batch_labels)) == self.batch_size - 1:


                while True:
                    idx = np.random.choice(len(self.labels))

                    if self.labels[idx]  in self.batch_labels:

                       
                        break

        self.batch_labels.append(self.labels[idx])
        
        label = self.labels[idx]
        
        data = self.text[idx]
        
        sample = {"Class": label,"Text": data}


    
        return sample

def freeze_layers(model,freeze_layers_count:int=0):

        """
        model : model object that we create 
        freeze_layers_count : the number of layers to freeze 
        """
        # write the code here
    
        # should not more than the number of layers in a backbone
        assert freeze_layers_count <= 12

        for name, param in model.named_parameters():
           

            keys = name.split(".")

            if str(freeze_layers_count) in keys or 'classifier' in keys:
                break
            
            param.requires_grad = False 


        #print all parameter that we want to train from scratch 
        
        for name, param in model.named_parameters():
            
            if param.requires_grad == True:
                 
                print(name)
        
    
        return model     
def train(model,device,optimizer,train_loader,valid_loader,tokenizer,epochs:int=30):

    train_loss_hist = [] 
    valid_loss_hist = []
    
    train_acc_hist = []
    valid_acc_hist = []
    

    test_acc = []

    min_valid_loss = np.inf
   
    for e in range(epochs):  # loop over the dataset multiple times

       
        model.train()
        correct = 0
        running_loss = 0.0
       
    
        for (idx, batch) in enumerate(train_loader):
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

            # Foward pass 
            # outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            outputs = model(**inputs,labels=labels)
            # get loss and output 
            loss, logits = outputs[:2]
            
            
            # Calculate gradients
            loss.backward()
            
           # Update Weights
            optimizer.step()
            
            # Calculate Loss
            running_loss += loss.item()
            
            #calculate nums of correction 
            correct += (torch.max(logits,dim=-1)[1] == labels).sum()
            
           
            
            #clear_output(wait=True)
        
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

            # move parameter to device
            inputs = {k:v.to(device) for k,v in inputs.items()}

            # map string labels to class idex
            labels = [label_maps[stringtoId] for stringtoId in (batch['Class'])]

            # convert list to tensor
            labels = torch.tensor(labels).unsqueeze(0)
            labels = labels.to(device)

            # Foward pass 
            # outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            outputs = model(**inputs,labels=labels)
            # get loss and output 
            loss, logits = outputs[:2]
        
            
            # Calculate Loss
            running_loss += loss.item()
            
            #calculate nums of correction 
            correct += (torch.max(logits,dim=-1)[1] == labels).sum()
            
           
        #  add to collect log 
        
        valid_loss_hist.append(running_loss/len(valid_data))
        valid_acc_hist.append(correct/len(valid_data))
        
        print(f' Validation Loss: {running_loss/len(valid_data)}, \t\t Validation acc: {correct/len(valid_data)}')
        
        print("valid correct : ",correct)
        print("valid total :",len(valid_data))
       
        # save best current model 
        if min_valid_loss > (running_loss/len(valid_data)):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{running_loss/len(valid_data):.6f}) \t Saving The Model')
            min_valid_loss = running_loss/len(valid_data) 
            torch.save(model.state_dict(), '../../../fewshot_models/saved_model.pth')
            
           
    return (train_acc_hist, train_loss_hist), (valid_acc_hist, valid_loss_hist)  


#  no gradients needed
def test(model,device,label_maps,test_loader,data_size,tokenizer):
    
    correct = 0
         
    with torch.no_grad():
        for (idx, batch) in enumerate(test_loader):
            sentence = batch["Text"]
            inputs = tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")

            # move parameter to device
            inputs = {k:v.to(device) for k,v in inputs.items()}

            # map string labels to class idex
            labels = [label_maps[stringtoId] for stringtoId in (batch['Class'])]

            # convert list to tensor
            labels = torch.tensor(labels).unsqueeze(0)
            labels = labels.to(device)
            
            # Foward pass 
            outputs = model(**inputs,labels=labels)

            # get loss and output 
            loss, logits = outputs[:2]
            
            _, predicted = torch.max(logits, -1)
            
            
           
         
            correct += (predicted == labels).sum().item()
            
    print("correct :",correct)
    print("total :",data_size)
         
    return correct / data_size
