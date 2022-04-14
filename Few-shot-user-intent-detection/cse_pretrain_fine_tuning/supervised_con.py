import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
import random
#from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.spatial.distance import cosine
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'simcse')))
from sim_utils import load_examples, Inputexample, CustomTextDataset, freeze_layers, test
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, AutoModel, AutoTokenizer
from loss import Similarity, create_supervised_pair, supervised_contrasive_loss
from train import train_contrastive_learnig
#comment this if you are not using puffer


N = 5
data = []
labels = []

train_samples = []
train_labels = []

valid_samples = []
valid_labels = []

test_samples = []
test_labels = []

embed_dim = 768
batch_size = 16 

lr=2e-3  # you can adjust 
temp = 0.3  # you can adjust 
lamda = 0.01  # you can adjust  
skip_time = 0 # the number of time that yi not equal to yj in supervised contrastive loss equation 
data_name = 'CLINC150'
model_name = 'simcse_sup'
shot_name = 'train_5'
exp_name = f'{model_name}_lr={lr}_t={temp}_{data_name}_{shot_name}'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path_shot = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/{shot_name}/'
valid_path = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/valid/'
test_path = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/test/'

print("train path : ",path_shot)
print("valid path : ",valid_path)
print("test path  : ",test_path)
print("experiment code name :",exp_name)



# Download data fewshot 
# https://downgit.github.io/#/home?url=https:%2F%2Fgithub.com%2Fjianguoz%2FFew-Shot-Intent-Detection%2Ftree%2Fmain%2FDatasets%2FHWU64

# load data
train_samples = load_examples(path_shot)
valid_samples = load_examples(valid_path)
test_samples = load_examples(test_path)


print("===== small train set ====")

for i in range(len(train_samples)):
    data.append(train_samples[i].text)
    labels.append(train_samples[i].label)


train_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=True)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)



print("===== validation set ====")

data = []
labels = []

for i in range(len(valid_samples)):
    data.append(valid_samples[i].text)
    labels.append(valid_samples[i].label)

valid_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=True)
valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True)


print("===== test set ====")

data = []
labels = []
    
for i in range(len(test_samples)):
    data.append(test_samples[i].text)
    labels.append(test_samples[i].label)

test_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)



# got the number of unique classes from dataset
num_class = len(np.unique(np.array(labels)))

# get text label of uniqure classes
unique_label = np.unique(np.array(labels))

# map text label to index classes
label_maps = {unique_label[i]: i for i in range(len(unique_label))}



tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
config = AutoConfig.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
config.num_labels = num_class
simcse = AutoModelForSequenceClassification.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased",config=config)
simcse = freeze_layers(simcse,freeze_layers_count=12)



optimizer= AdamW(simcse.parameters(), lr=lr)
simcse = simcse.to(device)

train_log, valid_log = train_contrastive_learnig(simcse,optimizer,nn.CrossEntropyLoss(),label_maps,temp,train_loader,tokenizer,valid_loader,train_data,valid_data,device,lamda=lamda,epochs=30)
