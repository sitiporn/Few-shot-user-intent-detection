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
from sim_utils import load_examples, Inputexample, CustomTextDataset, freeze_layers, train, test
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, AutoModel, AutoTokenizer

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'




N = 10
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

data_names = ['CLINC150','BANKING77','HWU64']
model_names = ['sup-simcse-roberta-base']

shot_name = 'train_10'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for data_name in data_names:
    for model_name in model_names:


        path_shot = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/{shot_name}/'

        valid_path = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/valid/'
        test_path = f'../../../datasets/Few-Shot-Intent-Detection/Datasets/{data_name}/test/'
        exp_name = f'{model_name}_lr={lr}_t={temp}_{data_name}_{shot_name}'

        # load data
        train_samples = load_examples(path_shot)
        valid_samples = load_examples(valid_path)
        test_samples = load_examples(test_path)


        print("===== small train set ====")

        for i in range(len(train_samples)):
            data.append(train_samples[i].text)
            labels.append(train_samples[i].label)


        train_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=False)
        train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)



        print("===== validation set ====")

        data = []
        labels = []

        for i in range(len(valid_samples)):
            data.append(valid_samples[i].text)
            labels.append(valid_samples[i].label)

        valid_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=False)
        valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True)



        print("===== test set ====")

        data = []
        labels = []
            
        for i in range(len(test_samples)):
            data.append(test_samples[i].text)
            labels.append(test_samples[i].label)

        test_data = CustomTextDataset(labels,data,batch_size=batch_size,repeated_label=False)
        test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)



# got the number of unique classes from dataset
        num_class = len(np.unique(np.array(labels)))

# get text label of uniqure classes
        unique_label = np.unique(np.array(labels))

# map text label to index classes
        label_maps = {unique_label[i]: i for i in range(len(unique_label))}

        print("number of class :",num_class)

        
        direct_name = f"princeton-nlp/{model_name}"

        print("direct_name :",direct_name)

        #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
        config = AutoConfig.from_pretrained(direct_name)
        config.num_labels = num_class
        simcse = AutoModelForSequenceClassification.from_pretrained(direct_name,config=config)
        
        simcse = freeze_layers(simcse,freeze_layers_count=12)
        optimizer= AdamW(simcse.parameters(), lr=lr)
        simcse = simcse.to(device)

        train_log, valid_log = train(exp_name,simcse,device,label_maps,optimizer,train_loader,valid_loader,train_data,valid_data,tokenizer,epochs=30)


        test_acc = test(simcse,device,label_maps,test_loader,len(test_data),tokenizer)

    break











        
        

