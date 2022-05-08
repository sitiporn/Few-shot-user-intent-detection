# Few shot user intent detection  in state of art 

### Goal of experiments

1. Emulate the success of DNNC performance of OOS detection
2. which model perform well in-scope benchmark datasets
3. Is the distribution uniformly of embeddings agree with the accuracy of model in-scope on test set
4. How much knowledege should I keep from previous task(pretrain stage) to fine on downstream tasks  

![image](https://user-images.githubusercontent.com/31414731/167282886-6d39ac51-60b0-442d-8380-2ba0c17d05b4.png)


## Simple Contrastive Learning of Sentence Embeddings with 150 intents

![image](https://user-images.githubusercontent.com/31414731/166154720-d0156ac3-2653-4ed7-b35a-4ce82bd22ef7.png)
## Roberta-base with 150 intents
![image](https://user-images.githubusercontent.com/31414731/166154787-527e72a6-5802-4903-8d58-1b3a4a2e2475.png)


![image](https://user-images.githubusercontent.com/31414731/167282905-d2b44597-ab9e-4d82-8186-5fd385b9cf96.png)


1. SimCSE: Simple Contrastive Learning of Sentence Embeddings
   ref: https://github.com/princeton-nlp/SimCSE  
2. DNNC-few-shot-intent  
   ref: https://github.com/salesforce/DNNC-few-shot-intent 

3. Efficient intent Detection with Dual sentence Encoders 
   
   ref: https://arxiv.org/pdf/2003.04807.pdf


4. Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning 
  
   ref: https://arxiv.org/abs/2109.06349
   source : https://github.com/sitiporn/Reading-paper/tree/main/implement_papers 
   
5. DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations
   
   ref : https://arxiv.org/pdf/2006.03659.pdf   
   source : https://github.com/JohnGiorgi/DeCLUTR
   
   web blogs:
   1. https://few-shot-text-classification.fastforwardlabs.com/
   2. https://d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-bert.html
   3. https://bhuvana-kundumani.medium.com/implementation-of-simcse-for-unsupervised-approach-in-pytorch-a3f8da756839
   4. https://monkeylearn.com/blog/intent-classification/
   5. https://medium.com/analytics-vidhya/dialog-augmentation-and-intent-detection-5d76bdf0fcb8

Source of Datasets for lanugage tasks:

https://www.topbots.com/generalized-language-models-tasks-datasets/
