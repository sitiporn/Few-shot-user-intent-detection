import torch  
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
