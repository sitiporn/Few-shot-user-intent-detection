# DeCLUTR : Deep Contrastive Learning for unsupervsied Textual Represenation



s_{i} : sample anchor from document
s_{j} : sample anchor from document


"span" : maximize the agreement between textual segments

d :  the document 
N : the mini-batch of document  


A : anchor maximize the agreement between textual segments per document

P : positive span per anchor 


simplify : A = P = 1

anchor positive span pair as s_{i}, s_{j}

s_{i}, s_{j} feed to same encoder f(.) and pooler g(.)

produce embedding: e_{i} = g(f(si)), e_{j} = g()

 ** other embedding in mini-batch as negative 

A : nums of anchor spans sampled per document
P : num of positive spans sampled per anchor


i \in {1...AN} : index of an arbitary anchor span

p \in {1..P} positive spans

An anchor span : si

its corresponding p \in {1...P}  : s_{i+pAN}


this required to maximize chance of sampling semantically similar anchor-positive pairs  




