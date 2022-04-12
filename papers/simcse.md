# SimCSE : Simple Contrastive Learning of sentence Embeddings


Unsupervised  -> predict itself in contrastive objective
  
  * idea  
     they foward the same sentence two time -> get positive pairs
     and other sentence in the batch as negative pairs

     model should be able to predict positive one among negatives  



What this method solve ? 
 
 * they solve collaps mode by resulting standropout from network
    
    why this algo make it good ? 
       * it make postive pairs more alignment and unifromly (contrastive learning) 

what do they mean ? 
  the positive pairs it align better when getting supervised signal

what is anisotropym mean ? 


Background 

1 Contrastive learning make this algo because it's better positive align and unifromly 

  eq2 calculate expected distance between pairs of embeddings by given prob of positive pairs 
      
      Ppos ~ prob distribution of positive pairs

  eq3 how well embeddings are unifromly distributed 
      
      Pdata ~ prob distribution of data 


Method 


Unsupervised SimCSE

z ~ standardard random mask for dropout 
     * it's just the way of augment data by resulting dropout in fc and encode it and get the result

