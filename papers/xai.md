# XAI

ref

1. https://arxiv.org/pdf/2106.07410.pdf
2. https://arxiv.org/pdf/2010.00711.pdf 
3. http://kth.diva-portal.org/smash/get/diva2:1335846/FULLTEXT01.pdf

The common relationship in Machine Learning   
  
  * The neg relationship exist between the performance of model and its explainability  

 XAI classification approaches  

1. Linguistic knowledge of Neural networks

  ante-hoc systems
    - explainability from start 
  post-hoc systems
    - explainability on test cases and result 
 
  
  Another approaches 

    - one may focused on differrent component 
        eg. the most salient input features or meaingful features inside deep nets  


    - seek explaination by feature 

         * how the model internally represent the data  involves in differrent components
            * differrent linguistic units ranging from characters to sentences
            * trends
                * what linguistic knowledge is capture by neural networks  
                * why they make certain predictions  
                   
                   * if they are robust, interpret the way they represent  language and how they fail


     what linguistic information  is captured in nn ? 

     1. the methods used to analyse networks (eg. classification or clustering) 
     2. the type of linguistic information
        eg. sentence len, part of speechs, or concepts 

     3. the part nueral networks being investigated (eg. atteiontion weights, activation functions, or embeddings)  


     Auxiliary prediction tasks

     1. diagnotis classifiers or probing tasks
          * running nerual netwroks on some tasks ,running nueral networks on corpus with linguistic annotations and record its representations to predict property of interests and take ability to predict such property as indication that neural networks has learnt the property   
    
     2. how often attention weights agree with a lin property  
     3. computing the relation between  nn's activation and some property 
           * in Elmo  the first layer better to predict part of speech
        the second layer predict word sense

           * in some high quality systems the transalation quality and performance on the Auxiliary were negatively correlated ***


           * if encoding linguistic properties causes the model to perform well in some downstream task ?  it's always true or not what is the indication  

2. Explain predictions 

    * ability to explain specific predictions. Compare to other trends, the abiltiy to explain predictions in NLP still litmited and mostly futher work in this area.


    1. one way asks the model explaination along with predictions requires hard manual to collect manual annotations

    2. NLI challenges dataset focused lexical semantics


4. Interpreting Language Embeddings

5. Adversarial Example

Insummary building trust in AI that's still problem 


# problem of AI not XAI

1) Lack of explainability based one their decision 

in critical system when models play role and making correctly or incorrecly decisions users will question after they get indenifies  
2) we do not knowledge they collect it might re
it can help to improve model as what benefits or wasted knowledge they keep or use when they predict  

3) 

