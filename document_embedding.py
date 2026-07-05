#!/usr/bin/env python
# coding: utf-8

# In[8]:


def get_document_embedding(word_list, k):
    
    import numpy as np
    import nltk
    nltk.download('punkt')
    
    # Define GloVe embedding dictionary

    embeddings_dict = {}

    with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    
        for line in f:
        
            values = line.split()
            
            word = values[0]
        
            vector  = np.asarray(values[1:], dtype = float)
        
            embeddings_dict[word] = vector
    
    # Get document embedding
    
    document_embedding = np.zeros(k, dtype = float) ## Create embedding of k zero-valued elements
    
    valid_words = 0
    
    for word in word_list:
        
        try:
            document_embedding = document_embedding + embeddings_dict[word]
            
            valid_words += 1
        
        except:
            
            pass # If word embedding is not available, then ignore the word
        
    if valid_words > 0:
        
        document_embedding = document_embedding / valid_words
    
    else: 
        
        document_embedding = np.zeros(k, dtype = float) # In case valid words = 0
    
    return document_embedding

