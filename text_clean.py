#!/usr/bin/env python
# coding: utf-8

# In[1]:


def word_cleaned(text):
    
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from string import punctuation
    import re
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    
    tokens  = word_tokenize(text.lower()) # Initial list
        
    punctuation_list = list(punctuation)
        
    stopwords_english = stopwords.words('english')
        
    new_tokens = [] # A new list
    
    if len(tokens) > 1:
        
        for token in tokens:
            
            if (token not in punctuation_list) and (token not in stopwords_english):
                
                token = token.replace("-", "")  # Remove hyphens from words
            
                token = token.replace(".", "") # Remove dots from words to normalise abbreviations
            
                regex_check = re.match(f'[a-z]+', token)
                
                if regex_check != None:
                    
                    new_tokens.append(token)
        
    tokens_lemmatized = []
        
    def penn_to_wordnet(penn_pos_tag):
        tag_dictionary = {'NN':'n', 'JJ':'a','VB':'v', 'RB':'r'}
        return tag_dictionary.get(penn_pos_tag[:2], 'n')
        
    wnl = WordNetLemmatizer()
        
    tokens_pos_tag = pos_tag(new_tokens)
        
    for word, tag in  tokens_pos_tag:
        
        tokens_lemmatized.append(wnl.lemmatize(word, pos = penn_to_wordnet(tag)))
        
    new_text = " ".join(tokens_lemmatized)
            
    return new_text

