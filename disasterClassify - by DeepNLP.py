# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:03:35 2020

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(columns = ['id','keyword','location'])
dataset_test = pd.read_csv('test.csv')
dataset_test = dataset_test.drop(columns = ['id','keyword','location'])
chat_train=dataset['text']
chat_test=dataset_test['text']
y_train=dataset['target']

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#pip install pyspellchecker
from spellchecker import SpellChecker 

#from autocorrect import Speller
#check = Speller(lang='en')
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

#Remove urls
def keep(word):
    if not str(word).lower().startswith('http'):
        return word
    else:
        return ' '

spell = SpellChecker(distance=1)
ps = PorterStemmer()
#clean text
def clean_text(text):
    text = text.split()
    text = [keep(word) for word in text ]
    text = ' '.join(text) 
    text = re.sub('[^a-zA-Z#@]', ' ',text )
    text = reduce_lengthening(text)
    text=text.lower()
    text = text.split()
    text= [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    #chat = [check.autocorrect_word(word) for word in chat ]
    text= [spell.correction(word) for word in text ]
    text = ' '.join(text) 
    return text
    
#clean chat
Xtrain_clean=[]
for chat in chat_train:
    Xtrain_clean.append(clean_text(chat))
    
Xtest_clean=[]
for chat in chat_test:
    Xtest_clean.append(clean_text(chat))

from keras.preprocessing.text import Tokenizer
token=Tokenizer()   
X_total=Xtrain_clean+Xtest_clean
token.fit_on_texts(X_total)
X_train=token.texts_to_sequences(Xtrain_clean)
X_test=token.texts_to_sequences(Xtest_clean)
from keras.preprocessing import sequence
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

vocabulary_size=len(token.word_index)+1
embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)


sample=pd.read_csv('test.csv')
id_identifier = sample['id']
sample = sample.drop(columns = ['id'])
y_samplepred=model.predict(X_test)
y_samplepred = (y_samplepred> 0.5).astype('int32')
final_results=pd.DataFrame(data=y_samplepred,columns=['target'])
final_results = pd.concat([id_identifier, final_results['target']], axis = 1).dropna()
final_results['id'] =final_results['id'].astype('int32')

final_results.to_csv('mysample_submission_deep.csv', index = False)

