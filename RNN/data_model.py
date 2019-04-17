import pandas as pd
import numpy as np
import json
import os
import io
from keras.preprocessing.sequence import pad_sequences
import string
import re


#======================
#utils.py
#======================
class DataModel(object):

    
    def __init__(self, dataFile):
        self.data = pd.read_csv(dataFile, encoding="utf-8")
        agg_func = lambda s: [(w, p, s, t) for w, p, s, t in zip(s["Token"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Shape"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.sentences = [s for s in self.data.groupby("Problem").apply(agg_func)]

    def get_sents(self):
        return self.sentences
        
    def prepare_data(self, train_ids, test_ids, max_len_sent, max_len_char):
        
        sent_tr = [self.sentences[i] for i in train_ids]
        sent_te = [self.sentences[i] for i in test_ids]
                  
        #prepare vocabularies
        words = list(set([props[0] for sent in self.sentences for props in sent]))
        chars = set([w_i for w in words for w_i in str(w)])
        pos = set([props[1] for sent in self.sentences for props in sent])
        tags = set([props[3] for sent in self.sentences for props in sent])
        print(tags)

        word2idx = self.prepare_vocab(words)
        char2idx = self.prepare_vocab(chars)
        pos2idx = self.prepare_vocab(pos)
        tag2idx = {t: i for i, t in enumerate(tags)}
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.idx2tag = {i: w for w, i in tag2idx.items()}
        
        #split dataset
        self.x_words_te = [[w[0] for w in s] for s in sent_te]
        
        #prepare numerical data
        self.n_words = len(word2idx)
        x_tr = [[word2idx[w[0]] for w in s] for s in sent_tr]
        x_tr = pad_sequences(maxlen=max_len_sent, sequences=x_tr, padding="post",  value=word2idx["PAD"])
        x_te = [[word2idx[w[0]] if w[0] in word2idx else word2idx["UNK"] for w in s] for s in sent_te]
        x_te = pad_sequences(maxlen=max_len_sent, sequences=x_te, padding="post",  value=word2idx["PAD"])

   
        self.n_char = len(char2idx)
        x_char_tr = self.prepare_x_char(sent_tr, char2idx, max_len_sent, max_len_char)
        x_char_te = self.prepare_x_char(sent_te, char2idx, max_len_sent, max_len_char)
        print(x_char_te[0])
       
        y_tr = [[tag2idx[w[3]] for w in s] for s in sent_tr]
        y_tr = pad_sequences(maxlen=max_len_sent, sequences=y_tr, padding="post", value=tag2idx["O"])
        y_te = [[tag2idx[w[3]] for w in s] for s in sent_te]
        y_te = pad_sequences(maxlen=max_len_sent, sequences=y_te, padding="post", value=tag2idx["O"]) 

        return (x_tr, x_te, y_tr, y_te, x_char_tr,x_char_te)


    def prepare_x_char(self, sentences, char2idx, max_len_sent, max_len_char):
        char = []
        for sentence in sentences:
            sent_seq = []#words in a sentence
            for i in range(max_len_sent):
                word_seq = []#chars in a word
                for j in range(max_len_char):
##                    try:
                    flag = True
                    if i < len(sentence):
                        if j < len(str(sentence[i][0])):
                            flag = False
                            try:
                                word_seq.append(char2idx.get(sentence[i][0][j]))
                            except:
##                                print(sentence[i][0])
##                                print(sentence[i][0][j])
                                word_seq.append(char2idx.get("UNK"))
                    if(flag):
                        word_seq.append(char2idx.get("PAD"))
##                        word_seq.append(char2idx.get(sentence[i][0][j]))
##                    except:
##                        word_seq.append(char2idx.get("PAD"))
                sent_seq.append(word_seq)
            char.append(np.array(sent_seq))
        return char
    
    def prepare_vocab(self, words):
        word2idx = {}
        word2idx = {w: i+2 for i, w in enumerate(words)}
        word2idx["UNK"] = 1
        word2idx["PAD"] = 0
        return word2idx
