import nltk
import numpy as np
import sklearn_crfsuite
import re
import spacy
from collections import Counter
import pandas as pd
import string
import Feature_extraction as fe
from sklearn.model_selection import StratifiedKFold
import csv

features = []
n_splits = 10
seed = 7

data = pd.read_csv("./../data/data_all.csv", encoding="utf-8")


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, s, t) for w, p, s, t in zip(s["Token"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Shape"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Problem").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s    
        except:
            self.empty = True
            return None, None, None, None


# A function for extracting features in documents
def extract_features(doc):
    return [fe.word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, shape, label) in doc]
def get_tokens(doc):
    return [token for (token, postag, shape, label) in doc]

def get_text(doc):
    return [(token,label) for (token, postag, shape, label) in doc]


def collectExpressions(taggedText, numOfLabels = 3):
    exps = []
    length = len(taggedText)
    i = 0
    while i < length:  # for i, tuple in enumerate(taggedText):
        # print(i)
        if taggedText[i][1] == 'B-e_1':
            exp = ''
            efs = False
            while i < len(taggedText) and taggedText[i][1] != 'O':
                exp += str(taggedText[i][0])
                if i + 1 < len(taggedText):
                    i += 1
                    if(taggedText[i][1] == 'B-e_1'):
                        break
                else:
                    efs = True
                    break
            if exp != '' : exps.append(exp)
            if (efs):
                break
        i = i + 1
    return exps

def get_tuples(array1, array2):
    return [(elem1, elem2) for elem1, elem2 in zip(array1, array2)]

getter = SentenceGetter(data)
sent = getter.get_next()
sentences = getter.sentences

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
Accuracy = 0.0
prec = 0.0
avg_recall = 0.0
f1_score = 0.0
n_x = len(sentences)

for train, test in kfold.split(sentences, np.zeros(shape=(n_x,1))):
    data_tr = [sentences[i] for i in train]
    data_te = [sentences[i] for i in test]

    t_train = [get_tokens(doc) for doc in data_tr]
    X_train = [extract_features(doc) for doc in data_tr]
    y_train = [get_labels(doc) for doc in data_tr]
    Z_train = [collectExpressions(get_text(doc)) for doc in data_tr]

    t_test = [get_tokens(doc) for doc in data_te]
    X_test = [extract_features(doc) for doc in data_te]
    y_test = [get_labels(doc) for doc in data_te]
    Z_test = [collectExpressions(get_text(doc)) for doc in data_te]

    labels = {"B-e_1": 1, "I-e_1": 2,"O": 0}

    #=======================================

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )



    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    predicted_exps = [collectExpressions(get_tuples(tokens, lebels),3) for tokens, lebels in zip(t_test, y_pred)]
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])


    sample_count = 0
    pred_count = 0
    true_positives = 0
    false_negative = 0
    false_positive = 0

    for i, doc in enumerate(Z_test):
        pred = predicted_exps[i]
        for j, actual_exp in enumerate(doc):
            sample_count = sample_count + 1
            if(actual_exp in pred):
                true_positives = true_positives + 1
            if(actual_exp not in pred):
                false_negative = false_negative + 1
        for predicted_exp in pred:
            if predicted_exp not in doc:
                false_positive = false_positive + 1
    acc = true_positives/(true_positives+false_negative+false_positive)
    recall = true_positives/(true_positives+false_negative)
    precision = true_positives/(true_positives+false_positive)
    f1 = 2*(recall * precision)/(recall + precision)

    Accuracy  += acc
    prec  += precision
    avg_recall  += recall
    f1_score  += f1

print('Accuracy',Accuracy/n_splits)
print('Recall',avg_recall/n_splits)
print('precision',prec/n_splits)
print('f1_score',f1_score/n_splits)
