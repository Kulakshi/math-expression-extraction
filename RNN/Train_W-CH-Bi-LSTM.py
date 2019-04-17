import pandas as pd
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import data_model
import evaluation
import os
max_len_char = 200
max_len_sent = 50
n_epoch = 30
n_tags = 3
n_splits = 10
seed = 7

data = data_model.DataModel("./../data/data_all.csv")
sentences = data.get_sents()

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
Acc = 0.0
prec = 0.0
recall = 0.0
f1 = 0.0
n_x = len(sentences)


for train, test in kfold.split(sentences, np.zeros(shape=(n_x,1))):
    X_word_tr, X_word_te, y_tr, y_te, X_char_tr, X_char_te  = data.prepare_data(train, test, max_len_sent, max_len_char)
    n_words = data.n_words
    n_chars = data.n_char
    print(n_words, n_chars)
    
    word_in = Input(shape=(max_len_sent,))
    emb_word = Embedding(input_dim=n_words+1, output_dim=20,
                         input_length=max_len_sent, mask_zero=True)(word_in)

    char_in = Input(shape=(max_len_sent, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=10,
                               input_length=max_len_char, mask_zero=True))(char_in)
    char_enc = TimeDistributed(LSTM(units=50, return_sequences=False,
                                    recurrent_dropout=0.2))(emb_char)

    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags+1, activation="softmax"))(main_lstm)

    model = Model([word_in, char_in], out)

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    n_epoch = n_epoch
    history = model.fit([X_word_tr,
                        np.array(X_char_tr).reshape(len(X_char_tr), max_len_sent, max_len_char)],
                        np.array(y_tr).reshape(len(y_tr), max_len_sent, 1),
                        batch_size=20,
                        epochs=n_epoch,
                        validation_split = 0.1,
                        verbose=1)

    #======================
    #Eval
    #====================== 
    result = evaluation.evaluate([X_word_te,
                    np.array(X_char_te).reshape((len(X_char_te),max_len_sent, max_len_char))],
                    y_te, model, data, X_word_te)
    
    Acc  += result['Accuracy']
    prec  += result['precision']
    recall  += result['Recall']
    f1  += result['f1_score']

print('Accuracy',Acc/n_splits)
print('Recall',recall/n_splits)
print('precision',prec/n_splits)
print('f1_score',f1/n_splits)




