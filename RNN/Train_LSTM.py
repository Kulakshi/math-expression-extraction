import numpy as np
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import CuDNNLSTM, LSTM, Embedding, Dense, TimeDistributed, Dropout
from sklearn.model_selection import StratifiedKFold
import data_model
import evaluation

max_len_char = 15
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
    X_word_tr, X_word_te, y_tr, y_te, _, _  = data.prepare_data(train, test, max_len_sent, max_len_char)
    y_tr = [to_categorical(i, num_classes=n_tags) for i in y_tr]
    n_words = data.n_words
    n_chars = data.n_char
    
    input = Input(shape=(max_len_sent,))
    model = Embedding(input_dim=n_words+int(2/3*n_words), output_dim=50, input_length=max_len_sent)(input)
    model = Dropout(0.2)(model)
    model = LSTM(units=50, return_sequences=True, recurrent_dropout=0.1)(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  

    model = Model(input, out)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(X_word_tr, np.array(y_tr), batch_size=10, epochs=n_epoch,
              validation_split=0.1,
              verbose=1)
    #======================
    #Eval
    #====================== 
    result = evaluation.evaluate(X_word_te,y_te, model, data, X_word_te)
    
    Acc  += result['Accuracy']
    prec  += result['precision']
    recall  += result['Recall']
    f1  += result['f1_score']

print('Accuracy',Acc/n_splits)
print('Recall',recall/n_splits)
print('precision',prec/n_splits)
print('f1_score',f1/n_splits)

