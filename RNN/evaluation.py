import numpy as np


def getExpressions(taggedText):
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


def evaluate(X_te, y_te, model, data_model, X_word_te):
    predicted_exps = []
    test_exps = []
    p = model.predict(X_te)
    if len(X_te)>1 : X_te = X_te[0]
    for i, prob in enumerate(p):
        prob = np.argmax(prob, axis=-1)
        true_tl = []
        pred_tl = []
        for j, (w, true, pred) in enumerate(zip(X_word_te[i], y_te[i], prob)):
            word = ''
            if w in data_model.idx2word.keys():
                word = data_model.idx2word[w]
            else:
                word = "UNK"
            
            true_tl.append((word, data_model.idx2tag[true]))
            pred_tl.append((word, data_model.idx2tag[pred]))
        predicted_exps.append(getExpressions(pred_tl))
        test_exps.append(getExpressions(true_tl))
##        print("EXPECTED  :", test_exps[i])
##        print("PREDICTED :", predicted_exps[i]
        
##        
    true_positives, false_negative, false_positive, pred_count, sample_count = 0,0,0,0,0
    for i, doc in enumerate(test_exps):
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
    f1_score = 2*(recall*precision)/(recall+precision)
    
    return {"Accuracy": acc,
            "Recall": recall,
            "precision": precision,
            "f1_score": f1_score}
