import nltk
import re
import spacy
from collections import Counter
import string




def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)
    
def hasLetters(inputString):
    hasLetters = re.search('[a-zA-Z]', inputString)
    return hasLetters is not None

def hasPunctuation(inputString):
     return any(char in string.punctuation for char in inputString)
    
def getWordPattern(inputString):
    pattern = ''
    for c in inputString:
        if c.isalpha():
            if c.isupper():
                pattern = pattern + 'A'
            else:
                pattern = pattern + 'a'
        elif c.isdigit():
            pattern = pattern + '0'
        else:
            pattern = pattern + c
    return pattern

def getSummarizedWordPattern(inputString):
    pattern = getWordPattern(inputString)
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if i>0 and (c == 'd' or c == 'X' or c =='x'):
            if pattern[i-1] == c:
                pattern = pattern[:i-1] + pattern[i:]
            else:
                i = i + 1
        else:
            i = i + 1
    return pattern

def word2features(doc, i):
    token = str(doc[i][0])
    postag = doc[i][1]
    shape = doc[i][2]
    features = []
    delim = [')', ']', '}', '|','(', '[', '{', '|']
    operators = ['+', '-', '/','sin','cos','tan', '△','×','∥','∠','≥','>','<','=']
    tokens = [word for word,pos,shape,label in doc]
    bigrm = list(nltk.bigrams(tokens))
        
    features = [
        #set A
        'bias',
        'token=' + token,
        "UnigramFreq=" + str(Counter(tokens)[token]),
        #set B
        'word.isChar=%s' % (len(token)==1),
        #set C
        'token.isupper=%s' % token.isupper(),
        'token.islower=%s' % token.islower(),
        'token.isFirstLetterCapital=%s' % token[0].isupper(),
        'token.isNonInitCapitals=%s' % ((len([l for l in token[1:] if l.isupper()]) > 0) and token[0].islower()),
        #set D
        'token.isalpha=%s' % token.isalpha(),
        'token.isdigit=%s' % token.isdigit(),
        'token.isOperator=%s' % (token in operators),
        'token.isdelim=%s' % (token in delim),
        'token.isMixofLettersDigits=%s' % (hasNumbers(token) and hasLetters(token)),
        'token.hasPunctuation=%s' % hasPunctuation(token),
        'token.wordPattern=' + shape,
        'token.wordPatternSum=' + getSummarizedWordPattern(shape),
        #set E 
        'token[-3:]=' + token[-3:],
        'token[-2:]=' + token[-2:],
        #set F
        'postag=' + postag,
    ]

    if i > 0 and len(str(doc[i-1][0])) > 0:
        token1 = str(doc[i-1][0])
        postag1 = str(doc[i-1][1])
        shape1 = doc[i-1][2]
        features.extend([
            '-1:token=' + token1,                
            "-1:Bigram=" + token1+" "+token,
            "-1:UnigramFreq=" + str(Counter(tokens)[token1]),
            
            '-1:token.isChar=%s' % (len(token1)==1),
            
            '-1:token.isupper=%s' % token1.isupper(),
            '-1:token.islower=%s' % token1.islower(),
            '-1:token.isNonInitCapitals=%s' % ((len([l for l in token1[1:] if l.isupper()]) > 0) and token1[0].islower()),
            '-1:token.isFirstLetterCapital=%s' % token1[0].isupper(),
            
            '-1:token.isalpha=%s' % token1.isalpha(),
            '-1:token.isdigit=%s' % token1.isdigit(),
            '-1:token.isOperator=%s' % (token1 in operators),
            '-1:token.isdelim=%s' % (token1 in delim),
            '-1:token.isMixofLettersDigits=%s' % (hasNumbers(token1) and hasLetters(token1)),
            '-1:token.hasPunctuation=%s' % hasPunctuation(token1),
            '-1:token.wordPattern=' + shape1,
            '-1:token.wordPatternSum=' + getSummarizedWordPattern(shape1),
                           
            '-1:token[-3:]=' + token1[-3:],
            '-1:token[-2:]=' + token1[-2:],
           
            '-1:postag=' + postag1,
        ])

    if i > 1 and len(str(doc[i-2][0])) > 0:
        token1 = str(doc[i-1][0])
        token2 = str(doc[i-2][0])
        postag2 = str(doc[i-2][1])
        shape2 = doc[i-2][2]
        features.extend([
            '-2:token=' + token2,    
            "-2:Bigram=" + token2+" "+token1,
            "-2:UnigramFreq=" + str(Counter(tokens)[token2]),
            
            '-2:token.isChar=%s' % (len(token2)==1),
            
            '-2:token.isupper=%s' % token2.isupper(),
            '-2:token.islower=%s' % token2.islower(),
            '-2:token.isNonInitCapitals=%s' % ((len([l for l in token2[1:] if l.isupper()]) > 0) and token2[0].islower()),
            '-2:token.isFirstLetterCapital=%s' % token2[0].isupper(),
            
            '-2:token.isalpha=%s' % token2.isalpha(),
            '-2:token.isdigit=%s' % token2.isdigit(),
            '-2:token.isOperator=%s' % (token2 in operators),
            '-2:token.isdelim=%s' % (token2 in delim),
            '-2:token.isMixofLettersDigits=%s' % (hasNumbers(token2) and hasLetters(token2)),
            '-2:token.hasPunctuation=%s' % hasPunctuation(token2),
            '-2:token.wordPattern=' + shape2,
            '-2:token.wordPatternSum=' + getSummarizedWordPattern(shape2),
                            
            '-2:token[-3:]=' + token2[-3:],
            '-2:token[-2:]=' + token2[-2:],
            
            '-2:postag=' + postag2,
                
          
        ])
        
    if i < len(doc)-1 and len(str(doc[i+1][0])) > 0:
        token1 = str(doc[i+1][0])
        postag1 = str(doc[i+1][1])
        shape1 = doc[i+1][2]
        features.extend([   
            "+1:Bigram=" + token +" "+token1,
            '+1:token=' + token1,
            "+1:UnigramFreq=" + str(Counter(tokens)[token1]),
            
            '+1:token.isChar=%s' % (len(token1)==1),
            '+1:token.isupper=%s' % token1.isupper(),
            '+1:token.islower=%s' % token1.islower(),
            '+1:token.isNonInitCapitals=%s' % ((len([l for l in token1[1:] if l.isupper()]) > 0) and token1[0].islower()),
            '+1:token.isFirstLetterCapital=%s' % token1[0].isupper(),
            
            '+1:token.isalpha=%s' % token1.isalpha(),
            '+1:token.isdigit=%s' % token1.isdigit(),
            '+1:token.isOperator=%s' % (token1 in operators),
            '+1:token.isdelim=%s' % (token1 in delim),
            '+1:token.isMixofLettersDigits=%s' % (hasNumbers(token1) and hasLetters(token1)),
            '+1:token.hasPunctuation=%s' % hasPunctuation(token1),
            '+1:token.wordPattern=' + shape1,
            '+1:token.wordPatternSum=' + getSummarizedWordPattern(shape1),
                            
            '+1:token[-3:]=' + token1[-3:],
            '+1:token[-2:]=' + token1[-2:],
            
            '+1:postag=' + postag1,
        ])

    if i < len(doc)-2 and len(str(doc[i+2][0])) > 0:
        token1 = str(doc[i+1][0])
        token2 = str(doc[i+2][0])
        shape2 = doc[i+2][2]
        postag2 = str(doc[i+2][1])
        features.extend([
            '+2:token=' + token2,    
            "+2:UnigramFreq=" + str(Counter(tokens)[token2]),
            "+2:Bigram=" + token1 +" "+token2,
            
            '+2:token.isChar=%s' % (len(token2)==1),
            
            '+2:token.isupper=%s' % token2.isupper(),
            '+2:token.islower=%s' % token2.islower(),
            '+2:token.isNonInitCapitals=%s' % ((len([l for l in token2[1:] if l.isupper()]) > 0) and token2[0].islower()),
            '+2:token.isFirstLetterCapital=%s' % token2[0].isupper(),
            
            '+2:token.isalpha=%s' % token2.isalpha(),
            '+2:token.isdigit=%s' % token2.isdigit(),
            '+2:token.isOperator=%s' % (token2 in operators),
            '+2:token.isdelim=%s' % (token2 in delim),
            '+2:token.isMixofLettersDigits=%s' % (hasNumbers(token2) and hasLetters(token2)),
            '+2:token.hasPunctuation=%s' % hasPunctuation(token2),
            '+2:token.wordPattern=' + shape2,
            '+2:token.wordPatternSum=' + getSummarizedWordPattern(shape2),
                            
            '+2:token[-3:]=' + token2[-3:],
            '+2:token[-2:]=' + token2[-2:],
            
            '+2:postag=' + postag2,
          
        ])
    return features


