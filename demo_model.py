'''
Created on 21 dic. 2018

@author: rodrigo
'''
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import pickle

def reverse(string):
    l = list(string)
    l.reverse()
    return "".join(l)

def load_dataset():
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(current_path + '/analisis_verbos.txt','r') as f:
        list_of_lines = f.readlines()
    inputs = []
    outputs = []
    for line in list_of_lines:
        x = line.split()
        inputs.append(x[0]+"9")
        outputs.append( x[1]+"9")
    return inputs, outputs

def use_model(model, tokenizer, word, part_of_lemma):
    word = reverse(word)
    
    part_of_lemma = reverse(part_of_lemma)
    
    tokenized_word = tokenizer.texts_to_sequences([list(word)])[0]
    tokenized_word = [0] * (15-len(tokenized_word)) + tokenized_word
    tokenized_lemma = tokenizer.texts_to_sequences([list(part_of_lemma)])[0]
    tokenized_lemma = [0] * (15-len(tokenized_lemma)) + tokenized_lemma
    
    tokenized_word = np.asarray([tokenized_word])
    tokenized_lemma = np.asarray([tokenized_lemma])
    
    predictions = model.predict([tokenized_word, tokenized_lemma])
    
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    
    return reverse_word_map[np.argmax(predictions)]


def load_model():
    with open(os.path.dirname(os.path.abspath(__file__))+"/lemasnn.json",'r') as json_file:
        json_model = json_file.read()
    
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights(os.path.dirname(os.path.abspath(__file__))+"/lemasnn.hd5")
    
    return loaded_model

def load_model2():
    with open(os.path.dirname(os.path.abspath(__file__))+"/lemasnn.pkl",'rb') as f:
        loaded_model = pickle.load(f)
    
    return loaded_model

def predict_lemma(model,
                  tokenizer,
                  verb):
    flag = True
    lemma = ""
    while(flag):
        try:
            c = use_model(model, tokenizer, verb, lemma)
            if c != '9':
                lemma = lemma + c
            else:
                flag = False 
            if(len(lemma)==15):
                flag = False
        except:
            flag = False
    return lemma
        
pretrained_model = load_model2()
all_letters = [chr(i) for i in range(ord('a'),ord('z')+1)]
all_letters = all_letters + ['á','é','í','ó','ú','ñ','9']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_letters)

list_of_verbs = ["quise", "canté", "murió", "romperé", "casado", "matamos","escribido","amemos","quisimos"]
for v in list_of_verbs:
    print("Lemma of " + v +": " + predict_lemma(pretrained_model, tokenizer, v))
    
errors = 0
verbs, lemmas = load_dataset()
for i in range(len(verbs)):
    if predict_lemma(pretrained_model, tokenizer, verbs[i]) != lemmas[i][:-1]:
        errors = errors + 1

print("Errors: " + str(errors/float(len(verbs)*100)))
print("OK: " + str(100-errors/float(len(verbs)*100)))
print("Number of errors: " + str(errors))
if __name__ == '__main__':
    pass