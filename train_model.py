'''
Created on 21 dic. 2018

@author: rodrigo
'''

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Add
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import os
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle


def reverse(string):
    l = list(string)
    l.reverse()
    return "".join(l)



def string_to_ints(string):
    return [ord(c)-ord('a') for c in string.lower().split()]
    
def load_dataset():
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(current_path + 'data/analisis_verbos.txt','r') as f:
        list_of_lines = f.readlines()
    inputs = []
    outputs = []
    for line in list_of_lines:
        x = line.split()
        inputs.append(x[0]+"9")
        outputs.append( x[1]+"9")
    return inputs, outputs

def divide_by_token_in_lemma(words, lemmas):
    assert(len(words) == len(lemmas))
    
    inputs_1 = []
    inputs_2 = []
    outputs =  []
    for i in range(len(words)):
        for j in range(len(list(lemmas[i]))):
            inputs_1.append(reverse(words[i].lower()))
            inputs_2.append(reverse(lemmas[i][:j].lower()))
            outputs.append(lemmas[i][j].lower())
    return inputs_1, inputs_2, outputs
            


    
#########################################################
# DATASET PROCESSING
########################################################
inputs, outputs = load_dataset()
print("Number of words : " + str(len(inputs)))
print("Unique conjugated verbs: " + str(len(list(set(inputs)))))
print("Unique lemmas: " + str(len(list(set(outputs)))))

inputs_1, inputs_2, outputs = divide_by_token_in_lemma(inputs, outputs)

print("Length dataset : " + str(len(inputs_1)))

all_letters = [chr(i) for i in range(ord('a'),ord('z')+1)]
all_letters = all_letters + ['á','é','í','ó','ú','ñ',"ü","9"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_letters)

tokenized_inputs_1 = [tokenizer.texts_to_sequences([list(x)])[0] for x in inputs_1]
tokenized_inputs_2 = [tokenizer.texts_to_sequences([list(x)])[0] for x in inputs_2]
tokenized_outputs = [tokenizer.texts_to_sequences([list(x)])[0] for x in outputs]


onehot_encoded_output = [to_categorical(x)[0] for x in tokenized_outputs]

tokenized_inputs_1 = pad_sequences(tokenized_inputs_1)
tokenized_inputs_2 = pad_sequences(tokenized_inputs_2)
onehot_encoded_output = pad_sequences(onehot_encoded_output, padding='post')

#########################################################
# NETWORK CREATION
########################################################
NUM_CHARS = len(all_letters) +1 # Due to padding
DIM_EMBEDD = 15



# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(NUM_CHARS, DIM_EMBEDD)(encoder_inputs)
x_left, state_h_left, state_c_left = LSTM(DIM_EMBEDD,
                           return_state=True, go_backwards=False)(x)

x_right, state_h_right, state_c_right = LSTM(DIM_EMBEDD,
                           return_state=True, go_backwards=True)(x)


state_h = Add()([state_h_right,state_h_left])
state_c = Add()([state_c_right,state_c_left])

encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(NUM_CHARS, DIM_EMBEDD)(decoder_inputs)
x = LSTM(DIM_EMBEDD, return_sequences=False)(x, initial_state=encoder_states)
decoder_outputs = Dense(NUM_CHARS, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


#########################################################
# TRAINING
########################################################
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
# decoder_target_data is ahead of decoder_input_data by one timestep


BATCH_SIZE = 64
EPOCHS = 1
VALIDATION_SPLIT = 0.2

tokenized_inputs_1 = np.array(tokenized_inputs_1)
tokenized_inputs_2 = np.array(tokenized_inputs_2)
onehot_encoded_output = np.array(onehot_encoded_output)

model.fit([tokenized_inputs_1, tokenized_inputs_2], 
          onehot_encoded_output,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          validation_split=VALIDATION_SPLIT)

#########################################################
# SAVE MODEL
########################################################

NAME_MODEL = "lemmasn_n.pkl"
with open(os.path.dirname(os.path.abspath(__file__))+ "/" + NAME_MODEL,'wb') as f:
    pickle.dump(model,f)
    