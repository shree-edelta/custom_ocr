import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

def save_token(tokenizer,filename = "new_s_token.pickle"):
    with open(filename,'wb') as handle:
        pickle.dump(tokenizer,handle)
        
vocab = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"') + ["'"]+['[PAD]', '[UNK]']
tokenizer = {char: idx for idx, char in enumerate(vocab)}
print(tokenizer)
# tokenizer = {char: idx for idx, char in enumerate(char_list)}
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(tokenizer)

save_token(tokenizer)

with open("new_s_token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
print(tokenizer)