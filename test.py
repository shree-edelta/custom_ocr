
import numpy as np
import tensorflow as tf
# from check_data_npy import SEBlock
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from check_custom_block import SEBlock,ctc_loss_lambda
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import matplotlib.pyplot as plt

inference_model = load_model("inference.keras", custom_objects={"SEBlock": SEBlock})
print(inference_model.summary())

with open("4token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
   
def convert_to_black_white_pil(image_path, threshold=128):
    img = Image.open(image_path).convert("L") 
    bw = img.point(lambda x: 255 if x > threshold else 0)
    return bw.convert("L")

def preprocess_image(image_path, image_shape=(64, 256)):
    image = convert_to_black_white_pil(image_path)
    img = image.resize(image_shape[::-1]) 
    img_array = np.array(img, dtype=np.float32) / 255.0
    image_array = np.expand_dims(img_array, axis=(0, -1))
    print(img_array.shape)
    img.save("pred.png")
    return image_array
    
    
def clean_decoded_output(decoded_seq, index_to_char):
    texts = []
    for seq in decoded_seq:
        prev = -1
        text = ''
        for char_idx in seq:
            if char_idx != -1 and char_idx != prev:
                text += index_to_char.get(char_idx, '')
            prev = char_idx
        texts.append(text)
    return texts

# image = convert_to_black_white_pil("images/name.jpg")

img1 = preprocess_image("output/name7.jpg")
print(img1.shape)


pred1 = inference_model.predict(img1)

print("Pred1 (logits):", pred1[0][:5])  

print(pred1.shape)
input_length1 = np.array([pred1.shape[1]], dtype=np.int32)
print(input_length1)

decoded1, _ = tf.keras.backend.ctc_decode(pred1, input_length=input_length1, greedy=False, beam_width=10)
decoded1 = decoded1[0].numpy()
print("Decoded:", decoded1)

index_to_char = tokenizer["idx2char"] 
print(index_to_char)
texts1 = clean_decoded_output(decoded1, index_to_char)
print("Final decoded texts:", texts1)


# I2I2G2kYNYNYNYNSNSNSNYNYNY

# I2I2z2Q2kYNYNYNYNSNSNSNYNYNY

# I2I2G2kYNYNYNYNSNSNSNYNYNY

# I2I2G2kYNYNYNYNSNSNSNYNYNY

# I2I2G2kYNYNYNYNSNSNSNYNYNY


