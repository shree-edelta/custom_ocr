# import string
# import pickle

# # Define character set (can include punctuation or special characters)
# vocab = list(string.ascii_lowercase + string.ascii_uppercase + "0123456789")

# # Assign unique integer to each character (0 for CTC blank label)
# char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # 0 reserved
# idx2char = {idx: char for char, idx in char2idx.items()}
# print(char2idx)
# # print(idx2char)
# # Save tokenizer
# with open("4token.pickle", "wb") as f:
#     token = pickle.dump({"char2idx": char2idx, "idx2char": idx2char}, f)
    
# print(type(token))

import os
import numpy as np
import cv2
import pickle
import pandas as pd
from tensoflow.keras.preprocessing.text import Tokenizer

with open("sentence/new_s_token.pickle", "rb") as f:
    token_data = pickle.load(f)
# char2idx = token_data["char2idx"]
# print(char2idx)

image_folder = "output"  


image_data = []
label_data = []

df = pd.read_csv("output.csv")

for idx, row in df.iterrows():
    filename = str(row["filename"])
    label = str(row["label"])
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: couldn't read {img_path}")
        continue
    if len(label)>0:
        # print(f"detect on {idx}")

        img = cv2.resize(img, (256, 64))

        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=-1)
        
        # tokenizer = Tokenizer(char_level = True)
        # tokenizer.fit_on_texts(label)
        tokenized = [token_data.get(c, 0) for c in label]  

        image_data.append(img)
        label_data.append(tokenized)

image_data = np.array(image_data, dtype=np.float32)
label_data = np.array(label_data, dtype=object) 

print(len(image_data))
print(label_data)
print(len(label_data))

np.save("ocr_test_unseen.npy", {"image": image_data, "label": label_data})
