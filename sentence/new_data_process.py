import numpy as np
from PIL import Image
import pandas as pd
import cv2
import pickle
from sklearn.model_selection import train_test_split

image_data = []
label_data = []

with open("new_s_token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
print(tokenizer)

df = pd.read_csv("train.csv")
print(len(df))
for idx, row in df.iterrows():
    filename = str(row["path"])
    # print(filename)
    label = str(row["word"])
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: couldn't read {filename}")
        continue
    if len(label)>0:
        # print(f"detect on {idx}")

        img = cv2.resize(img, (256, 64))

        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=-1)

        tokenized = [tokenizer.get(c, 0) for c in label]  

        image_data.append(img)
        label_data.append(tokenized)

image_data = np.array(image_data, dtype=np.float32)
label_data = np.array(label_data, dtype=object) 

print(image_data[:5])
print(label_data[:5])
print(len(image_data))
np.save("new_ocr_test.npy", {"image": image_data, "label": label_data})
