
import numpy as np
import tensorflow as tf
# from check_data_npy import SEBlock
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
# from check_custom_block import SEBlock,ctc_loss_lambda
from tensorflow.keras.models import load_model
from PIL import Image,ImageChops
import pickle
import matplotlib.pyplot as plt
import cv2

@tf.keras.utils.register_keras_serializable()
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

inference_model = load_model("new_data_inference.keras", custom_objects={"SEBlock": se_block})
print(inference_model.summary())

with open("new_s_token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
    
index_to_char = {value: key for key, value in tokenizer.items()}

print(index_to_char)

def clean(image, image_shape=(64, 256)):
    # img = image.resize(image_shape[::-1]) 
    img = cv2.imread(image)
    
    print("type img :",img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)

    # Morphological operation to remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Subtract the lines from the original thresholded image
    no_lines = cv2.subtract(thresh, detected_lines)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_lines, connectivity=8)
    min_area = 250  # area threshold for keeping components (adjust as needed)

    mask = np.zeros(no_lines.shape, dtype="uint8")
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255


    # Invert back for better visualization
    result = cv2.bitwise_not(mask)

    # Optional: Crop white space
    coords = cv2.findNonZero(255 - result)  # get non-white pixel coords
    x, y, w, h = cv2.boundingRect(coords)
    cropped = result[y:y+h, x:x+w]
    
    return cropped

def convert_to_black_white_pil(image_path, threshold=128):
    img = Image.open(image_path).convert("L") 
    bw = img.point(lambda x: 255 if x > threshold else 0)
    
    return bw.convert("L")
# s = clean("../images/shree_0.jpg")
# print(type(s))

def preprocess_image(image_path,image_shape=(64, 256)):
    image = clean(image_path)
    image = cv2.resize(image, image_shape[::-1])
    # image = cv2.resize(image, image_shape[::-1])
    print("after clean",image.shape)
    # image = Image.open(image_path).convert("L")
    # img = image.resize(image_shape[::-1]) 
    img_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(img_array, axis=(0, -1))
    print(img_array.shape)
    # img.save("pred2.png")
    return image_array
    
    
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
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


img1 = preprocess_image("../images/20250520_130200.jpg")
print(img1.shape)


pred1 = inference_model.predict(img1)

print("Pred1 (logits):", pred1[0][:5])  

print(pred1.shape)
input_length1 = np.array([pred1.shape[1]], dtype=np.int32)
print(input_length1)

decoded1, _ = tf.keras.backend.ctc_decode(pred1, input_length=input_length1, greedy=False, beam_width=15)
decoded1 = decoded1[0].numpy()
print("Decoded:", decoded1)

texts1 = clean_decoded_output(decoded1, index_to_char)
print("Final decoded texts:", texts1)


