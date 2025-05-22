import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
# from check_custom_block import SEBlock
from jiwer import wer, cer
import gc
import pickle
from tensorflow.keras import layers, Model

K.clear_session()
gc.collect()


with open("new_s_token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
    

index_to_char = {value: key for key, value in tokenizer.items()}

print(index_to_char)

def clean_decoded_output(decoded_seq, index_to_char):
    texts = []
    for seq in decoded_seq:
        # print(seq)
        prev = -1
        text = ''
        for char_idx in seq:
            if char_idx != -1 and char_idx != prev:
                text += index_to_char.get(char_idx, '')
            prev = char_idx
        texts.append(text)
    return texts

data = np.load("ocr_test_unseen.npy", allow_pickle=True).item()
images = data['image']
label = data['label']
test_images = np.expand_dims(images, axis=-1).astype(np.float32) 
print("Test image shape:", images.shape)


labels = []
labels.append(clean_decoded_output(label,index_to_char))
# print("labels....",labels)

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

batch_size = 32
predictions = inference_model.predict(images, batch_size=batch_size)
input_length = np.ones(predictions.shape[0]) * predictions.shape[1]

decoded, _ = tf.keras.backend.ctc_decode(
    predictions,
    input_length=input_length,
    greedy=False, beam_width=10
)
decoded = decoded[0].numpy()
# print("Decoded:", decoded)


decoded_text = []
texts   = clean_decoded_output(decoded, index_to_char)
decoded_text.append(texts)
# print(decoded_text)

for i in range(10):
    print(f"Prediction {i+1}: {decoded_text[0][i]} (True: {labels[0][i]})")

df_out = pd.DataFrame({"Prediction": decoded_text})
df_out.to_csv("new_test_predictions.csv", index=False)
print("Saved predictions to ocr_test_predictions.csv")

def evaluate_cer(true_labels, predicted_labels):
    cer_scores = [cer(true, pred) for true, pred in zip(true_labels, predicted_labels)]
    return np.mean(cer_scores)

def evaluate_wer(true_labels, predicted_labels):
    wer_scores = [wer(true, pred) for true, pred in zip(true_labels, predicted_labels)]
    return np.mean(wer_scores)

cer_score = evaluate_cer(labels[0], decoded_text[0])
wer_score = evaluate_wer(labels[0], decoded_text[0])
accuracy = np.mean([true == pred for true, pred in zip(labels[0], decoded_text[0])])

print(f"Character Error Rate (CER): {cer_score:.4f}")
print(f"Word Error Rate (WER): {wer_score:.4f}")
print(f"OCR Model Accuracy: {accuracy:.4f}")