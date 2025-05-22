import numpy as np
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

model = tf.keras.models.load_model(
    'final_ocr_model.keras', 
    custom_objects={"SEBlock": SEBlock,
        "ctc_loss_lambda": ctc_loss_lambda},
    compile=False
)
print(model.get_layer("predictions").get_weights())
print(model.summary())

inference_input = layers.Input(shape=(64, 256, 1), name="image")
print(inference_input)
x = model.get_layer("conv2d").output 
print(x)
inference_model = Model(inputs=model.inputs[0],
                        outputs=model.get_layer("predictions").output)
inference_model.save("inference.keras")

