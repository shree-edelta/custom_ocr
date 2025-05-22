import numpy as np
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
from PIL import Image
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

@tf.keras.utils.register_keras_serializable()
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

model = tf.keras.models.load_model(
    'new_data_model.keras', 
    custom_objects={"SEBlock": se_block,
        "ctc_loss_lambda": ctc_lambda_func},
    compile=False
)
print(model.get_layer("y_pred").get_weights())
print(model.summary())

inference_input = layers.Input(shape=(64, 256, 1), name="image")
print(inference_input)
x = model.get_layer("conv2d").output 
print(x)
inference_model = Model(inputs=model.inputs[0],
                        outputs=model.get_layer("y_pred").output)
inference_model.save("new_data_inference.keras")
