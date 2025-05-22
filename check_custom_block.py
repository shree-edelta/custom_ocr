import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):  # Accept extra arguments
        super(SEBlock, self).__init__(**kwargs)  # Pass them to Layer
        self.channels = channels
        self.reduction = reduction
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(channels // reduction, activation="relu")
        self.dense2 = tf.keras.layers.Dense(channels, activation="sigmoid")

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        return inputs * x

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config

@tf.keras.utils.register_keras_serializable()
def ctc_loss_lambda(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

