import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
# from check_custom_block import ctc_loss_lambda
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import pandas as pd
import pickle
import os

# def SEBlock(input_tensor, reduction_ratio=16):
#     """Squeeze-and-Excitation block."""
#     channels = input_tensor.shape[-1]  # Number of channels (e.g., 256, 512, etc.)

#     # Squeeze: Global Average Pooling
#     squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    
#     # Fully connected layers (Excitation)
#     excitation = layers.Dense(units=channels // reduction_ratio, activation='relu')(squeeze)
#     excitation = layers.Dense(units=channels, activation='sigmoid')(excitation)

#     # Reshape to match original dimensions for scaling
#     excitation = layers.Reshape((1, 1, channels))(excitation)

#     # Scale the input feature maps (channel-wise multiplication)
#     scale = layers.Multiply()([input_tensor, excitation])

#     return scale

def load_data(filepath):
    data = np.load(filepath,allow_pickle=True).item()
    # data = [sample for sample in data if len(sample['label']) > 0]
    images = data['image']         
    labels = data['label'] 
    
    # print(len(labels))
    # print(len(images))
    # print(images.shape)
    valid_images = []
    valid_labels = []

    for img, lbl in zip(images, labels):
        if len(lbl) > 0:
            valid_images.append(img)
            valid_labels.append(lbl)

    print(f"Filtered {len(images) - len(valid_images)} samples with empty labels.")
    max_label_len = 27
    # padded_labels = pad_sequences(valid_labels, padding='post', value=0)
    padded_labels = pad_sequences(valid_labels, maxlen=max_label_len, padding="post", value=0)
    # print(padded_labels)
    f_labels = np.array(padded_labels, dtype=np.int32)
    f_images = np.array(valid_images)

    # print(f"Valid images shape: {f_images.shape}")
    # print(f"Valid labels shape: {f_labels.shape}")

    return f_images, f_labels
# def load_data(path):
#     data = np.load(path, allow_pickle=True)
#     original_len = len(data)
#     data = [sample for sample in data if sample['label'] is not None and len(sample['label']) > 0]
#     print(f"Removed {original_len - len(data)} entries with empty or None labels")
#     images = [sample['image'] for sample in data]
#     labels = [sample['label'] for sample in data]
#     return images, labels

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

with open("new_s_token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
# print(tokenizer)

vocab_size = len(tokenizer)+1
print(vocab_size)


train_image,train_label = load_data("new_ocr_train.npy")
valid_image,valid_label = load_data("new_ocr_valid.npy")

cleaned_tl = [
    [lbl if isinstance(lbl, (list, np.ndarray)) else [lbl] for lbl in batch]
    for batch in train_label
]
max_label_len = max(len(labels) for labels in train_label )


input_shape = (64, 256, 1)
num_classes = vocab_size  

@tf.autograph.experimental.do_not_convert
def dummy_loss(y_true, y_pred):
    return y_pred

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def dummy_loss(y_true, y_pred):
    return y_pred 

# def ctc_loss_function(y_true, y_pred):
#     return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        
inputs = layers.Input(name='input', shape=input_shape, dtype='float32')

x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(256, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x)
x = layers.MaxPooling2D(pool_size=(2, 1))(x)


new_shape = (x.shape[2], x.shape[1] * x.shape[3])
x = layers.Reshape(target_shape=new_shape)(x)
# x = layers.Reshape(target_shape=(x.shape[1], -1))(x)
# shape = tf.keras.backend.shape(x)
# x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

y_pred = layers.Dense(num_classes, activation='softmax', name='y_pred')(x)

labels = layers.Input(name='labels', shape=(max_label_len,), dtype='int32')
input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

ctc_loss = layers.Lambda(ctc_lambda_func, name='ctc_loss')(
    [y_pred, labels, input_length, label_length]
)

model = Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,clipnorm=1.0)
model.compile(optimizer=optimizer,loss=dummy_loss)

model.summary()


def data_generator(images, labels, batch_size, max_label_len):
    num_samples = len(images)
    time_steps = model.get_layer("y_pred").output.shape[1]

    while True:
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            # input_len = np.ones((batch_size, 1)) * (256 // 32)
            valid_images = []
            valid_labels = []
          
            for img, lbl in zip(batch_images, batch_labels):
                actual_label = lbl[lbl != 0]  # Remove padding zeros
                if len(actual_label) > 0:
                    valid_images.append(img)
                    valid_labels.append(lbl)

            if len(valid_images) == 0:
                print(f"Skipping batch at index {i} due to all empty labels")
                continue  # skip this batch

            batch_images = np.array(valid_images, dtype=np.float32)
            batch_labels = np.array(valid_labels)
            
            current_batch_size = len(batch_images)

            input_len = np.ones((current_batch_size, 1), dtype=np.int32) * time_steps
            label_len = np.array([[np.count_nonzero(lbl)] for lbl in batch_labels], dtype=np.int32)
            # if label_len[i] == 0:
            #     continue

            yield (
                {
                    'input': batch_images,
                    'labels': batch_labels,
                    'input_length': input_len,
                    'label_length': label_len
                },
                np.zeros(current_batch_size)
            )

batch_size =32
#  ignore_longer_outputs_than_inputs=True
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_image, train_label, batch_size, max_label_len),
    output_signature=(
        {
            'input': tf.TensorSpec(shape=(None, 64, 256, 1), dtype=tf.float32),
            'labels': tf.TensorSpec(shape=(None, max_label_len), dtype=tf.int32),
            'input_length': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            'label_length': tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(valid_image, valid_label, batch_size, max_label_len),
    output_signature=(
        {
            'input': tf.TensorSpec(shape=(None, 64, 256, 1), dtype=tf.float32),
            'labels': tf.TensorSpec(shape=(None, max_label_len), dtype=tf.int32),
            'input_length': tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            'label_length': tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)


# train_dataset = tf.data.Dataset.from_generator(
#     lambda: data_generator(train_image,train_label, batch_size),
#     output_signature=(
#         (
#             tf.TensorSpec(shape=(None, 64, 256, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(None, max_label_len), dtype=tf.int32),
#             tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
#             tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
#         ),
#         tf.TensorSpec(shape=(None,), dtype=tf.float32),
#     )
# ).prefetch(tf.data.AUTOTUNE).repeat()
# val_dataset = tf.data.Dataset.from_generator(
#     lambda: data_generator("3half_ocr_validation.npy", batch_size),
#     output_signature=(
#         (
#             tf.TensorSpec(shape=(None, 64, 256, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(None, max_label_len), dtype=tf.int32),
#             tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
#             tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
#         ),
#         tf.TensorSpec(shape=(None,), dtype=tf.float32),
#     )
# ).prefetch(tf.data.AUTOTUNE).repeat()


# train_dataset_r = train_dataset.filter(lambda img, label: tf.shape(label)[0] > 0)
# val_dataset_r = val_dataset.filter(lambda img, label: tf.shape(label)[0] > 0)
train_dataset_r = train_dataset.filter(
    lambda x, y: tf.shape(x['labels'])[0] > 0
)

valid_dataset_r = valid_dataset.filter(
    lambda x, y: tf.shape(x['labels'])[0] > 0
)

history_data = []
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

def save_history(df, csv_path="history.csv", plot_path="loss_plot.png"):
    # df = pd.DataFrame(history_list)
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))

    if 'loss' in df.columns:
        plt.plot(df['epoch'], df['loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    if 'accuracy' in df.columns:
        plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in df.columns:
        plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy')
    if 'lr' in df.columns:
        plt.plot(df['epoch'], df['lr'], label='Learning Rate', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
print(model.summary())
# tracker_cb = get_tracker_callback()

steps_per_epoch = len(train_label) //batch_size
validation_steps = len(valid_label) // batch_size
# print(steps_per_epoch,validation_steps)
# print("step per epoch",steps_per_epoch)

history_data  =  model.fit(train_dataset_r, validation_data=valid_dataset_r, epochs=50,steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,callbacks=callbacks)

model.save("new_data_model.keras")

hist_df = pd.DataFrame(history_data.history)
hist_df['epoch'] = range(1,len(hist_df)+1)
save_history(hist_df, "train_metrics.csv", "training_plot.png")

print(model.summary()) 





#  x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)  # 64x256 → 32x128

#     # CNN Block 2
#     x = Conv2D(128, (3, 3), padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)  # 32x128 → 16x64

#     # CNN Block 3
#     x = Conv2D(256, (3, 3), padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 1))(x)  # 16x64 → 8x64 (keep width high for time steps)

#     # Reshape for RNN input: (batch, time, features)
#     new_shape = K.int_shape(x)  # (None, 8, 64, 256)
#     x = Reshape((new_shape[2], new_shape[1] * new_shape[3]))(x)  # (batch, 64, 2048)

#     # RNN Layers
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)

#     # Final Dense Layer
#     outputs = Dense(num_classes, activation='softmax', name='output')(x)
