import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from check_custom_block import SEBlock, ctc_loss_lambda
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import pandas as pd
import pickle
import os
def load_data(filepath):
    data = np.load(filepath,allow_pickle=True).item()
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
    # max_label_len = 27
    padded_labels = pad_sequences(valid_labels,padding="post", value=0)
    # print(padded_labels)
    f_labels = np.array(padded_labels, dtype=np.int32)
    f_images = np.array(valid_images)

    # print(f"Valid images shape: {f_images.shape}")
    # print(f"Valid labels shape: {f_labels.shape}")

    return f_images, f_labels

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

with open("4token.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
# print(tokenizer)
vocab_size = len(tokenizer['char2idx'])+1
print(vocab_size)


# def load_data(file_path, batch_size=32):
#     data = np.load(file_path)
#     images = data['images']  
#     labels = data['labels']
#     print("load data.")
#     for i in range(0, len(images), batch_size):
#         batch_images = images[i:i+batch_size].astype(np.float32) 
#         batch_images = np.expand_dims(batch_images, axis=-1)
#         yield batch_images, labels[i:i+batch_size]


ti,tl = load_data("ocr_train.npy")

cleaned_tl = [
    [lbl if isinstance(lbl, (list, np.ndarray)) else [lbl] for lbl in batch]
    for batch in tl
]
max_label_len = max(len(labels) for labels in tl )

# max_label_len = max(len(lbl) for batch_labels in cleaned_tl for lbl in batch_labels)
# Conv2D(64, (3,3), padding='same', activation='relu')  
# BatchNormalization()
# MaxPooling2D(pool_size=(2,2))

# Conv2D(128, (3,3), padding='same', activation='relu')
# BatchNormalization()
# MaxPooling2D(pool_size=(2,2))

# Conv2D(256, (3,3), padding='same', activation='relu')
# BatchNormalization()
# Conv2D(256, (3,3), padding='same', activation='relu')
# MaxPooling2D(pool_size=(2,1))  # Preserve time dimension

# Conv2D(512, (3,3), padding='same', activation='relu')
# BatchNormalization()
# MaxPooling2D(pool_size=(2,1))

# Conv2D(512, (2,2), padding='valid', activation='relu')

# print("max_number",max_label_len)
input_shape = (64, 256,1)
inputs = layers.Input(shape=input_shape, name="image")

# x = layers.Embedding(input_dim = vocab_size,output_dim = 256)(inputs)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = SEBlock(64)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = SEBlock(128)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = SEBlock(256)(x)
x = layers.MaxPooling2D(pool_size=(2, 1))(x)

x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = SEBlock(512)(x)
x = layers.MaxPooling2D(pool_size=(2, 1))(x)

x = layers.Conv2D(512, (2, 2), activation="relu", padding="valid")(x)

# x = layers.Permute((2, 1, 3))(x)  
# x = layers.TimeDistributed(layers.Flatten())(x)
# x = layers.Reshape(target_shape=(-1, 128))(x)
# (None, H, W, C)
# reshaped = Reshape(target_shape=(W, H * C))(cnn_output)

x = layers.Reshape((-1,x.shape[-2], x.shape[-3] * x.shape[-1]))(x)

x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

num_classes = vocab_size

outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

labels_input = layers.Input(name="labels", shape=(max_label_len,), dtype="int32")
input_length = layers.Input(name="input_length", shape=(1,), dtype="int32")
label_length = layers.Input(name="label_length", shape=(1,), dtype="int32")

loss_output = layers.Lambda(ctc_loss_lambda, output_shape=(1,), name="ctc_loss")(
    [outputs, labels_input, input_length, label_length]
)


model = Model(inputs=[inputs, labels_input, input_length, label_length], outputs=loss_output)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,clipnorm=1.0)

model.compile(optimizer=optimizer, loss=lambda y_true, y_pred:y_pred)
print(model.summary())
# def data_generator(file_path, batch_size=32):
#     images, labels = load_data(file_path)
#     num_samples = len(images)

#     for i in range(0, num_samples, batch_size):
#         batch_images = images[i:i+batch_size].astype(np.float32)
#         batch_labels = labels[i:i+batch_size]

#         # Filter out samples with empty labels
#         valid_images = []
#         valid_labels = []

#         for img, lbl in zip(batch_images, batch_labels):
#             actual_label = lbl[lbl != 0]  # Remove padding zeros
#             if len(actual_label) > 0:
#                 valid_images.append(img)
#                 valid_labels.append(lbl)

#         if len(valid_images) == 0:
#             print(f"Skipping batch at index {i} due to all empty labels")
#             continue  # skip this batch

#         batch_images = np.array(valid_images, dtype=np.float32)
#         batch_labels = np.array(valid_labels)

#         time_steps = model.get_layer("predictions").output.shape[1]
#         input_length = np.full((len(batch_images), 1), time_steps, dtype=np.int32)

#         label_length = np.array([[len(lbl[lbl != 0])] for lbl in batch_labels], dtype=np.int32)

#         yield (
#             batch_images,
#             batch_labels,
#             input_length,
#             label_length
#         ), np.zeros(len(batch_images), dtype=np.float32)
        
def data_generator(file_path, batch_size=32):
    images, labels = load_data(file_path)
    num_samples = len(images)

    while True:
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i+batch_size].astype(np.float32)
            batch_images = np.expand_dims(batch_images, axis=-1)  # (batch_size, 64, 256, 1)
            batch_labels = labels[i:i+batch_size]

            valid_images = []
            valid_labels = []

            for img, lbl in zip(batch_images, batch_labels):
                
                actual_label = lbl[lbl != 0]  # Remove padding zeros
                if len(actual_label) > 0:
                    valid_images.append(img)
                    valid_labels.append(lbl)

            if len(valid_images) == 0:
                continue

            batch_images = np.array(valid_images, dtype=np.float32)
            batch_labels = np.array(valid_labels, dtype=np.int32)

            time_steps = model.get_layer("predictions").output.shape[1]
            input_length = np.full((len(batch_images), 1), time_steps, dtype=np.int32)
            label_length = np.array([[np.count_nonzero(lbl)] for lbl in batch_labels], dtype=np.int32)

            # yield (
            #     {
            #         "image": batch_images,
            #         "labels": batch_labels,
            #         "input_length": input_length,
            #         "label_length": label_length
            #     },
            #     np.zeros(len(batch_images), dtype=np.float32)  # Dummy y_true for loss
            # )
            yield (batch_images.astype(np.float32), batch_labels, input_length, label_length), labels

batch_size =32
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("ocr_train.npy", batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 64, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_label_len), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
).prefetch(tf.data.AUTOTUNE).repeat()
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


def filter_empty_labels(image, label):
    return len(label) > 0

train_dataset_r = train_dataset.filter(lambda img, label: tf.shape(label)[0] > 0)
# val_dataset_r = val_dataset.filter(lambda img, label: tf.shape(label)[0] > 0)


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
    

# tracker_cb = get_tracker_callback()

steps_per_epoch = len(tl) //batch_size

# validation_steps = len(v_label) // batch_size

history_data  =  model.fit(train_dataset_r, epochs=20,steps_per_epoch=8,callbacks=callbacks)


model.save("final_ocr_model.keras")

hist_df = pd.DataFrame(history_data.history)
hist_df['epoch'] = range(1,len(hist_df)+1)
save_history(hist_df, "train_metrics.csv", "training_plot.png")

print(model.summary()) 

