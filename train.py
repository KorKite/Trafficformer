import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Importing Libraries
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from model import transformer
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename", type=str, default ="example.traff", help="data directory"
)
parser.add_argument(
    "--n_days", type=int, help="data directory"
)
parser.add_argument(
    "--save_name", type=str, help="data directory"
)

args = parser.parse_args()
DATA_PATH = args.filename
N_DAYS = args.n_days
SAVE_NAME = args.save_name

def make_dataset(data, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data[i:i+window_size]))
        label_list.append(np.array(data[i+window_size]))
    return np.array(feature_list).reshape(-1, window_size, 1), np.array(label_list)


data = []
with open(f"data/{DATA_PATH}", "r") as f:
    for d in f.readlines():
        data.append(int(d))
data = np.array(data)

train_x, train_y = make_dataset(data)
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2)

model = transformer(
    input_shape=[N_DAYS, 1],
    head_size=64,
    num_heads=3,
    ff_dim=3,
    num_transformer_blocks=3,
    mlp_units=[64],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"model/{SAVE_NAME}.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
            epochs=200, 
            batch_size=32,
            validation_data=(x_valid, y_valid), 
            callbacks=[early_stop, checkpoint]
            )
