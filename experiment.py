import datetime
import argparse
from os import path

import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping

from data import read_train_data
from models import get_model, CustomTensorBoard
from utils import train_test_many_split, save_predictions

# Experiment parameters
mass_metric = 'M500c'
epochs = 400
patience = 100
batch_size = 64
learning_rate = 0.0001
n_img = 20000  # TODO: read all images

# Get folder name for tensorboard logging
timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='experiment tag, added to logs name')
args = parser.parse_args()
log_folder = '{tag}, lr={lr}, bs={bs}, {ts}'.format(ts=timestamp_start.replace('_', ' '), tag=args.tag,
                                                    lr=learning_rate, bs=batch_size)
log_dir = path.join('tensorboard', log_folder)


# Read and split data
# data, labels, folds = read_tng_data()
X, y = read_train_data('/users/snakoneczny/data/lensing_takahashi_17/cmb_lens_imgs', n_img=n_img, col_y=mass_metric)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# TODO: sample equal distribution on mass

# TODO
cmb_min = X.min()
cmb_max = X.max()
X = (X - cmb_min) / (cmb_max - cmb_min)

y_train, y_test_low, y_test_high, y_test_random, X_train, X_test_low, X_test_high, X_test_random = \
    train_test_many_split(y, X, side_test_size=0.05, random_test_size=0.1)

# TODO: image data generator

# Train model
model = get_model(input_shape=X[0].shape, lr=learning_rate)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    CustomTensorBoard(log_dir=log_dir,
                      validation_data={'low': (X_test_low, y_test_low), 'high': (X_test_high, y_test_high)}),
]
model.fit(X_train, y_train, validation_data=(X_test_random, y_test_random), batch_size=batch_size, epochs=epochs,
          callbacks=callbacks)

# TODO: print best results on validation data

y_pred = model.predict(X_test_random)
predictions = pd.DataFrame({mass_metric: y_test_random, 'm_pred': y_pred[:, 0]})
save_predictions(predictions, args.tag, timestamp_start)
