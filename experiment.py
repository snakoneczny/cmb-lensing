import datetime
import argparse
import resource
from os import path

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

from data import read_train_data
from models import get_model

# Experiment parameters
epochs = 400
patience = 100
batch_size = 64
learning_rate = 0.0001
n_rows = 20000

# Get folder name for tensorboard logging
timestamp_start = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='experiment tag, added to logs name')
args = parser.parse_args()
log_folder = '{ts}, {tag}, lr={lr}, bs={bs}'.format(ts=timestamp_start.replace('_', ' '), tag=args.tag,
                                                    lr=learning_rate, bs=batch_size)
log_dir = path.join('tensorboard', log_folder)

# MacOS requirements for number of open files
# TODO: Read all images available
resource.setrlimit(resource.RLIMIT_NOFILE, (n_rows + 1000, -1))
# Read and split data
# data, labels, folds = read_tng_data()
X, y = read_train_data('/users/snakoneczny/data/lensing_takahashi_17/cmb_lens_imgs', n_rows=n_rows)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
# TODO: lowest mass (5%), highest mass (5%) and random from the rest (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8725)

# Get model
input_shape = X[0].shape
model = get_model(input_shape, lr=learning_rate)

# Train model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    TensorBoard(log_dir=log_dir),
]
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=callbacks)

# TODO: print best results on validation data

# TODO: save predictions
