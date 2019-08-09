import datetime
import argparse
from os import path

import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import multi_gpu_model

from data import read_train_data, get_flat_mass_dist
from models import get_model, CustomTensorBoard
from utils import DATA_DIR_RELATIVE, save_predictions, get_tensorboard_dir
from evaluation import train_test_many_split

# Timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='experiment tag, added to logs name')
parser.add_argument('--test', dest='is_test', action='store_true', help='flag for making a quick test')
args = parser.parse_args()

# Experiment parameters
# TODO: read all images, read more when limiting mass
# train_files_path = path.join(DATA_DIR_RELATIVE, 'file_base/cut_nres13_n054_flat_nbins-100_binsize-100.csv')
# train_data_path = path.join(DATA_DIR_RELATIVE, 'cut_nres13_r054_fits_org')
data_file_name = 'train_base/halo_nres13_r000_M200b-min-1e14_equator-pi-9_flat-mass-100-100.csv'
images_file_name = 'train_base/imgs_nres13_r000_M200b-min-1e14_equator-pi-9_flat-mass-100-100.npy'
data_path = path.join(DATA_DIR_RELATIVE, data_file_name)
images_path = path.join(DATA_DIR_RELATIVE, images_file_name)

mass_metric = 'M500c'
batch_size = 64
learning_rate = 0.0001
if not args.is_test:
    epochs = 20000
    patience = 600
    n_img = None
else:
    epochs = 400
    patience = 10
    n_img = 200

# Read, sample, reshape and split data
# X, y = read_train_data(train_data_path, n_img=n_img, col_y=mass_metric,
#                        file_list_path=train_files_path, image_size=20)
# X, y = get_flat_mass_dist(X, y, n_bins=100, max_bin_size=100)

data = pd.read_csv(data_path)
X = np.load(images_path)
y = data[mass_metric].values

y = np.log(y)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

y_train, y_test_low, y_test_high, y_test_random, X_train, X_test_low, X_test_high, X_test_random = \
    train_test_many_split(y, X, side_test_size=0.05, random_test_size=0.1)

# Get and train data generator
# TODO: only solid angle rotations
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    # rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(X_train)
# Standardize validation datasets
X_test_random = datagen.standardize(X_test_random)
X_test_high = datagen.standardize(X_test_high)
X_test_low = datagen.standardize(X_test_low)

# Train model
# TODO: send some images to tensorboard
callbacks = [
    CustomTensorBoard(log_dir=get_tensorboard_dir(args, timestamp, learning_rate, batch_size),
                      validation_data={'low': (X_test_low, y_test_low), 'high': (X_test_high, y_test_high)}),
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
]
model = get_model(input_shape=X[0].shape, lr=learning_rate)
# model = multi_gpu_model(model, gpus=4)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) / batch_size,
                    validation_data=(X_test_random, y_test_random), epochs=epochs, callbacks=callbacks)

# Predict and save
to_pred = [
    ('random', X_test_random, y_test_random),
    ('low', X_test_low, y_test_low),
    ('high', X_test_high, y_test_high),
]
predictions = pd.DataFrame()
for test_name, X_test, y_test in to_pred:
    y_pred = model.predict(X_test)
    predictions = predictions.append(pd.DataFrame(
        {mass_metric: y_test, 'm_pred': y_pred[:, 0], 'test': test_name}), ignore_index=True)
    # TODO: print the best epoch and the best results on 3 validation sets
save_predictions(predictions, args, timestamp)
