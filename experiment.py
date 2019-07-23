import datetime
import argparse

import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from data import read_train_data, get_flat_mass_function
from models import get_model, CustomTensorBoard
from utils import train_test_many_split, save_predictions, get_tensorboard_dir

# Timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='experiment tag, added to logs name')
parser.add_argument('--test', dest='is_test', action='store_true', help='flag for making a quick test')
args = parser.parse_args()

# Experiment parameters
# TODO: read all images
# TODO: read more images when limiting mass or redshift
mass_metric = 'M500c'
batch_size = 64
learning_rate = 0.0001
if not args.is_test:
    epochs = 1000
    patience = 140
    n_img = 20000
else:
    epochs = 400
    patience = 10
    n_img = 200

# Read and split data
# TODO: sample equal distribution on mass
X, y = read_train_data('/users/snakoneczny/data/lensing_takahashi_17/cmb_lens_imgs', n_img=n_img, col_y=mass_metric)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

X, y = get_flat_mass_function(X, y, n_bins=100, max_bin_size=100)

# Normalize CMB images
# TODO: use some normalizer and include test splitting (check first how minimum and maximum values differ between sets)
cmb_min = X.min()
cmb_max = X.max()
X = (X - cmb_min) / (cmb_max - cmb_min)

# Train test split
y_train, y_test_low, y_test_high, y_test_random, X_train, X_test_low, X_test_high, X_test_random = \
    train_test_many_split(y, X, side_test_size=0.05, random_test_size=0.1)

# Train data generator
# TODO: image data generator
# TODO: fits as normal images maybe data generator will manage do read all
datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,)
datagen.fit(X_train)

# Train model
callbacks = [
    # TensorBoard(log_dir=get_tensorboard_dir(args, timestamp, learning_rate, batch_size)),
    CustomTensorBoard(log_dir=get_tensorboard_dir(args, timestamp, learning_rate, batch_size),
                      validation_data={'low': (X_test_low, y_test_low), 'high': (X_test_high, y_test_high)}),
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
]
# model.fit(X_train, y_train, validation_data=(X_test_random, y_test_random), batch_size=batch_size, epochs=epochs,
#           callbacks=callbacks)
model = get_model(input_shape=X[0].shape, lr=learning_rate)
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
    # TODO: print best results on validation data
save_predictions(predictions, args, timestamp)
