from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout


def get_model(input_shape):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='linear'))  # was relu in paper
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.0005))  # TODO: lr=0.0001

    return model
