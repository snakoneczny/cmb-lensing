from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, \
    BatchNormalization, Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error


def get_model(input_shape, lr=0.001):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # TODOL flatten
    model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.1))

    model.add(Dense(200))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.1))

    model.add(Dense(100))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(20))  # linear try here
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))

    optimizer = Adam(lr)  # TODO: lr=0.0001, was 0.0005
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir, validation_data):
        self.custom_validation_data = validation_data
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs_to_send = logs.copy()
        for test_name, (X_test, y_test) in self.custom_validation_data.items():
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            logs_to_send['val_loss_{}'.format(test_name)] = mse
        super().on_epoch_end(epoch, logs_to_send)
