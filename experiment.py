from data import read_tng_data
from models import get_model

data, labels, folds = read_tng_data()
input_shape = data[0].shape

# TODO: train, val, test split

model = get_model(input_shape)
model.fit(train_data, train_labels, batch_size=32, epochs=400)

prediction = model.predict(validation_data)
