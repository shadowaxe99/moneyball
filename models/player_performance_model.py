
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])
    return model

def train_model(model, train_data, train_labels, epochs=100):
    history = model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2)
    return model, history
