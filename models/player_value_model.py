
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("../data/player_stats.csv")

# Preprocess the data
features = data.drop('value', axis=1)
target = data['value']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(features_train[0])]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)

# Train the model
history = model.fit(
    features_train, target_train,
    validation_data=(features_test, target_test),
    batch_size=256,
    epochs=100,
    verbose=0
)

# Save the model
model.save('player_value_model.h5')
