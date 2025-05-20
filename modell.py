import numpy as np
import matplotlib as plt
from tensorflow import keras
import seaborn as sn

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train_flat = X_train.reshape(len(X_train), (28*28))
X_test_flat = X_test.reshape(len(X_test), (28*28))

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation = 'relu'),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(
    optimizer = 'adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_flat, y_train, epochs=20)

