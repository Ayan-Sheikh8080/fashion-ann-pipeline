import os
import numpy as np
from tensorflow import keras

os.makedirs("data/raw", exist_ok=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
np.savez("data/raw/train.npz", x=x_train, y=y_train)
np.savez("data/raw/test.npz", x=x_test, y=y_test)
print("Saved raw data:", x_train.shape, x_test.shape)