import numpy as np, os
from tensorflow import keras
(xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
os.makedirs("data/raw", exist_ok=True)
np.savez("data/raw/fashion.npz", x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)