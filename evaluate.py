import numpy as np, json, os
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.load("data/processed/data.npz")
m = keras.models.load_model("models/model.h5")
loss, acc = m.evaluate(d["x_test"], d["y_test"])
pred = m.predict(d["x_test"]).argmax(1)
os.makedirs("plots", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot()
plt.savefig("plots/confusion_matrix.png")
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"))