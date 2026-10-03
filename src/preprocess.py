# preprocessing step: normalize and split data
import numpy as np, os, yaml
from sklearn.model_selection import train_test_split
p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion.npz")
x = d["x_train"].astype("float32") / 255.0
xt = d["x_test"].astype("float32") / 255.0
xtr, xv, ytr, yv = train_test_split(x, d["y_train"], test_size=p["test_size"], random_state=p["seed"])
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/data.npz", x_train=xtr, y_train=ytr, x_val=xv, y_val=yv, x_test=xt, y_test=d["y_test"])