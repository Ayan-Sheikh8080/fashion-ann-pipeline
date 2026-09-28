import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

test = np.load("data/processed/test.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
with open("metrics.json", "w") as f:
    json.dump({"loss": float(loss), "accuracy": float(acc)}, f, indent=2)

preds = np.argmax(model.predict(test["x"], verbose=0), axis=1)
cm = confusion_matrix(test["y"], preds)
os.makedirs("reports", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
plt.savefig("reports/confusion_matrix.png", dpi=120, bbox_inches="tight")
print(f"Test loss={loss:.4f} accuracy={acc:.4f}")