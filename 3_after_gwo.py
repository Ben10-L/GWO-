import json, time, tracemalloc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

d = np.load("data/data.npz")
X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
models = {"KNN": KNeighborsClassifier(n_neighbors=5), "SVM": SVC()}

def evaluate(model, Xtr, ytr, Xte, yte):
    ft, pt = [], []
    for _ in range(10):
        t = time.perf_counter(); model.fit(Xtr, ytr); ft.append(time.perf_counter() - t)
        t = time.perf_counter(); pred = model.predict(Xte); pt.append(time.perf_counter() - t)
    tracemalloc.start()
    model.fit(Xtr, ytr); model.predict(Xte)
    mem = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return {"features": Xtr.shape[1],
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, average="macro", zero_division=0),
            "recall": recall_score(yte, pred, average="macro", zero_division=0),
            "f1": f1_score(yte, pred, average="macro", zero_division=0),
            "train_time": np.mean(ft), "predict_time": np.mean(pt), "memory_KB": mem}, pred

rows, preds, run_acc = {}, {}, {}
for name, model in models.items():
    g = json.load(open(f"results/gwo_{name}.json"))
    s = g["best"]
    rows[f"{name} before"], preds[f"{name} before"] = evaluate(model, X_train, y_train, X_test, y_test)
    rows[f"{name} after"], preds[f"{name} after"] = evaluate(model, X_train[:, s], y_train, X_test[:, s], y_test)
    run_acc[name] = [accuracy_score(y_test, model.fit(X_train[:, r], y_train).predict(X_test[:, r])) for r in g["runs"]]
    a = np.array(run_acc[name])
    print(f"{name}: accuracy of 20 runs = {a.mean():.4f} +/- {a.std():.4f} | baseline = {rows[name + ' before']['accuracy']:.4f}")
    try:
        print("   Wilcoxon p-value:", round(wilcoxon(a - rows[name + " before"]["accuracy"])[1], 4))
    except ValueError:
        print("   Wilcoxon: all differences are zero")

table = pd.DataFrame(rows).T
table.to_csv("results/comparison.csv")
table.round(4)

table[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 4), ylim=(0, 1), rot=0)
plt.title("Metrics before vs after GWO"); plt.tight_layout()
plt.savefig("results/cmp_metrics.png", dpi=300); plt.show()

table[["train_time", "predict_time", "memory_KB"]].plot.bar(subplots=True, layout=(1, 3), figsize=(13, 4), legend=False, rot=45)
plt.tight_layout(); plt.savefig("results/cmp_time_memory.png", dpi=300); plt.show()

table["features"].plot.bar(figsize=(6, 3.5), rot=0, color="seagreen", title="Number of features")
plt.tight_layout(); plt.savefig("results/cmp_features.png", dpi=300); plt.show()

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
for a, (k, p) in zip(ax.flat, preds.items()):
    sns.heatmap(confusion_matrix(y_test, p), cmap="Blues", cbar=False, ax=a); a.set_title(k)
plt.tight_layout(); plt.savefig("results/cmp_confusion.png", dpi=300); plt.show()

plt.boxplot(list(run_acc.values())); plt.xticks([1, 2], list(run_acc.keys()))
plt.title("Test accuracy of the 20 GWO runs"); plt.ylabel("Accuracy")
plt.savefig("results/cmp_boxplot.png", dpi=300); plt.show()
