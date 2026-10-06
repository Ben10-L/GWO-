import json
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

RUNS, WOLVES, ITER = 20, 20, 50      # use ITER = 100 for the final run
d = np.load("data/data.npz")
X_train, y_train = d["X_train"], d["y_train"]
DIM = X_train.shape[1]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {"KNN": KNeighborsClassifier(n_neighbors=5), "SVM": SVC()}

def fitness(model, pos):
    mask = pos > 0.5
    if mask.sum() == 0:
        return 1.0
    acc = cross_val_score(model, X_train[:, mask], y_train, cv=cv).mean()
    return 0.99 * (1 - acc) + 0.01 * mask.sum() / DIM

def gwo(model, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((WOLVES, DIM))
    fit = np.array([fitness(model, x) for x in X])
    idx = np.argsort(fit)[:3]
    leaders, lfit = X[idx].copy(), fit[idx].copy()      # alpha, beta, delta
    curve = []
    for t in range(ITER):
        a = 2 - 2 * t / ITER
        for i in range(WOLVES):
            new = 0
            for L in leaders:
                A = 2 * a * rng.random(DIM) - a
                C = 2 * rng.random(DIM)
                new = new + L - A * np.abs(C * L - X[i])
            X[i] = np.clip(new / 3, 0, 1)
            f = fitness(model, X[i])
            w = lfit.argmax()
            if f < lfit[w]:
                leaders[w], lfit[w] = X[i].copy(), f
            order = np.argsort(lfit)
            leaders, lfit = leaders[order], lfit[order]
        curve.append(float(lfit[0]))
    return [int(i) for i in np.where(leaders[0] > 0.5)[0]], float(lfit[0]), curve

if __name__ == "__main__":
    curves = {}
    for name, model in models.items():
        out = Parallel(n_jobs=-1)(delayed(gwo)(model, 42 + r) for r in range(RUNS))
        feats, fits, curves[name] = [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]
        best = int(np.argmin(fits))
        json.dump({"best": feats[best], "runs": feats}, open(f"results/gwo_{name}.json", "w"))
        print(name, "| best subset:", len(feats[best]), "features | mean kept:", np.mean([len(f) for f in feats]))

    for name in models:
        plt.plot(np.mean(curves[name], axis=0), label=name)
    plt.xlabel("Iteration"); plt.ylabel("Best fitness"); plt.title("GWO convergence (mean of runs)")
    plt.legend(); plt.savefig("results/gwo_convergence.png", dpi=300); plt.show()
