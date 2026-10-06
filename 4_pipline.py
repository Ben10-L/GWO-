import os
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION (SET YOUR FILE PATH HERE)
# ==========================================
DATA_FILE = "data.csv"  
DATASET_NAME = os.path.splitext(os.path.basename(DATA_FILE))[0]

RUNS = 20
WOLVES = 20
ITER = 50  
SEED = 42

RESULTS_DIR = os.path.join("results", DATASET_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. DATA LOADING & SPLITTING
# ==========================================
df = pd.read_csv(DATA_FILE)

# Assumes last column is target, all previous columns are features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)
DIM = X_train.shape[1]
print(f"[{DATASET_NAME}] Train shape: {X_train.shape} | Test shape: {X_test.shape} | Total Features: {DIM}")

# ==========================================
# 3. HELPER FUNCTIONS & MODELS
# ==========================================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", C=1.0)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate(model, Xtr, ytr, Xte, yte):
    ft, pt = [], []
    for _ in range(10):
        t = time.perf_counter(); model.fit(Xtr, ytr); ft.append(time.perf_counter() - t)
        t = time.perf_counter(); pred = model.predict(Xte); pt.append(time.perf_counter() - t)
    
    tracemalloc.start()
    model.fit(Xtr, ytr); model.predict(Xte)
    mem = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    
    return {
        "features": Xtr.shape[1],
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, average="macro", zero_division=0),
        "recall": recall_score(yte, pred, average="macro", zero_division=0),
        "f1": f1_score(yte, pred, average="macro", zero_division=0),
        "train_time": np.mean(ft),
        "predict_time": np.mean(pt),
        "memory_KB": mem
    }, pred

def fitness(model, pos):
    mask = pos > 0.5
    if mask.sum() == 0:
        return 1.0
    acc = cross_val_score(model, X_train[:, mask], y_train, cv=cv).mean()
    return 0.999 * (1 - acc) + 0.001 * (mask.sum() / DIM)

def gwo(model, run_seed):
    rng = np.random.default_rng(run_seed)
    X_pop = rng.random((WOLVES, DIM))
    fit = np.array([fitness(model, x) for x in X_pop])
    
    idx = np.argsort(fit)[:3]
    leaders, lfit = X_pop[idx].copy(), fit[idx].copy()
    curve = []
    
    for t in range(ITER):
        a = 2 - 2 * t / ITER
        for i in range(WOLVES):
            new = 0
            for L in leaders:
                A = 2 * a * rng.random(DIM) - a
                C = 2 * rng.random(DIM)
                new = new + L - A * np.abs(C * L - X_pop[i])
            X_pop[i] = np.clip(new / 3, 0, 1)
            
            f = fitness(model, X_pop[i])
            w = lfit.argmax()
            if f < lfit[w]:
                leaders[w], lfit[w] = X_pop[i].copy(), f
            
            order = np.argsort(lfit)
            leaders, lfit = leaders[order], lfit[order]
            
        curve.append(float(lfit[0]))
        
    best_mask = leaders[0] > 0.5
    return [int(i) for i in np.where(best_mask)[0]], float(lfit[0]), curve

# ==========================================
# 4. GWO OPTIMIZATION RUNS
# ==========================================
if __name__ == "__main__":
    curves = {}
    for name, model in models.items():
        print(f"Running GWO for {name}...")
        out = Parallel(n_jobs=-1)(delayed(gwo)(model, SEED + r) for r in range(RUNS))
        
        feats = [o[0] for o in out]
        fits = [o[1] for o in out]
        curves[name] = [o[2] for o in out]
        best_idx = int(np.argmin(fits))
        
        json.dump(
            {"best": feats[best_idx], "runs": feats, "curves": curves[name]},
            open(os.path.join(RESULTS_DIR, f"gwo_{name}.json"), "w")
        )
        print(f"  {name} | Best: {len(feats[best_idx])} feats | Mean kept: {np.mean([len(f) for f in feats]):.1f}")

    # Convergence Plot
    plt.figure(figsize=(8, 5))
    for name in models:
        all_runs = np.array(curves[name])
        plt.plot(all_runs.T, alpha=0.15, color="gray")
        plt.plot(np.mean(all_runs, axis=0), linewidth=2.5, label=f"{name} (Mean)")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness")
    plt.title(f"GWO Convergence - {DATASET_NAME}")
    plt.savefig(os.path.join(RESULTS_DIR, "gwo_convergence.png"), dpi=300)
    plt.close()

    # ==========================================
    # 5. EVALUATION & STATISTICAL ANALYSIS
    # ==========================================
    rows, preds, run_acc = {}, {}, {}
    for name, model in models.items():
        g = json.load(open(os.path.join(RESULTS_DIR, f"gwo_{name}.json")))
        selected_feats = g["best"]
        
        rows[f"{name} before"], preds[f"{name} before"] = evaluate(model, X_train, y_train, X_test, y_test)
        rows[f"{name} after"], preds[f"{name} after"] = evaluate(model, X_train[:, selected_feats], y_train, X_test[:, selected_feats], y_test)
        
        run_acc[name] = [
            accuracy_score(y_test, model.fit(X_train[:, r], y_train).predict(X_test[:, r])) 
            for r in g["runs"]
        ]
        
        acc_array = np.array(run_acc[name])
        base_acc = rows[f"{name} before"]["accuracy"]
        print(f"\n{name} Results:")
        print(f"  20 Runs Test Acc: {acc_array.mean():.4f} +/- {acc_array.std():.4f} | Baseline: {base_acc:.4f}")
        try:
            p_val = wilcoxon(acc_array - base_acc)[1]
            print(f"  Wilcoxon p-value: {p_val:.4f}")
        except ValueError:
            print("  Wilcoxon: All run differences are identical")

    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
    print("\nFinal Results Table:")
    print(table.round(4))

    # ==========================================
    # 6. VISUALIZATIONS
    # ==========================================
    table[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 4), ylim=(0, 1), rot=0)
    plt.title(f"Metrics Before vs After GWO ({DATASET_NAME})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmp_metrics.png"), dpi=300)
    plt.close()

    table[["train_time", "predict_time", "memory_KB"]].plot.bar(
        subplots=True, layout=(1, 3), figsize=(13, 4), legend=False, rot=45
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmp_time_memory.png"), dpi=300)
    plt.close()

    table["features"].plot.bar(figsize=(6, 3.5), rot=0, color="seagreen", title=f"Selected Features ({DATASET_NAME})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmp_features.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for a, (k, p) in zip(ax.flat, preds.items()):
        sns.heatmap(confusion_matrix(y_test, p), cmap="Blues", cbar=False, ax=a)
        a.set_title(k)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cmp_confusion.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.boxplot(list(run_acc.values()))
    plt.xticks([1, 2], list(run_acc.keys()))
    plt.title(f"Test Accuracy across 20 Runs ({DATASET_NAME})")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, "cmp_boxplot.png"), dpi=300)
    plt.close()

    print(f"\nFinished. Outputs saved to '{RESULTS_DIR}/'.")