from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

data = pd.read_csv(
    RESULTS_DIR / "comparison.csv",
    index_col=0
)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# 1. Accuracy before vs after GWO
data["accuracy"].plot.bar(
    figsize=(8, 5),
    ylim=(0.95, 1.0),
    rot=0
)

plt.title("Accuracy before and after GWO")
plt.ylabel("Accuracy")
save_plot("fig_accuracy.png")


# 2. Precision, Recall and F1
data[
    ["precision", "recall", "f1"]
].plot.bar(
    figsize=(10, 5),
    ylim=(0.95, 1.0),
    rot=0
)

plt.title("Classification performance")
plt.ylabel("Score")
save_plot("fig_metrics.png")


# 3. Number of features
data["features"].plot.bar(
    figsize=(8, 5),
    rot=0
)

plt.title("Number of features before and after GWO")
plt.ylabel("Number of features")
save_plot("fig_features.png")


# 4. Feature reduction percentage
reduction = pd.Series({
    "KNN": (
        1
        - data.loc["KNN after", "features"]
        / data.loc["KNN before", "features"]
    ) * 100,

    "SVM": (
        1
        - data.loc["SVM after", "features"]
        / data.loc["SVM before", "features"]
    ) * 100,
})

reduction.plot.bar(
    figsize=(7, 5),
    rot=0
)

plt.title("Feature reduction with GWO")
plt.ylabel("Reduction (%)")
save_plot("fig_feature_reduction.png")


# 5. Training time
data["train_time"].plot.bar(
    figsize=(8, 5),
    rot=0
)

plt.title("Training time before and after GWO")
plt.ylabel("Time (seconds)")
save_plot("fig_train_time.png")


# 6. Prediction time
data["predict_time"].plot.bar(
    figsize=(8, 5),
    rot=0
)

plt.title("Prediction time before and after GWO")
plt.ylabel("Time (seconds)")
save_plot("fig_predict_time.png")


# 7. Memory usage
data["memory_KB"].plot.bar(
    figsize=(8, 5),
    rot=0
)

plt.title("Memory usage before and after GWO")
plt.ylabel("Memory (KB)")
save_plot("fig_memory.png")


print("Figures saved in:", RESULTS_DIR)