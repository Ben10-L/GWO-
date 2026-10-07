import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


RUNS = 20
WOLVES = 20
ITER = 50
N_JOBS = 2

CV_FOLDS = 5

ALPHA = 0.99

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


with np.load(BASE_DIR / "data" / "data.npz") as data:
    X_train = data["X_train"]
    y_train = data["y_train"]

DIM = X_train.shape[1]


cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=42,
)


models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(),
    ),
}


def fitness(model, position):
    mask = position > 0.5

    if mask.sum() == 0:
        return 1.0

    # Mean accuracy over CV folds
    accuracy = cross_val_score(
        model,
        X_train[:, mask],
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    ).mean()

    # Classification quality + feature reduction
    return (
        ALPHA * (1 - accuracy)
        + (1 - ALPHA) * mask.sum() / DIM
    )


def gwo(model, seed):
    rng = np.random.default_rng(seed)
    positions = rng.random((WOLVES, DIM))

    cache = {}

    def score(position):
        mask = position > 0.5
        key = np.packbits(mask).tobytes()

        if key not in cache:
            cache[key] = fitness(model, position)

        return cache[key]

    scores = np.array([score(p) for p in positions])

    best = np.argsort(scores)[:3]
    leaders = positions[best].copy()
    leader_scores = scores[best].copy()

    curve = []

    for t in range(ITER):
        a = 2 - 2 * t / ITER

        for i in range(WOLVES):
            candidates = []

            for leader in leaders:
                A = 2 * a * rng.random(DIM) - a
                C = 2 * rng.random(DIM)

                candidates.append(
                    leader - A * np.abs(C * leader - positions[i])
                )

            positions[i] = np.clip(
                np.mean(candidates, axis=0),
                0,
                1,
            )

            current_score = score(positions[i])
            worst = np.argmax(leader_scores)

            if current_score < leader_scores[worst]:
                leaders[worst] = positions[i].copy()
                leader_scores[worst] = current_score

                order = np.argsort(leader_scores)
                leaders = leaders[order]
                leader_scores = leader_scores[order]

        curve.append(float(leader_scores[0]))

    features = np.flatnonzero(leaders[0] > 0.5).tolist()

    return features, float(leader_scores[0]), curve


if __name__ == "__main__":
    curves = {}

    print("Train :", X_train.shape)
    print(
        f"RUNS={RUNS}, WOLVES={WOLVES}, "
        f"ITER={ITER}, CV={CV_FOLDS}"
    )

    for name, model in models.items():
        print(f"\nGWO - {name}")

        outputs = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(gwo)(model, 42 + run)
            for run in range(RUNS)
        )

        subsets = [x[0] for x in outputs]
        scores = [x[1] for x in outputs]
        curves[name] = [x[2] for x in outputs]

        best = int(np.argmin(scores))

        result = {
            "best": subsets[best],
            "runs": subsets,
            "fitness": scores,
            "dimension": DIM,
            "cv_folds": CV_FOLDS,
        }

        with open(
            RESULTS_DIR / f"gwo_{name}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(result, file, indent=2)

        print(
            f"Best subset: "
            f"{len(subsets[best])}/{DIM}"
        )

        print(
            f"Mean features: "
            f"{np.mean([len(x) for x in subsets]):.2f}"
        )

    # Mean convergence over all runs
    for name, values in curves.items():
        plt.plot(
            range(1, ITER + 1),
            np.mean(values, axis=0),
            label=name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title("GWO convergence")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gwo_convergence.png",
        dpi=300,
    )

    plt.show()