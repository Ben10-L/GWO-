from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = [
    "mfeat-fac",
    "mfeat-fou",
    "mfeat-kar",
    "mfeat-mor",
    "mfeat-pix",
    "mfeat-zer",
]


# Load and combine all feature groups
blocks = [np.loadtxt(DATA_DIR / name) for name in FILES]
X = np.hstack(blocks)

# 10 classes, 200 samples per class
y = np.repeat(np.arange(10), 200)

assert X.shape == (2000, 649)


# 70% train / 30% final test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=SEED,
)


np.savez(
    DATA_DIR / "data.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)


print("Dataset :", X.shape)
print("Train   :", X_train.shape)
print("Test    :", X_test.shape)
print("Classes :", len(np.unique(y)))