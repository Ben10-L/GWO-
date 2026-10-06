import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Fix seed
SEED = 42
np.random.seed(SEED)

# __data check
df = pd.read_csv("data/movement_libras.data", header=None)   
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X.columns = [f"f{i}" for i in range(X.shape[1])]

#Shape X: (360, 90)
print("Shape X:", X.shape)
# 15 classes
print("Classes :", y.nunique())
# balanced classes 24 sample each , 24 x 15 = 360
y.value_counts().sort_index()

# 0
print("Missing values:", int(X.isnull().sum().sum()))
X = X.fillna(X.mean())
X.describe().T.head()

# __split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
print(X_train.shape, X_test.shape)

# __Scaling
#scaler = StandardScaler()
#X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
#X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# __Save
train = X_train.copy() 
train["target"] = y_train.values
test = X_test.copy()
test["target"] = y_test.values
train.to_csv("data/libras_train.csv", index=False)
test.to_csv("data/libras_test.csv", index=False)

print("Saved:", train.shape, test.shape)