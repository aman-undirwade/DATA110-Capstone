"""DATA110 Lab 02 - Classification algorithms: Logistic Regression, KNN, Decision Tree, SVM.
This is a compact practice implementation for the algorithms repeatedly demonstrated in class.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

DATA_PATH = "data.csv"
TARGET = "target"
df = pd.read_csv(DATA_PATH)
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
    "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))]),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n===", name, "===")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))
