import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

# Load dataset
df = pd.read_csv("data\data.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

# Encode target
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  # M=1, B=0

# Prepare data
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BAT Feature Selector
class BatFeatureSelector:
    def __init__(self, model, n_bats=20, n_iter=50, alpha=0.9, gamma=0.9):
        self.model = model
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.alpha = alpha
        self.gamma = gamma
        self.best_features = None

    def fitness(self, X, y, feature_mask):
        if np.count_nonzero(feature_mask) == 0:
            return 0
        X_sel = X[:, feature_mask == 1]
        clf = clone(self.model)
        clf.fit(X_sel, y)
        pred = clf.predict(X_sel)
        return accuracy_score(y, pred)

    def fit(self, X, y):
        dim = X.shape[1]
        bats = np.random.randint(0, 2, size=(self.n_bats, dim))
        velocity = np.zeros((self.n_bats, dim))
        freq = np.zeros(self.n_bats)
        fitness = np.array([self.fitness(X, y, bats[i]) for i in range(self.n_bats)])
        best_idx = np.argmax(fitness)
        best_bat = bats[best_idx].copy()
        best_fit = fitness[best_idx]

        for t in range(self.n_iter):
            for i in range(self.n_bats):
                freq[i] = np.random.rand()
                velocity[i] = velocity[i] + (bats[i] ^ best_bat) * freq[i]
                s = 1 / (1 + np.exp(-velocity[i]))
                rand = np.random.rand(dim)
                bats[i] = np.where(rand < s, 1 - bats[i], bats[i])
                f = self.fitness(X, y, bats[i])
                if f > fitness[i]:
                    fitness[i] = f
                    if f > best_fit:
                        best_bat = bats[i].copy()
                        best_fit = f

        self.best_features = best_bat

    def transform(self, X):
        return X[:, self.best_features == 1]

# Apply BAT algorithm
bat_selector = BatFeatureSelector(GaussianNB())
bat_selector.fit(X_train, y_train)
X_train_sel = bat_selector.transform(X_train)
X_test_sel = bat_selector.transform(X_test)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train_sel, y_train)
y_pred = model.predict(X_test_sel)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save results
output_dir = "breast_cancer_model_output"
os.makedirs(output_dir, exist_ok=True)

# Save confusion matrix image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.jpeg"), format='jpeg')
plt.close()

# Save metrics
with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Selected Feature Indices: {np.where(bat_selector.best_features == 1)[0].tolist()}")

# Save model and feature mask
joblib.dump(model, os.path.join(output_dir, "naive_bayes_model.pkl"))
joblib.dump(bat_selector.best_features, os.path.join(output_dir, "selected_features.pkl"))

print("Model training and export complete. Outputs saved in:", output_dir)