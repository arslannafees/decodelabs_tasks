import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ============================================================
# 1. LOAD & EXPLORE THE DATA
# ============================================================
# Why: Before modeling, we must understand the dataset shape,
# features, target classes, and class balance to validate
# assumptions and avoid surprises during training.

iris = load_iris()

X = iris.data    # Feature matrix (150 samples x 4 features)
y = iris.target  # Labels (0=setosa, 1=versicolor, 2=virginica)

# Build a DataFrame for easy inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df["target_name"] = df["target"].map(lambda idx: iris.target_names[idx])

print("=" * 55)
print("  1. LOAD & EXPLORE THE DATA")
print("=" * 55)
print(f"Dataset shape    : {X.shape}  (rows x features)")
print(f"Feature names    : {list(iris.feature_names)}")
print(f"Class names      : {list(iris.target_names)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Class distribution — should be balanced (50 per class)
class_dist = df["target_name"].value_counts().sort_index()
print(f"\nClass distribution (samples per class):")
print(class_dist)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================
# Why: We split BEFORE scaling to prevent data leakage.
# The scaler must only learn statistics from training data.
# shuffle=True removes order bias (Iris is sorted by class).
# random_state=42 ensures reproducible results every run.

print("\n" + "=" * 55)
print("  2. TRAIN-TEST SPLIT  (80% train | 20% test)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

print(f"Training set shape : {X_train.shape}")
print(f"Testing  set shape : {X_test.shape}")


# ============================================================
# 3. FEATURE SCALING  (StandardScaler)
# ============================================================
# Why: KNN classifies by measuring distances between points.
# Without scaling, a feature with large numeric values
# (e.g. 0–1000) will dominate over a feature scaled 0–1,
# producing biased distance calculations.
# StandardScaler transforms each feature to mean=0, std=1.
# CRITICAL: fit only on X_train, then transform both sets.

print("\n" + "=" * 55)
print("  3. FEATURE SCALING  (StandardScaler)")
print("=" * 55)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)        # transform only (no fit)

print(f"Scaler fitted on training data only (prevents data leakage).")
print(f"Mean  per feature (train, after scaling): "
      f"{X_train_scaled.mean(axis=0).round(4)}")
print(f"Std   per feature (train, after scaling): "
      f"{X_train_scaled.std(axis=0).round(4)}")


# ============================================================
# 4. TRAIN KNN MODEL  (k = 5)
# ============================================================
# Why: KNN stores all training points and classifies a new
# point by majority vote of its k nearest neighbours.
# k=5 is the standard baseline — odd number avoids ties.
# Steps: INSTANTIATE → FIT → PREDICT

print("\n" + "=" * 55)
print("  4. TRAIN KNN MODEL  (k=5)")
print("=" * 55)

print("Model training deferred until the elbow sweep finds an optimal k.")
print("Will train the final KNN with the chosen `best_k` after the sweep.")


# ============================================================
# 5. FIND OPTIMAL K  (Elbow Method)
# ============================================================
# Why: Different K values change the bias-variance tradeoff.
#   K=1  → overfits (memorises noise)
#   K=100 → underfits (too generic)
# We sweep K from 1 to 20, record the test error rate,
# then look for the "elbow" — the point where error is lowest
# before it starts climbing again.

print("\n" + "=" * 55)
print("  5. ELBOW METHOD: Finding Optimal K  (K = 1 to 20)")
print("=" * 55)

error_rates = []
k_values    = range(1, 21)

for k in k_values:
    model_k  = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(X_train_scaled, y_train)
    y_pred_k = model_k.predict(X_test_scaled)
    error    = np.mean(y_pred_k != y_test)
    error_rates.append(error)

best_k    = list(k_values)[int(np.argmin(error_rates))]
min_error = min(error_rates)

print(f"Optimal K (lowest error in sweep) : {best_k}")
print(f"Minimum error rate                : {min_error:.4f}")
print(f"Corresponding accuracy            : {1 - min_error:.4f}")

# Plot: Error Rate vs K
plt.figure(figsize=(9, 5))
plt.plot(list(k_values), error_rates, marker="o",
         linestyle="--", color="steelblue", linewidth=2,
         markersize=7, label="Error Rate")
plt.axvline(x=best_k, color="tomato", linestyle=":",
            linewidth=2, label=f"Optimal K = {best_k}")
plt.title("Elbow Method: Error Rate vs K", fontsize=14, fontweight="bold")
plt.xlabel("K (Number of Neighbours)", fontsize=12)
plt.ylabel("Error Rate", fontsize=12)
plt.xticks(list(k_values))
plt.legend(fontsize=11)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
plt.show()
print("Elbow chart saved as 'elbow_method.png'")

# --- Train final KNN with the chosen best_k from the sweep ---
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
print(f"Final model trained with n_neighbors={best_k}.")


# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================
# Why: Accuracy alone is misleading on imbalanced datasets
# ("Accuracy Mirage"). We use:
#   • Confusion Matrix  — shows per-class TP / FP / FN / TN
#   • Precision         — of all predicted positives, how many
#                         were actually positive? (spam filter)
#   • Recall            — of all actual positives, how many did
#                         we catch? (medical diagnosis)
#   • F1 Score          — harmonic mean of Precision & Recall
#   • Accuracy          — overall correct / total (context only)

print("\n" + "=" * 55)
print("  6. MODEL EVALUATION")
print("=" * 55)

y_pred = knn.predict(X_test_scaled)

# --- Confusion Matrix (labeled) ---
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index   =[f"Actual: {n}"    for n in iris.target_names],
    columns =[f"Predicted: {n}" for n in iris.target_names],
)
print("\nConfusion Matrix (labeled):")
print(cm_df)

# --- Classification Report ---
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    digits=4,
))

# --- Overall Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy Score : {accuracy:.4f}  ({accuracy*100:.2f}%)")


# ============================================================
# 7. VISUALISE  (Confusion Matrix Heatmap)
# ============================================================
# Why: A colour-coded heatmap makes it instantly obvious which
# classes the model confuses. Diagonal = correct predictions.
# Off-diagonal = misclassifications (FP / FN).

print("\n" + "=" * 55)
print("  7. CONFUSION MATRIX HEATMAP")
print("=" * 55)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot      =True,
    fmt        ="d",
    cmap       ="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    linewidths =0.5,
    linecolor  ="white",
)
plt.title(f"Confusion Matrix Heatmap  (KNN, k={best_k})",
          fontsize=13, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix heatmap saved as 'confusion_matrix.png'")

print("\n" + "=" * 55)
print("  PROJECT 2 COMPLETE — DecodeLabs Batch 2026")
print("=" * 55)