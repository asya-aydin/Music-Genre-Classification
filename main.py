import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

PATH = "features_3_sec.csv"

df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df['label'].value_counts())


df = df.drop(columns=['filename', 'length'], errors='ignore')

X = df.drop(columns=['label'])
y = df['label']


le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)


def get_metrics(y_true, y_pred, name):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\nDetailed Report:\n")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

get_metrics(y_test, y_pred_dt,  "Decision Tree")
get_metrics(y_test, y_pred_knn, "k-NN (k=5)")


metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

dt_scores = [
    accuracy_score(y_test, y_pred_dt),
    precision_score(y_test, y_pred_dt, average='weighted'),
    recall_score(y_test, y_pred_dt, average='weighted'),
    f1_score(y_test, y_pred_dt, average='weighted')
]
knn_scores = [
    accuracy_score(y_test, y_pred_knn),
    precision_score(y_test, y_pred_knn, average='weighted'),
    recall_score(y_test, y_pred_knn, average='weighted'),
    f1_score(y_test, y_pred_knn, average='weighted')
]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, dt_scores,  width, label='Decision Tree', color='steelblue')
bars2 = ax.bar(x + width/2, knn_scores, width, label='k-NN (k=5)',    color='coral')

ax.set_ylabel('Score')
ax.set_title('Decision Tree vs k-NN — Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3)
ax.bar_label(bars2, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, y_pred, title in zip(axes,
                              [y_pred_dt, y_pred_knn],
                              ['Decision Tree', 'k-NN (k=5)']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

print("\nSaved images: comparison_chart.png, confusion_matrices.png")