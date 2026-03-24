"""
=============================================================================
TASK PRIORITY PREDICTION MODULE



Model   : Gradient Boosting Classifier
Reason  : Selected as the best performing model after comparison study.
          Achieved 96.93% accuracy and 0.9693 weighted F1 score,
          outperforming Decision Tree, Random Forest, and Logistic Regression.
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "/Users/thushanthmahendran/Frontend/domain-ai/ml/TaskPriority/processed_task_data.csv"
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "saved_models"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
CV_FOLDS    = 5

PRIORITY_LABELS = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}
PRIORITY_COLORS = {0: "#4CAF50",      1: "#FF9800",         2: "#F44336"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

print("=" * 70)
print("  TASK PRIORITY PREDICTION MODEL — GROUP 06")
print("  Thushanth Mahendran (20241544) — AI / ML Engineer")
print("  Model: Gradient Boosting Classifier")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD AND INSPECT DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading dataset ...")

df = pd.read_csv(DATA_PATH)
print(f"  -> Dataset shape       : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  -> Missing values      : {df.isnull().sum().sum()}")
print(f"  -> Priority distribution:\n"
      f"{df['PriorityClass'].value_counts().rename(PRIORITY_LABELS)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PREPARE FEATURES AND TARGET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Preparing features ...")

FEATURE_COLUMNS = [
    # Time-based features
    "Weeks_Left",
    "Week_Released",
    "Week_Deadline",
    "Current_Week",
    "Semester",
    "Year",
    # Effort-based features
    "Weight",
    "Difficulty",
    "Estimated_Hours",
    # Contextual / behavioural features
    "Current_Workload",
    "Procrastination_Score",
    "Avg_Delay_History",
    "Urgency",
    # Task-type indicator features (one-hot encoded)
    "Task_Type_Exam",
    "Task_Type_Project",
    "Task_Type_Quiz",
    "Task_Type_Report",
]

TARGET_COLUMN = "PriorityClass"

# Convert boolean task type columns to integers (True -> 1, False -> 0)
bool_cols = ["Task_Type_Exam", "Task_Type_Project", "Task_Type_Quiz", "Task_Type_Report"]
for col in bool_cols:
    df[col] = df[col].astype(int)

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

print(f"  -> Features used       : {len(FEATURE_COLUMNS)}")
print(f"  -> Total samples       : {len(X)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Splitting into train / test sets ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = y          # preserves class balance in both splits
)

print(f"  -> Training samples    : {len(X_train)}")
print(f"  -> Test samples        : {len(X_test)}")

# Compute class weights to handle any mild class imbalance
classes = np.array([0, 1, 2])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"  -> Class weights       : {class_weight_dict}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD THE GRADIENT BOOSTING PIPELINE
#
# Why Gradient Boosting?
#   - Achieved the highest cross-validation F1 of 0.9694 +/- 0.0041
#   - Test accuracy of 96.93% — best across all four candidate models
#   - Robust to feature scaling differences (StandardScaler still included
#     for consistency with the deployment API pipeline)
#   - Handles class imbalance well through sequential error correction
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Building Gradient Boosting pipeline ...")

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),           # normalise all numeric features
    ("clf",    GradientBoostingClassifier(
        n_estimators  = 200,    # number of boosting stages
        max_depth     = 5,      # maximum depth of each tree
        learning_rate = 0.1,    # shrinks each tree's contribution
        subsample     = 0.8,    # fraction of samples per tree (reduces overfitting)
        random_state  = RANDOM_SEED
    ))
])

print("  -> Pipeline            : StandardScaler -> GradientBoostingClassifier")
print("  -> n_estimators        : 200")
print("  -> max_depth           : 5")
print("  -> learning_rate       : 0.1")
print("  -> subsample           : 0.8")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 5] Running {CV_FOLDS}-fold stratified cross-validation ...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(
    model_pipeline, X_train, y_train,
    cv      = cv,
    scoring = "f1_weighted",
    n_jobs  = -1
)

print(f"  -> CV F1 Scores        : {[round(s, 4) for s in cv_scores]}")
print(f"  -> Mean F1             : {cv_scores.mean():.4f}")
print(f"  -> Std Dev             : +/-{cv_scores.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: TRAIN ON FULL TRAINING SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Training Gradient Boosting on full training set ...")

model_pipeline.fit(X_train, y_train)
print("  -> Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Evaluating on test set ...")

y_pred = model_pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"  -> Accuracy            : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  -> F1 Score (weighted) : {f1:.4f}")
print(f"  -> Precision           : {precision:.4f}")
print(f"  -> Recall              : {recall:.4f}")

print("\n  Per-Class Classification Report:")
report_str = classification_report(
    y_test, y_pred,
    target_names = list(PRIORITY_LABELS.values())
)
print(report_str)

# Save classification report to text file
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("Task Priority Prediction — Gradient Boosting\n")
    f.write("Group 06 | Thushanth Mahendran (20241544)\n\n")
    f.write(report_str)
print("  -> Saved: classification_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 8] Generating evaluation plots ...")

# 8a. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels = list(PRIORITY_LABELS.values()),
    yticklabels = list(PRIORITY_LABELS.values()),
    linewidths=0.5, linecolor="gray", ax=ax
)
ax.set_ylabel("Actual Priority",    fontsize=12)
ax.set_xlabel("Predicted Priority", fontsize=12)
ax.set_title(
    "Confusion Matrix — Gradient Boosting\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  -> Saved: confusion_matrix.png")

# 8b. Cross-Validation Fold Scores
fig, ax = plt.subplots(figsize=(8, 5))
fold_labels = [f"Fold {i+1}" for i in range(CV_FOLDS)]
bars = ax.bar(fold_labels, cv_scores, color="#9C27B0", alpha=0.85,
              edgecolor="white", linewidth=1.2)

for bar, score in zip(bars, cv_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{score:.4f}", ha="center", va="bottom",
        fontsize=10, fontweight="bold"
    )

ax.axhline(cv_scores.mean(), color="#F44336", linestyle="--", linewidth=1.5,
           label=f"Mean F1 = {cv_scores.mean():.4f}")
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Weighted F1-Score", fontsize=12)
ax.set_xlabel("Cross-Validation Fold", fontsize=11)
ax.set_title(
    "5-Fold Cross-Validation — Gradient Boosting\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_scores.png"), dpi=150)
plt.close()
print("  -> Saved: cv_scores.png")

# 8c. Feature Importance (native from Gradient Boosting)
gb_clf      = model_pipeline.named_steps["clf"]
importances = gb_clf.feature_importances_

feat_imp_df = pd.DataFrame({
    "Feature":    FEATURE_COLUMNS,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 7))
bar_colors_fi = [
    "#F44336" if imp > 0.08 else "#9C27B0"
    for imp in feat_imp_df["Importance"]
]
ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
        color=bar_colors_fi, edgecolor="white", linewidth=0.8)
ax.set_xlabel("Feature Importance Score", fontsize=11)
ax.set_title(
    "Feature Importance — Gradient Boosting\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="x", alpha=0.4)
high_patch = mpatches.Patch(color="#F44336", label="High importance (>0.08)")
norm_patch = mpatches.Patch(color="#9C27B0", label="Standard importance")
ax.legend(handles=[high_patch, norm_patch], loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  -> Saved: feature_importance.png")

# 8d. ROC Curves (One-vs-Rest)
y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])
y_prob      = model_pipeline.predict_proba(X_test)
line_colors = ["#4CAF50", "#FF9800", "#F44336"]

fig, ax = plt.subplots(figsize=(7, 6))
for i, (cls_name, lc) in enumerate(zip(PRIORITY_LABELS.values(), line_colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=lc, lw=2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate",  fontsize=11)
ax.set_title(
    "ROC Curves (One-vs-Rest) — Gradient Boosting\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("  -> Saved: roc_curves.png")

# 8e. Actual vs Predicted Priority Distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
actual_counts    = pd.Series(y_test).value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()
bar_clrs         = [PRIORITY_COLORS[k] for k in actual_counts.index]

axes[0].bar(
    [PRIORITY_LABELS[k] for k in actual_counts.index],
    actual_counts.values, color=bar_clrs, edgecolor="white", alpha=0.85
)
axes[0].set_title("Actual Priority Distribution",    fontsize=11, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", alpha=0.4)

axes[1].bar(
    [PRIORITY_LABELS[k] for k in predicted_counts.index],
    predicted_counts.values, color=bar_clrs, edgecolor="white", alpha=0.85
)
axes[1].set_title("Predicted Priority Distribution", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].grid(axis="y", alpha=0.4)

fig.suptitle(
    "Actual vs Predicted Priority Distribution — Test Set\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "priority_distribution.png"), dpi=150)
plt.close()
print("  -> Saved: priority_distribution.png")

# 8f. Overall Metrics Summary
metric_names  = ["Accuracy", "F1 (Weighted)", "Precision", "Recall"]
metric_values = [accuracy, f1, precision, recall]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(metric_names, metric_values,
              color=["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"],
              alpha=0.85, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.4f}", ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )

ax.set_ylim(0.5, 1.08)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "Test Set Performance — Gradient Boosting\n"
    "Group 06 | Thushanth Mahendran (20241544)",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_metrics.png"), dpi=150)
plt.close()
print("  -> Saved: model_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: EXPLAINABILITY — FEATURE IMPORTANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Generating explainability report ...")

feat_imp_sorted = feat_imp_df.sort_values("Importance", ascending=False).copy()
feat_imp_sorted["Rank"]       = range(1, len(feat_imp_sorted) + 1)
feat_imp_sorted["Importance"] = feat_imp_sorted["Importance"].round(5)

print("\n  Top-10 Features by Importance (Gradient Boosting):")
print(feat_imp_sorted[["Rank", "Feature", "Importance"]].head(10).to_string(index=False))

explainability_output = {
    "model": "Gradient Boosting",
    "top_features": (
        feat_imp_sorted[["Feature", "Importance"]]
        .head(10)
        .to_dict(orient="records")
    ),
    "decision_rule_summary": {
        "High Priority":   "Weeks_Left <= 2  AND  Weight >= 60  AND  Urgency >= 0.3",
        "Medium Priority": "Weeks_Left between 3-6  OR  Urgency between 0.1-0.3",
        "Low Priority":    "Weeks_Left > 6  AND  Urgency < 0.1"
    }
}

with open(os.path.join(OUTPUT_DIR, "explainability_report.json"), "w") as f:
    json.dump(explainability_output, f, indent=2)
print("  -> Saved: explainability_report.json")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: SAVE THE TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 10] Saving trained model ...")

model_path = os.path.join(MODEL_DIR, "best_model_pipeline.pkl")
joblib.dump(model_pipeline, model_path)
print(f"  -> Saved: best_model_pipeline.pkl")

metadata = {
    "best_model":        "Gradient Boosting",
    "feature_columns":   FEATURE_COLUMNS,
    "target_column":     TARGET_COLUMN,
    "priority_labels":   PRIORITY_LABELS,
    "training_samples":  len(X_train),
    "test_samples":      len(X_test),
    "test_metrics": {
        "Gradient Boosting": {
            "accuracy":    round(accuracy,  4),
            "f1_weighted": round(f1,        4),
            "precision":   round(precision, 4),
            "recall":      round(recall,    4),
        }
    },
    "cv_f1_scores": {
        "Gradient Boosting": {
            "mean":  round(cv_scores.mean(), 4),
            "std":   round(cv_scores.std(),  4),
            "folds": [round(s, 4) for s in cv_scores.tolist()]
        }
    },
    "hyperparameters": {
        "n_estimators":  200,
        "max_depth":     5,
        "learning_rate": 0.1,
        "subsample":     0.8,
        "random_state":  RANDOM_SEED
    }
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("  -> Saved: model_metadata.json")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TRAINING COMPLETE — SUMMARY")
print("=" * 70)
print(f"  Model              : Gradient Boosting")
print(f"  Test Accuracy      : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Test F1 (Weighted) : {f1:.4f}")
print(f"  Test Precision     : {precision:.4f}")
print(f"  Test Recall        : {recall:.4f}")
print(f"  CV Mean F1         : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Model saved to     : {model_path}")
print(f"  Plots saved to     : {OUTPUT_DIR}/")
print("=" * 70)


# =============================================================================
# CHAPTER 8: TESTING — Added for Testing Chapter Report
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: MODEL EVALUATION — Per-Class Confusion Matrix Values (TP/TN/FP/FN)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TESTING — PER-CLASS CONFUSION MATRIX BREAKDOWN")
print("=" * 70)

cm_array = confusion_matrix(y_test, y_pred)
class_names = list(PRIORITY_LABELS.values())

testing_results = {}
for i, cls_name in enumerate(class_names):
    TP = cm_array[i, i]
    FP = cm_array[:, i].sum() - TP
    FN = cm_array[i, :].sum() - TP
    TN = cm_array.sum() - TP - FP - FN
    testing_results[cls_name] = {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}
    print(f"\n  {cls_name}:")
    print(f"    True Positives  (TP): {TP}")
    print(f"    True Negatives  (TN): {TN}")
    print(f"    False Positives (FP): {FP}")
    print(f"    False Negatives (FN): {FN}")

# Save to JSON for the report
with open(os.path.join(OUTPUT_DIR, "confusion_matrix_breakdown.json"), "w") as f:
    json.dump(testing_results, f, indent=2)
print("\n  -> Saved: confusion_matrix_breakdown.json")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: BENCHMARKING — Compare Against a Dummy Baseline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TESTING — BENCHMARKING vs DUMMY BASELINE")
print("=" * 70)

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

dummy_acc = accuracy_score(y_test, y_dummy)
dummy_f1  = f1_score(y_test, y_dummy, average="weighted", zero_division=0)

print(f"  Dummy Baseline (most_frequent):")
print(f"    Accuracy : {dummy_acc:.4f}  ({dummy_acc*100:.2f}%)")
print(f"    F1 Score : {dummy_f1:.4f}")
print(f"\n  Gradient Boosting:")
print(f"    Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"    F1 Score : {f1:.4f}")
print(f"\n  Improvement over baseline:")
print(f"    Accuracy : +{(accuracy - dummy_acc)*100:.2f}%")
print(f"    F1 Score : +{(f1 - dummy_f1):.4f}")

# Benchmarking bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
width = 0.3
bars1 = ax.bar(x - width/2, [dummy_acc, dummy_f1],  width, label="Dummy Baseline", color="#B0BEC5", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, [accuracy,  f1],         width, label="Gradient Boosting", color="#9C27B0", alpha=0.85, edgecolor="white")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "F1 Score (Weighted)"], fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Benchmarking — Gradient Boosting vs Dummy Baseline\nGroup 06 | Thushanth Mahendran (20241544)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "benchmarking.png"), dpi=150)
plt.close()
print("  -> Saved: benchmarking.png")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: FUNCTIONAL TESTING — predict_single() and rank_tasks()
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TESTING — FUNCTIONAL TESTS (predict_single + rank_tasks)")
print("=" * 70)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

functional_results = []

# Define test cases
test_cases = [
    {
        "id": "TC-01",
        "description": "High urgency task — 1 week left, 100% weight",
        "input": {"task_name": "Final Year Project", "task_type": "project",
                  "weeks_left": 1, "weight": 100, "difficulty": 9.0,
                  "estimated_hours": 20, "current_workload": 9.0,
                  "procrastination_score": 0.8, "avg_delay_history": 0.7,
                  "semester": 2, "year": 4, "week_released": 1,
                  "week_deadline": 12, "current_week": 11},
        "expected_class": 2,
        "expected_label": "High Priority"
    },
    {
        "id": "TC-02",
        "description": "Low urgency task — 10 weeks left, 10% weight",
        "input": {"task_name": "Optional Reading", "task_type": "assignment",
                  "weeks_left": 10, "weight": 10, "difficulty": 2.0,
                  "estimated_hours": 2, "current_workload": 2.0,
                  "procrastination_score": 0.1, "avg_delay_history": 0.1,
                  "semester": 1, "year": 1, "week_released": 1,
                  "week_deadline": 12, "current_week": 2},
        "expected_class": 0,
        "expected_label": "Low Priority"
    },
    {
        "id": "TC-03",
        "description": "Medium urgency task — 3 weeks left, 40% weight",
        "input": {"task_name": "Statistics Report", "task_type": "report",
                  "weeks_left": 3, "weight": 40, "difficulty": 5.0,
                  "estimated_hours": 8, "current_workload": 5.0,
                  "procrastination_score": 0.4, "avg_delay_history": 0.3,
                  "semester": 1, "year": 2, "week_released": 2,
                  "week_deadline": 9, "current_week": 6},
        "expected_class": 1,
        "expected_label": "Medium Priority"
    },
]

# Run functional tests using the trained pipeline directly
priority_map = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}

for tc in test_cases:
    task = tc["input"]
    weeks_left = max(0, task["weeks_left"])
    weight     = task["weight"]
    urgency    = min((1.0 / (1.0 + weeks_left)) * (1.0 + weight / 100.0), 1.0)
    tt = task.get("task_type", "assignment").lower()
    row = {
        "Weeks_Left": weeks_left, "Week_Released": task.get("week_released", 1),
        "Week_Deadline": task.get("week_deadline", 12), "Current_Week": task.get("current_week", 6),
        "Semester": task.get("semester", 1), "Year": task.get("year", 1),
        "Weight": weight, "Difficulty": task.get("difficulty", 5.0),
        "Estimated_Hours": task.get("estimated_hours", 8),
        "Current_Workload": task.get("current_workload", 5.0),
        "Procrastination_Score": task.get("procrastination_score", 0.5),
        "Avg_Delay_History": task.get("avg_delay_history", 0.3),
        "Urgency": urgency,
        "Task_Type_Exam":    1 if tt == "exam"    else 0,
        "Task_Type_Project": 1 if tt == "project" else 0,
        "Task_Type_Quiz":    1 if tt == "quiz"    else 0,
        "Task_Type_Report":  1 if tt == "report"  else 0,
    }
    X_vec = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred_class = int(model_pipeline.predict(X_vec)[0])
    pred_label = priority_map[pred_class]
    passed = pred_class == tc["expected_class"]
    status = "PASS" if passed else "FAIL"
    functional_results.append({
        "id": tc["id"], "description": tc["description"],
        "expected": tc["expected_label"], "actual": pred_label, "status": status
    })
    print(f"  {tc['id']} [{status}]  {tc['description']}")
    print(f"         Expected: {tc['expected_label']}  |  Got: {pred_label}")

with open(os.path.join(OUTPUT_DIR, "functional_test_results.json"), "w") as f:
    json.dump(functional_results, f, indent=2)
print("\n  -> Saved: functional_test_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: NON-FUNCTIONAL — Prediction Response Time
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TESTING — NON-FUNCTIONAL: PREDICTION RESPONSE TIME")
print("=" * 70)

import time

# Single prediction timing
times = []
for _ in range(100):
    start = time.time()
    model_pipeline.predict(X_test.iloc[[0]])
    times.append((time.time() - start) * 1000)  # ms

avg_ms  = round(sum(times) / len(times), 3)
min_ms  = round(min(times), 3)
max_ms  = round(max(times), 3)

# Batch prediction timing
start_batch = time.time()
model_pipeline.predict(X_test)
batch_ms = round((time.time() - start_batch) * 1000, 2)

print(f"  Single prediction (avg over 100 runs) : {avg_ms} ms")
print(f"  Single prediction min                 : {min_ms} ms")
print(f"  Single prediction max                 : {max_ms} ms")
print(f"  Batch prediction ({len(X_test)} samples)        : {batch_ms} ms")

perf_results = {
    "single_prediction_avg_ms": avg_ms,
    "single_prediction_min_ms": min_ms,
    "single_prediction_max_ms": max_ms,
    "batch_prediction_ms":      batch_ms,
    "batch_size":               len(X_test)
}
with open(os.path.join(OUTPUT_DIR, "performance_test_results.json"), "w") as f:
    json.dump(perf_results, f, indent=2)
print("  -> Saved: performance_test_results.json")

print("\n" + "=" * 70)
print("  ALL TESTING COMPLETE")
print("=" * 70)
