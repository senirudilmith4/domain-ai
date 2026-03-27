"""
=============================================================================

This module handles:
  • Real-time priority prediction for incoming tasks
  • Task ranking based on predicted priorities
  • Feedback logging and retraining pipeline
  • Integration with the Task Manager frontend
"""

import os
import json
import csv
import uuid
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = "saved_models/best_model_pipeline.pkl"
METADATA_PATH = "saved_models/model_metadata.json"
FEEDBACK_LOG  = "feedback/feedback_log.csv"
OUTPUT_DIR    = "outputs"

os.makedirs("feedback", exist_ok=True)
os.makedirs("outputs",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY MAPPING
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_LABELS = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}
PRIORITY_COLORS = {0: "🟢 Low",       1: "🟡 Medium",       2: "🔴 High"}

FEATURE_COLUMNS = [
    "Weeks_Left", "Week_Released", "Week_Deadline", "Current_Week",
    "Semester", "Year", "Weight", "Difficulty", "Estimated_Hours",
    "Current_Workload", "Procrastination_Score", "Avg_Delay_History",
    "Urgency", "Task_Type_Exam", "Task_Type_Project",
    "Task_Type_Quiz", "Task_Type_Report",
]


# ─────────────────────────────────────────────────────────────────────────────
# STORY 5  ▸  TaskPriorityPredictor  (the "API endpoint")
# ─────────────────────────────────────────────────────────────────────────────
class TaskPriorityPredictor:
    """
    Loads the best-trained model and exposes a clean prediction interface.

    Methods
    -------
    predict_single(task_dict)   → priority label + confidence
    predict_batch(task_list)    → ranked task list
    rank_tasks(task_list)       → sorted task list with explanations
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model    = joblib.load(model_path)
        self.metadata = json.load(open(METADATA_PATH))
        self.loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TaskPriorityPredictor] Model '{self.metadata['best_model']}' loaded at {self.loaded_at}")

    # ── Utility: Compute urgency score from deadline context ─────────────────
    @staticmethod
    def compute_urgency(weeks_left: int, weight: int) -> float:
        """
        Rule-based urgency score (mirrors the data generation logic).
        Higher weight + fewer weeks → higher urgency.
        """
        if weeks_left <= 0:
            return 1.0
        base = 1.0 / (1.0 + weeks_left)
        weight_factor = weight / 100.0
        return round(min(base * (1 + weight_factor), 1.0), 6)

    # ── Core: Build feature vector from a human-readable task dict ───────────
    def _build_feature_vector(self, task: dict) -> pd.DataFrame:
        """
        Converts a task description dict into the feature format the model
        expects.  Defaults are applied for fields not supplied.
        """
        weeks_left    = max(0, task.get("weeks_left", 4))
        weight        = task.get("weight", 50)
        urgency       = self.compute_urgency(weeks_left, weight)
        task_type_raw = task.get("task_type", "assignment").lower()

        row = {
            "Weeks_Left":            weeks_left,
            "Week_Released":         task.get("week_released", 1),
            "Week_Deadline":         task.get("week_deadline", weeks_left + task.get("week_released", 1)),
            "Current_Week":          task.get("current_week",  task.get("week_released", 1) + 1),
            "Semester":              task.get("semester", 1),
            "Year":                  task.get("year", 1),
            "Weight":                weight,
            "Difficulty":            task.get("difficulty", 5.0),
            "Estimated_Hours":       task.get("estimated_hours", 8),
            "Current_Workload":      task.get("current_workload", 6.0),
            "Procrastination_Score": task.get("procrastination_score", 0.5),
            "Avg_Delay_History":     task.get("avg_delay_history", 0.3),
            "Urgency":               urgency,
            "Task_Type_Exam":        1 if task_type_raw == "exam" else 0,
            "Task_Type_Project":     1 if task_type_raw == "project" else 0,
            "Task_Type_Quiz":        1 if task_type_raw == "quiz" else 0,
            "Task_Type_Report":      1 if task_type_raw == "report" else 0,
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    # ── Story 5  ▸  Single Prediction ────────────────────────────────────────
    def predict_single(self, task: dict) -> dict:
        """
        Returns priority label, confidence, and human-readable explanation.

        Parameters
        ----------
        task : dict
            Keys expected (optional defaults shown):
              task_name, weeks_left (4), weight (50), difficulty (5.0),
              estimated_hours (8), task_type ('assignment'),
              current_workload (6.0), procrastination_score (0.5),
              avg_delay_history (0.3), semester (1), year (1)

        Returns
        -------
        dict with keys: task_name, priority_class, priority_label,
                        confidence, urgency_score, explanation
        """
        X_vec      = self._build_feature_vector(task)
        pred_class = int(self.model.predict(X_vec)[0])
        proba      = self.model.predict_proba(X_vec)[0]
        confidence = round(float(proba[pred_class]) * 100, 1)
        urgency    = X_vec["Urgency"].values[0]

        explanation = self._explain(task, pred_class, X_vec)

        return {
            "task_name":     task.get("task_name", "Unnamed Task"),
            "priority_class":  pred_class,
            "priority_label":  PRIORITY_LABELS[pred_class],
            "priority_display": PRIORITY_COLORS[pred_class],
            "confidence":      f"{confidence}%",
            "urgency_score":   round(float(urgency), 4),
            "weeks_left":      int(X_vec["Weeks_Left"].values[0]),
            "explanation":     explanation,
            "predicted_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ── Story 5  ▸  Batch Prediction & Ranking ───────────────────────────────
    def predict_batch(self, task_list: list) -> list:
        """Predict priority for a list of tasks."""
        return [self.predict_single(t) for t in task_list]

    def rank_tasks(self, task_list: list) -> list:
        """
        Predict and sort tasks by urgency (highest first).
        Returns a ranked list with a 'rank' field added.
        """
        predictions = self.predict_batch(task_list)
        # Sort: High (2) → Medium (1) → Low (0), then by urgency score
        ranked = sorted(predictions,
                        key=lambda x: (-x["priority_class"], -x["urgency_score"]))
        for i, task in enumerate(ranked):
            task["rank"] = i + 1
        return ranked

    # ── Explainability helper ─────────────────────────────────────────────────
    @staticmethod
    def _explain(task: dict, pred_class: int, X_vec: pd.DataFrame) -> str:
        weeks_left = int(X_vec["Weeks_Left"].values[0])
        weight     = int(X_vec["Weight"].values[0])
        urgency    = float(X_vec["Urgency"].values[0])

        if pred_class == 2:   # High
            return (f"Classified as HIGH priority because this task has only "
                    f"{weeks_left} week(s) remaining, carries a {weight}% weight, "
                    f"and has an urgency score of {urgency:.2f}. Immediate action is recommended.")
        elif pred_class == 1:  # Medium
            return (f"Classified as MEDIUM priority. With {weeks_left} week(s) left "
                    f"and a {weight}% weighting, this task should be started soon "
                    f"to avoid it becoming high priority.")
        else:                  # Low
            return (f"Classified as LOW priority. {weeks_left} week(s) remain and "
                    f"the urgency score is low ({urgency:.2f}). Schedule time for "
                    f"this task but focus on higher-priority items first.")


# ─────────────────────────────────────────────────────────────────────────────
# STORY 6  ▸  FeedbackCollector
# ─────────────────────────────────────────────────────────────────────────────
class FeedbackCollector:
    """
    Logs student feedback on priority predictions to CSV for future retraining.

    Story 6 deliverable: feedback-driven improvement pipeline.
    """

    COLUMNS = [
        "feedback_id", "timestamp", "task_name",
        "predicted_class", "predicted_label",
        "student_corrected_class", "student_corrected_label",
        "was_correct", "weeks_left", "weight",
        "urgency_score", "student_comment",
    ]

    def __init__(self, log_path: str = FEEDBACK_LOG):
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def log(self, prediction: dict, student_corrected_class: int = None,
            comment: str = "") -> dict:
        """
        Log a prediction outcome.
        If student_corrected_class is None, the student agreed with the prediction.
        """
        if student_corrected_class is None:
            student_corrected_class = prediction["priority_class"]
            was_correct = True
        else:
            was_correct = (student_corrected_class == prediction["priority_class"])

        row = {
            "feedback_id":             str(uuid.uuid4())[:8],
            "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_name":               prediction.get("task_name", "Unknown"),
            "predicted_class":         prediction["priority_class"],
            "predicted_label":         prediction["priority_label"],
            "student_corrected_class": student_corrected_class,
            "student_corrected_label": PRIORITY_LABELS[student_corrected_class],
            "was_correct":             was_correct,
            "weeks_left":              prediction.get("weeks_left", -1),
            "weight":                  prediction.get("weight", -1),
            "urgency_score":           prediction.get("urgency_score", -1),
            "student_comment":         comment,
        }

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

        return row

    def get_accuracy_summary(self) -> dict:
        """Returns accuracy statistics from logged feedback."""
        df = pd.read_csv(self.log_path)
        if df.empty:
            return {"error": "No feedback logged yet."}
        total      = len(df)
        correct    = df["was_correct"].sum()
        accuracy   = round(correct / total * 100, 2)
        return {
            "total_feedback":   total,
            "correct_preds":    int(correct),
            "feedback_accuracy": f"{accuracy}%",
            "most_corrected":   df[~df["was_correct"]]["predicted_label"].value_counts().to_dict()
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRATION — Story 5: Real-time prediction simulation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  TASK PRIORITY PREDICTION — API DEMO")
    print("  Rashila Jayasinghe (20240718) — Integration / Frontend Engineer")
    print("=" * 70)

    predictor = TaskPriorityPredictor()
    collector = FeedbackCollector()

    # ── Simulate a student's active task list ─────────────────────────────────
    sample_tasks = [
        {
            "task_name":          "Final Year Project Proposal",
            "task_type":          "project",
            "weeks_left":         1,
            "weight":             100,
            "difficulty":         8.0,
            "estimated_hours":    20,
            "current_workload":   8.5,
            "procrastination_score": 0.7,
            "avg_delay_history":  0.6,
            "semester": 2, "year": 4,
            "week_released": 1, "week_deadline": 12, "current_week": 11,
        },
        {
            "task_name":          "Machine Learning Assignment 1",
            "task_type":          "assignment",
            "weeks_left":         4,
            "weight":             60,
            "difficulty":         7.0,
            "estimated_hours":    12,
            "current_workload":   6.0,
            "procrastination_score": 0.5,
            "avg_delay_history":  0.3,
            "semester": 2, "year": 2,
            "week_released": 2, "week_deadline": 10, "current_week": 6,
        },
        {
            "task_name":          "Research Methodology Review",
            "task_type":          "report",
            "weeks_left":         2,
            "weight":             40,
            "difficulty":         5.5,
            "estimated_hours":    6,
            "current_workload":   5.0,
            "procrastination_score": 0.3,
            "avg_delay_history":  0.2,
            "semester": 1, "year": 3,
            "week_released": 3, "week_deadline": 9, "current_week": 7,
        },
        {
            "task_name":          "Data Structures Quiz",
            "task_type":          "quiz",
            "weeks_left":         6,
            "weight":             20,
            "difficulty":         6.0,
            "estimated_hours":    4,
            "current_workload":   4.0,
            "procrastination_score": 0.2,
            "avg_delay_history":  0.1,
            "semester": 1, "year": 2,
            "week_released": 1, "week_deadline": 8, "current_week": 2,
        },
        {
            "task_name":          "Statistics Coursework",
            "task_type":          "assignment",
            "weeks_left":         0,
            "weight":             70,
            "difficulty":         8.5,
            "estimated_hours":    15,
            "current_workload":   9.0,
            "procrastination_score": 0.9,
            "avg_delay_history":  0.8,
            "semester": 2, "year": 3,
            "week_released": 4, "week_deadline": 12, "current_week": 12,
        },
    ]

    # ── Single prediction demo ────────────────────────────────────────────────
    print("\n── SINGLE PREDICTION DEMO ──────────────────────────────────────────")
    result = predictor.predict_single(sample_tasks[0])
    print(f"  Task       : {result['task_name']}")
    print(f"  Priority   : {result['priority_display']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Urgency    : {result['urgency_score']}")
    print(f"  Explanation: {result['explanation']}")

    # ── Ranked task list demo ─────────────────────────────────────────────────
    print("\n── RANKED TASK LIST ────────────────────────────────────────────────")
    ranked = predictor.rank_tasks(sample_tasks)
    print(f"  {'#':<3} {'Task':<35} {'Priority':<20} {'Conf':>6}  {'Urgency':>8}")
    print("  " + "-" * 78)
    for t in ranked:
        print(f"  {t['rank']:<3} {t['task_name']:<35} "
              f"{t['priority_display']:<20} {t['confidence']:>6}  "
              f"{t['urgency_score']:>8.4f}")

    # ── Story 6: Feedback simulation ──────────────────────────────────────────
    print("\n── FEEDBACK COLLECTION DEMO ────────────────────────────────────────")
    for t in ranked:
        # Simulate: student agrees with predictions except first one
        if t["rank"] == 2:
            # Student thinks ML Assignment 1 is actually High (2)
            fb = collector.log(t, student_corrected_class=2,
                               comment="Harder than expected, not enough time")
        else:
            fb = collector.log(t)  # Student agrees
        correction = "✗ Corrected" if not fb["was_correct"] else "✔ Confirmed"
        print(f"  {correction}  {fb['task_name']} → "
              f"{fb['predicted_label']} → {fb['student_corrected_label']}")

    # ── Feedback accuracy summary ─────────────────────────────────────────────
    summary = collector.get_accuracy_summary()
    print(f"\n── FEEDBACK ACCURACY SUMMARY ───────────────────────────────────────")
    print(f"  Total feedback logged : {summary['total_feedback']}")
    print(f"  Correct predictions   : {summary['correct_preds']}")
    print(f"  Feedback accuracy     : {summary['feedback_accuracy']}")

    # ── Save ranked output to JSON for frontend integration ───────────────────
    output_path = os.path.join(OUTPUT_DIR, "ranked_tasks_demo.json")
    with open(output_path, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"\n  → Ranked output saved: {output_path}")
    print("\n" + "=" * 70)
    print("  API & FEEDBACK PIPELINE DEMO COMPLETE")
    print("=" * 70)
