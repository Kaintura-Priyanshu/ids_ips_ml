"""Evaluate the trained classifier on the held-out test split and produce
a text report + confusion matrix + per-class ROC curves."""
import yaml
import joblib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def main(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(cfg["model"]["artifacts_dir"])
    rf = joblib.load(model_dir / "rf_classifier.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    test_split = joblib.load(model_dir / "test_split.joblib")

    X_test, y_test = test_split["X_test"], test_split["y_test"]
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    report_dir = Path(cfg["reports"]["output_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- Classification report ---
    report_str = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
    print(report_str)
    (report_dir / "classification_report.txt").write_text(report_str)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("IDS Confusion Matrix (Random Forest)")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC curves (one-vs-rest per class) ---
    classes = np.arange(len(le.classes_))
    y_test_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8, 7))
    for i, cls_name in enumerate(le.classes_):
        if y_test_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("IDS ROC Curves (per class, one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curves.png", dpi=150)
    plt.close()

    print(f"\nSaved report + plots to {report_dir}/")


if __name__ == "__main__":
    main()
