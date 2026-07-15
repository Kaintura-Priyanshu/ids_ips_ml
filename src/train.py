"""Train the multi-class Random Forest classifier and the Isolation Forest
anomaly detector, then persist all artifacts (model, scaler, encoder, feature list)."""
import yaml
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from src.preprocess import load_data, split_and_scale


def main(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    df = load_data(cfg["data"]["source_csv"])
    split = split_and_scale(df, cfg["data"]["test_size"], cfg["data"]["random_state"])

    print(f"Train rows: {len(split['X_train']):,} | Test rows: {len(split['X_test']):,} "
          f"| Features: {len(split['feature_cols'])} | Classes: {list(split['label_encoder'].classes_)}")

    # --- Random Forest multi-class classifier ---
    rf_cfg = cfg["model"]["rf"]
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        n_jobs=rf_cfg["n_jobs"],
        class_weight=rf_cfg["class_weight"],
        random_state=cfg["data"]["random_state"],
    )
    rf.fit(split["X_train"], split["y_train"])

    # --- Isolation Forest anomaly detector (trained on BENIGN only) ---
    if_cfg = cfg["model"]["isolation_forest"]
    benign_label = split["label_encoder"].transform(["BENIGN"])[0]
    benign_mask = split["y_train"] == benign_label
    print(f"Training Isolation Forest on {benign_mask.sum():,} benign flows...")
    iso = IsolationForest(
        n_estimators=if_cfg["n_estimators"],
        contamination=if_cfg["contamination"],
        random_state=cfg["data"]["random_state"],
    )
    iso.fit(split["X_train"][benign_mask])

    # --- Persist everything ---
    out_dir = Path(cfg["model"]["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_dir / "rf_classifier.joblib")
    joblib.dump(iso, out_dir / "isolation_forest.joblib")
    joblib.dump(split["scaler"], out_dir / "scaler.joblib")
    joblib.dump(split["label_encoder"], out_dir / "label_encoder.joblib")
    joblib.dump(split["feature_cols"], out_dir / "feature_cols.joblib")

    # Save test split for evaluate.py / detect.py to reuse
    joblib.dump(
        {"X_test": split["X_test"], "y_test": split["y_test"], "meta_test": split["meta_test"]},
        out_dir / "test_split.joblib",
    )

    print(f"Saved model artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
