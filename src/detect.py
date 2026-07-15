"""
Simulates a live IDS/IPS: streams flows (from the held-out test split, so
ground truth is known but not used for decisions), runs each through the
trained RF classifier + Isolation Forest, and feeds the result to the IPS
engine for a real-time detect -> decide -> act loop, printed like a SOC console.
"""
import time
import yaml
import joblib
import numpy as np
from pathlib import Path

from src.ips import IPSEngine

ACTION_COLOR = {"ALLOW": "\033[92m", "ALERT": "\033[93m", "RATE_LIMIT": "\033[95m", "BLOCK": "\033[91m"}
RESET = "\033[0m"


def load_artifacts(cfg):
    model_dir = Path(cfg["model"]["artifacts_dir"])
    return {
        "rf": joblib.load(model_dir / "rf_classifier.joblib"),
        "iso": joblib.load(model_dir / "isolation_forest.joblib"),
        "le": joblib.load(model_dir / "label_encoder.joblib"),
        "test_split": joblib.load(model_dir / "test_split.joblib"),
    }


def run(config_path: str = "config/config.yaml", n: int = 200, delay: float = 0.0, seed: int = 7):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    art = load_artifacts(cfg)
    rf, iso, le = art["rf"], art["iso"], art["le"]
    X_test, y_test, meta_test = art["test_split"]["X_test"], art["test_split"]["y_test"], art["test_split"]["meta_test"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)

    ips = IPSEngine(cfg["policy"], cfg["logging"])

    counts = {"ALLOW": 0, "ALERT": 0, "RATE_LIMIT": 0, "BLOCK": 0}
    correct_flags = 0  # attack flows that got a non-ALLOW response

    print(f"{'TIME':<10} {'SRC IP':<16} {'TRUE':<14} {'PRED':<14} {'CONF':<6} {'ANOM':<7} ACTION")
    print("-" * 90)

    for i in idx:
        x = X_test[i:i + 1]
        true_label = le.inverse_transform([y_test[i]])[0]
        src_ip = meta_test.iloc[i]["Src IP"]

        proba = rf.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = le.classes_[pred_idx]
        confidence = float(proba[pred_idx])
        anomaly_score = float(iso.decision_function(x)[0])

        action, reason = ips.decide(src_ip, pred_label, confidence, anomaly_score)
        ips.act(src_ip, action, reason, pred_label)
        counts[action] += 1
        if true_label != "BENIGN" and action != "ALLOW":
            correct_flags += 1

        color = ACTION_COLOR.get(action, "")
        ts = time.strftime("%H:%M:%S")
        print(f"{ts:<10} {src_ip:<16} {true_label:<14} {pred_label:<14} "
              f"{confidence:<6.2f} {anomaly_score:<7.3f} {color}{action}{RESET}")
        if delay:
            time.sleep(delay)

    n_attacks = int((y_test[idx] != le.transform(['BENIGN'])[0]).sum())
    print("-" * 90)
    print(f"Processed {len(idx)} flows | actions: {counts}")
    if n_attacks:
        print(f"Attack flows caught (non-ALLOW response): {correct_flags}/{n_attacks} "
              f"({100*correct_flags/n_attacks:.1f}%)")
    print(f"Active blocklist size: {len(ips.blocklist)} -> {cfg['logging']['blocklist_file']}")
    print(f"Full action log: {cfg['logging']['action_log_file']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    run(n=args.n, delay=args.delay)
