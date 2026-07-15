"""
Generates a synthetic network-flow dataset that mirrors the CICIDS2017
feature schema (flow duration, packet counts/sizes, inter-arrival times,
TCP flag counts, etc.) with per-class statistical signatures, so that a
classifier trained on it learns realistic-looking decision boundaries.

If you have the real CICIDS2017/2018 CSVs, skip this script entirely and
point config/config.yaml -> data.source_csv at your combined real file
(it must have the same numeric feature columns + a 'Label' column).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# Attack classes and their relative prevalence (BENIGN dominates, like real traffic)
CLASS_WEIGHTS = {
    "BENIGN": 0.70,
    "DDoS": 0.10,
    "PortScan": 0.08,
    "Bot": 0.04,
    "Brute Force": 0.04,
    "Web Attack": 0.03,
    "Infiltration": 0.01,
}

FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "SYN Flag Count", "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Subflow Fwd Packets", "Subflow Bwd Packets",
    "Active Mean", "Idle Mean",
]

# Per-class (mean, std) multipliers relative to a BENIGN baseline, chosen to
# reflect well-known real-world signatures (e.g. DDoS = huge packet rate,
# PortScan = tiny packets/many short flows, Brute Force = many short flows
# with repeated SYN/ACK, Infiltration = long, low-and-slow flows).
CLASS_PROFILES = {
    "BENIGN":       dict(duration=1.0, pkt_rate=1.0, pkt_size=1.0, syn=1.0, iat=1.0),
    "DDoS":         dict(duration=0.3, pkt_rate=12.0, pkt_size=0.6, syn=4.0, iat=0.05),
    "PortScan":     dict(duration=0.05, pkt_rate=3.0, pkt_size=0.2, syn=6.0, iat=0.02),
    "Bot":          dict(duration=2.5, pkt_rate=0.8, pkt_size=1.3, syn=1.2, iat=1.8),
    "Brute Force":  dict(duration=0.4, pkt_rate=2.5, pkt_size=0.7, syn=3.0, iat=0.15),
    "Web Attack":   dict(duration=0.8, pkt_rate=1.5, pkt_size=1.8, syn=1.5, iat=0.5),
    "Infiltration": dict(duration=8.0, pkt_rate=0.3, pkt_size=1.1, syn=0.8, iat=4.0),
}

BASE_MEANS = {
    "Flow Duration": 5_000_000, "Total Fwd Packets": 12, "Total Backward Packets": 10,
    "Total Length of Fwd Packets": 1500, "Total Length of Bwd Packets": 1400,
    "Fwd Packet Length Mean": 120, "Fwd Packet Length Std": 40,
    "Bwd Packet Length Mean": 130, "Bwd Packet Length Std": 45,
    "Flow Bytes/s": 5000, "Flow Packets/s": 20,
    "Flow IAT Mean": 200000, "Flow IAT Std": 90000, "Flow IAT Max": 800000, "Flow IAT Min": 500,
    "Fwd IAT Mean": 180000, "Bwd IAT Mean": 190000,
    "SYN Flag Count": 1, "ACK Flag Count": 8, "PSH Flag Count": 4, "RST Flag Count": 0,
    "Down/Up Ratio": 1.0, "Average Packet Size": 125,
    "Subflow Fwd Packets": 12, "Subflow Bwd Packets": 10,
    "Active Mean": 300000, "Idle Mean": 100000,
}


def _make_ip_pool(n, kind="private"):
    if kind == "private":
        return [f"10.0.{RNG.integers(0,255)}.{RNG.integers(1,255)}" for _ in range(n)]
    return [f"{RNG.integers(1,223)}.{RNG.integers(0,255)}.{RNG.integers(0,255)}.{RNG.integers(1,255)}" for _ in range(n)]


def generate(n_rows: int = 60000, seed: int = 42) -> pd.DataFrame:
    global RNG
    RNG = np.random.default_rng(seed)

    classes = list(CLASS_WEIGHTS.keys())
    probs = np.array(list(CLASS_WEIGHTS.values()))
    labels = RNG.choice(classes, size=n_rows, p=probs)

    data = {f: np.zeros(n_rows) for f in FEATURES}

    for cls in classes:
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        prof = CLASS_PROFILES[cls]
        for feat in FEATURES:
            base = BASE_MEANS[feat]
            if feat in ("Flow Duration", "Fwd IAT Mean", "Bwd IAT Mean", "Flow IAT Mean",
                        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Active Mean", "Idle Mean"):
                mult = prof["duration"] if "Duration" in feat else prof["iat"]
            elif feat in ("Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets",
                          "Total Backward Packets", "Subflow Fwd Packets", "Subflow Bwd Packets"):
                mult = prof["pkt_rate"]
            elif "Packet Length" in feat or "Average Packet Size" in feat or "Length of" in feat:
                mult = prof["pkt_size"]
            elif "Flag" in feat:
                mult = prof["syn"]
            else:
                mult = 1.0
            mean = base * mult
            std = max(mean * 0.25, 1e-3)
            vals = RNG.normal(loc=mean, scale=std, size=len(idx))
            data[feat][idx] = np.clip(vals, 0, None)

    df = pd.DataFrame(data)
    df["Label"] = labels

    # Source/destination IPs — used by the IPS engine for blocking, NOT as an ML feature
    df["Src IP"] = _make_ip_pool(n_rows, "private")
    attack_mask = df["Label"] != "BENIGN"
    df.loc[attack_mask, "Src IP"] = _make_ip_pool(attack_mask.sum(), "public")
    df["Dst IP"] = _make_ip_pool(n_rows, "private")
    df["Dst Port"] = RNG.choice([80, 443, 22, 21, 3389, 8080, 3306], size=n_rows)

    # shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--out", type=str, default="data/synthetic_cicids2017_sample.csv")
    args = parser.parse_args()

    df = generate(args.rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} rows -> {args.out}")
    print(df["Label"].value_counts())
