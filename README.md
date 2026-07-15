# ML-Based IDS/IPS (Intrusion Detection & Prevention System)

A machine-learning-driven network security system that **detects** intrusions
in network flow data and **actively prevents** them (blocking / rate-limiting
/ alerting), modeled on the CICIDS2017 flow-feature schema.

## Architecture

```
                       ┌────────────────────────┐
   Network Flows  ───▶ │   Preprocessing         │
   (78 CICIDS2017      │   (clean, scale,        │
    features/flow)     │    encode)              │
                       └───────────┬─────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │  Detection Engine        │
                       │  ├─ Random Forest        │  → known attack class
                       │  │  (multi-class)         │    (DDoS/PortScan/...)
                       │  └─ Isolation Forest      │  → unknown / zero-day
                       │     (anomaly score)        │    anomaly flag
                       └───────────┬─────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │  IPS Decision Engine      │
                       │  (rules + confidence      │
                       │   thresholds)             │
                       └───────────┬─────────────┘
                        ┌──────────┼───────────┐
                        ▼          ▼           ▼
                     ALLOW      ALERT        BLOCK / RATE-LIMIT
                                              (blocklist + firewall
                                               rule simulation + log)
```

## Why synthetic data?

The real CICIDS2017/2018 datasets are multi-gigabyte downloads hosted by
the University of New Brunswick and can't be fetched inside this sandbox.
`src/generate_sample_data.py` generates a **schema-identical** synthetic
dataset (same 78 flow features, same attack categories, realistic
statistical distributions per class) so the entire pipeline trains and runs
end-to-end out of the box.

**To use the real dataset instead:** download the CSVs from
https://www.unb.ca/cic/datasets/ids-2017.html, drop them in `data/raw/`,
and point `config/config.yaml` → `data.source_csv` at the combined file.
Nothing else in the pipeline changes — it consumes any CSV with a `Label`
column and numeric flow features.

## Project structure

```
ids_ips_ml/
├── config/config.yaml         # thresholds, policy, feature list
├── data/
│   ├── generate_sample_data.py
│   └── synthetic_cicids2017_sample.csv
├── models/                    # trained artifacts (.joblib)
├── reports/                   # metrics, confusion matrix, ROC curves
├── logs/                      # IPS action logs (blocklist, alerts)
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ips.py                 # detection→decision→action engine
│   └── detect.py              # live traffic simulation + IPS in action
└── main.py                    # CLI entry point
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the synthetic training data (schema-matched to CICIDS2017)
python main.py generate-data --rows 60000

# 2. Train the classifier + anomaly detector
python main.py train

# 3. Evaluate on held-out test data
python main.py evaluate

# 4. Run the live IDS/IPS simulation (streams flows, detects, blocks)
python main.py run --n 200
```

## How detection works

- **Random Forest (multi-class)** — trained on labeled flows to recognize
  known attack categories: `BENIGN`, `DDoS`, `PortScan`, `Bot`,
  `Brute Force`, `Web Attack`, `Infiltration`.
- **Isolation Forest (unsupervised)** — trained only on BENIGN traffic to
  flag flows that don't resemble normal behavior, catching **novel/zero-day**
  attacks the classifier has never seen labeled examples of.
- A flow is escalated to the IPS if *either* model raises a flag, with the
  Random Forest's class + confidence driving the specific response.

## How prevention (IPS) works

`src/ips.py` implements a policy engine (`config/config.yaml` → `policy`):

| Condition                                   | Action               |
|----------------------------------------------|----------------------|
| Benign / low anomaly score                   | ALLOW                |
| Known attack, confidence ≥ block_threshold   | BLOCK source IP       |
| Known attack, confidence between thresholds  | RATE_LIMIT source IP |
| Anomaly-only flag (Isolation Forest), no known class | ALERT (analyst review) |
| Repeated offenses from same IP               | Escalate to permanent block |

Blocking is **simulated safely** — it writes to `logs/blocklist.json` and
`logs/ips_actions.log`, and prints the firewall rule it *would* issue
(e.g. `iptables -A INPUT -s <ip> -j DROP`). Enabling
`policy.enforce_real_firewall_rules: true` in the config is intentionally
left as a stub (`_apply_real_firewall_rule`) that raises `NotImplementedError`
— wiring it to a real `iptables`/`nftables`/cloud security-group call is an
infrastructure decision for wherever you deploy this, deliberately left out
of a sandboxed demo project.

## Results

See `reports/classification_report.txt`, `reports/confusion_matrix.png`,
and `reports/roc_curves.png` after running `evaluate`.

## Notes / limitations

- This is a **defensive security / educational project**. It detects and
  blocks malicious traffic; it contains no attack or exploit code.
- Synthetic data approximates real traffic statistics but won't match live
  network conditions — retrain on real CICIDS2017/2018 or your own labeled
  NetFlow/PCAP-derived data before any production use.
- Random Forest was chosen for interpretability and strong baseline
  performance on CICIDS-style tabular flow data; swap in XGBoost/LightGBM
  in `src/train.py` for a performance bump if desired.
