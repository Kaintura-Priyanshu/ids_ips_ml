"""CLI entry point for the IDS/IPS ML project."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="ML-based IDS/IPS")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-data", help="Generate synthetic CICIDS2017-schema training data")
    p_gen.add_argument("--rows", type=int, default=60000)
    p_gen.add_argument("--out", type=str, default="data/synthetic_cicids2017_sample.csv")

    sub.add_parser("train", help="Train the RF classifier + Isolation Forest")
    sub.add_parser("evaluate", help="Evaluate on the held-out test split")

    p_run = sub.add_parser("run", help="Run the live IDS/IPS simulation")
    p_run.add_argument("--n", type=int, default=200)
    p_run.add_argument("--delay", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "generate-data":
        from data.generate_sample_data import generate
        df = generate(args.rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Generated {len(df):,} rows -> {args.out}")
        print(df["Label"].value_counts())

    elif args.command == "train":
        from src.train import main as train_main
        train_main()

    elif args.command == "evaluate":
        from src.evaluate import main as eval_main
        eval_main()

    elif args.command == "run":
        from src.detect import run
        run(n=args.n, delay=args.delay)


if __name__ == "__main__":
    main()
