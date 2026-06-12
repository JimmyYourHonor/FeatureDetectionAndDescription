import argparse

from config import load_config
from build import build_datasets
from runner import run_training


def main():
    parser = argparse.ArgumentParser(description="Train feature detection model")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    args, extra = parser.parse_known_args()

    cfg = load_config(args.config, *extra)
    datasets = build_datasets(cfg)
    run_training(cfg, datasets)


if __name__ == "__main__":
    main()
