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
    parser.add_argument(
        "--model",
        default=None,
        help="Model architecture name; overrides model.name in the config",
    )
    args, extra = parser.parse_known_args()

    overrides = []
    if args.model is not None:
        overrides.append(f"model.name={args.model}")
    overrides.extend(extra)

    cfg = load_config(args.config, *overrides)
    datasets = build_datasets(cfg)
    run_training(cfg, datasets)


if __name__ == "__main__":
    main()
