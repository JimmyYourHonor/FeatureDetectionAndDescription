from omegaconf import OmegaConf, DictConfig

from config.schema import RunCfg


def load_config(path: str, *overrides: str) -> RunCfg:
    """Load a YAML config and apply key=value overrides.

    Args:
        path: Path to a YAML file.
        *overrides: Dotted-path overrides like "optim.learning_rate=1e-3".

    Returns:
        A structured RunCfg instance with types enforced by the dataclass schema.
    """
    schema = OmegaConf.structured(RunCfg)
    file_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, file_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(merged, cli_cfg)
    return OmegaConf.to_object(merged)


def dump_config(cfg: RunCfg, path: str) -> None:
    """Serialize a RunCfg to a YAML file.

    Args:
        cfg: A RunCfg dataclass instance (or OmegaConf DictConfig).
        path: Destination file path.
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.structured(cfg)
    OmegaConf.save(cfg, path)
