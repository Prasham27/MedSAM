import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = {"liver": "msd_liver_ct.yaml", "prostate": "msd_prostate_mri.yaml"}


def load_config(task):
    with open(ROOT / "configs" / CONFIG_FILES[task], encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def get_logger(script, task):
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{script}_{task}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(message)s")
        fh = logging.FileHandler(log_dir / f"{script}_{task}.log", encoding="utf-8")
        sh = logging.StreamHandler()
        for h in (fh, sh):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger
