import argparse
import random

import numpy as np
import torch

from .config import from_dict, load_config
from .variants import variant_overrides


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--variant", default=None, help="named model variant (see mbtf/variants.py)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. model.dim=192 train.batch_size=8")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def build_cfg(args):
    return load_config(args.config, variant_overrides(args.variant) + list(args.set))


def cfg_from_ckpt(ckpt, extra=()):
    return from_dict(ckpt["cfg"], extra)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
