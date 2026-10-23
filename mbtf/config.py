import copy

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.model.dim)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(d):
    if isinstance(d, dict):
        return Cfg({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_wrap(v) for v in d]
    return d


def to_dict(c):
    if isinstance(c, dict):
        return {k: to_dict(v) for k, v in c.items()}
    if isinstance(c, list):
        return [to_dict(v) for v in c]
    return c


def apply_overrides(d, overrides):
    d = copy.deepcopy(to_dict(d))
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        key, val = ov.split("=", 1)
        val = yaml.safe_load(val)
        cur = d
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return d


def load_config(path, overrides=()):
    with open(path) as f:
        d = yaml.safe_load(f)
    return _wrap(apply_overrides(d, overrides))


def from_dict(d, overrides=()):
    return _wrap(apply_overrides(d, overrides))
