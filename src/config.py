"""Portable paths and explicit experiment settings; no machine-specific defaults.

Machine-specific locations live in a local environment file, never in tracked configs. Configs refer to them as
``${VAR}``; :func:`load_env_file` reads ``$TCR_ENV_FILE`` or ``./.env`` (``KEY=VALUE`` lines). Variables already
set in the process environment take precedence. See ``.env.example``.
"""

import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def load_env_file(path=None):
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding existing values. Returns the file used."""
    path = Path(path or os.environ.get("TCR_ENV_FILE") or ".env").expanduser()
    if not path.is_file():
        if os.environ.get("TCR_ENV_FILE"):
            raise FileNotFoundError(f"TCR_ENV_FILE points to a missing file: {path}")
        return None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return path


def expand_vars(value, where):
    """Expand ``$VAR`` / ``${VAR}``; an unset variable is an error, not an empty string."""

    def sub(match):
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise KeyError(
                f"{where}: environment variable {name} is not set (define it in .env; see .env.example)"
            )
        return os.environ[name]

    return _VAR.sub(sub, value)


DEFAULTS = dict(
    device="cuda",
    passes=2,
    variant="full",
    ridge_ratio=0.05,
    max_correction_ratio=3.0,
    rows_per_call=10,
    rows_per_request=16,
    first_pass_weighting="prior",
    strict_paper_budget=True,
)
# "full": frozen-expert execution caches (TCR). "demo": demonstration-observation caches (baselines only).
VARIANTS = ("full", "demo")
CACHE_KEYS = ("cache_a", "cache_b", "cache_c")


def load_config(path, *, baseline=False):
    """Load and validate a JSON config. ``baseline=True`` allows one-pass demo-input baselines."""
    load_env_file()
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    allowed = {*DEFAULTS, "base_model", "experts", "output"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = {**DEFAULTS, **deepcopy(raw)}

    def resolve(value):
        p = Path(expand_vars(value, path.name)).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())

    for key in ("base_model", "output"):
        cfg[key] = resolve(cfg[key])
    experts = cfg.get("experts")
    if not isinstance(experts, dict) or len(experts) < 2:
        raise ValueError("At least two named experts are required")
    if cfg["passes"] not in (1, 2, 3):
        raise ValueError("passes must be 1, 2 or 3")
    cache_keys = CACHE_KEYS[: cfg["passes"]]
    required = {"checkpoint", *cache_keys}
    for name, entry in experts.items():
        if not name or name != name.strip().lower() or not isinstance(entry, dict):
            raise ValueError("Expert names must be nonempty lowercase strings")
        if not required <= set(entry) or set(entry) - {"checkpoint", *CACHE_KEYS}:
            raise ValueError(f"{name}: expected checkpoint and {', '.join(cache_keys)}")
        for key in entry:
            entry[key] = resolve(entry[key])
        if len({entry[key] for key in cache_keys}) != len(cache_keys):
            raise ValueError(f"{name}: each pass needs its own cache")
    if cfg["variant"] not in VARIANTS:
        raise ValueError("Unknown variant")
    if not baseline and cfg["variant"] != "full":
        raise ValueError(
            "TCR merges frozen-expert execution caches (variant=full); demo is for baselines"
        )
    if cfg["first_pass_weighting"] not in ("prior", "none"):
        raise ValueError("first_pass_weighting must be prior or none")
    for key in ("ridge_ratio", "max_correction_ratio"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("rows_per_call", "rows_per_request"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["strict_paper_budget"] and (
        len(experts) != 4 or cfg["rows_per_call"] != 10 or cfg["rows_per_request"] != 16
    ):
        raise ValueError(
            "The paper budget requires four experts and row caps 10/16; disable strict_paper_budget for other budgets"
        )
    return cfg


def validate_inputs(cfg):
    from tcr_merging.calibration.traces import validate_trace
    from tcr_merging.pi05.checkpoints import validate_support

    roots = [Path(e["checkpoint"]) for e in cfg["experts"].values()]
    for root in [Path(cfg["base_model"]), *roots]:
        for name in ("model.safetensors", "config.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        if json.loads((root / "config.json").read_text()).get("use_peft", False):
            raise ValueError(f"{root}: materialize the adapter as a dense checkpoint first")
    validate_support(roots)
    for name, e in cfg["experts"].items():
        for index, key in enumerate(CACHE_KEYS[: cfg["passes"]], 1):
            cache = Path(e[key])
            if not (cache / "replay.safetensors").is_file():
                raise FileNotFoundError(cache / "replay.safetensors")
            manifest = json.loads((cache / "replay.json").read_text())
            validate_trace(manifest, e["checkpoint"], name, index, cfg)
    return cfg
