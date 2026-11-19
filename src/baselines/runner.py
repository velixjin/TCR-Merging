"""Portable dense pi0.5 baseline entry points. Never connects to a robot."""

import argparse
import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tcr_merging.config import load_config
from tcr_merging.pi05.checkpoints import copy_support, make_staging, sha256, validate_support
from tcr_merging.pi05.scope import target_keys


def parameter_merge(cfg, method, output):
    """Model Soups: equal mean of the dense experts over the adapted scope; no calibration."""
    if method != "soup":
        raise ValueError("Unknown parameter baseline")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    experts = {n: Path(e["checkpoint"]) for n, e in cfg["experts"].items()}
    validate_support(list(experts.values()))
    base_root = Path(cfg["base_model"])
    adapted = sorted(target_keys())
    with ExitStack() as stack:
        base = stack.enter_context(
            safe_open(base_root / "model.safetensors", framework="pt", device="cpu")
        )
        sources = {
            n: stack.enter_context(safe_open(p / "model.safetensors", framework="pt", device="cpu"))
            for n, p in experts.items()
        }
        keys = set(base.keys())
        if not set(adapted) <= keys or any(set(s.keys()) != keys for s in sources.values()):
            raise ValueError("Base/expert schema differs from the supported pi0.5 scope")
        for root in [base_root, *experts.values()]:
            if json.loads((root / "config.json").read_text()).get("use_peft", False):
                raise ValueError("Materialize LoRA as a dense checkpoint first")
        tensors = {}
        for key in sorted(keys):
            b = base.get_tensor(key)
            values = [source.get_tensor(key) for source in sources.values()]
            if not torch.isfinite(b).all() or any(
                v.shape != b.shape or v.dtype != b.dtype or not torch.isfinite(v).all()
                for v in values
            ):
                raise ValueError(f"Incompatible or nonfinite tensor: {key}")
            if key not in adapted:
                if any(not torch.equal(v, b) for v in values):
                    raise ValueError(f"Adaptation outside supported scope: {key}")
                tensors[key] = b
            else:
                tensors[key] = (sum(v.float() for v in values) / len(values)).to(b.dtype)
        info = {"weighting": "uniform", "calibration_used": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = make_staging(output.parent, ".baseline-staging-")
        copy_support(next(iter(experts.values())), staging, output)
        save_file({k: v.contiguous() for k, v in tensors.items()}, staging / "model.safetensors")
    report = dict(
        method=method,
        created_at=datetime.now(timezone.utc).isoformat(),
        base_model_sha256=sha256(base_root / "model.safetensors"),
        expert_sha256={n: sha256(p / "model.safetensors") for n, p in experts.items()},
        model_sha256=sha256(staging / "model.safetensors"),
        modified_tensor_count=len(adapted),
        config=cfg,
        settings=info,
    )
    (staging / "baseline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.chmod(0o755)
    staging.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("soup", "regmeanpp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regmeanpp-solver", choices=("spectral", "original"), default="spectral")
    parser.add_argument("--offdiag-scale", type=float, default=0.3)
    parser.add_argument(
        "--calibration-calls",
        choices=("all", "first"),
        default="all",
        help="RegMean++: use all cached generation calls (0/5/9) or only the first (paper)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, baseline=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if not math.isfinite(args.offdiag_scale) or not 0 < args.offdiag_scale < 1:
        raise ValueError("offdiag_scale must lie in (0, 1)")
    settings = {
        k: getattr(args, k)
        for k in ("method", "regmeanpp_solver", "offdiag_scale", "calibration_calls")
    }
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    method=args.method,
                    output=str(output),
                    settings=settings if args.method == "regmeanpp" else {},
                    calibration="none" if args.method == "soup" else "cache_a; one pass",
                    rows_per_call=cfg["rows_per_call"],
                    dry_run=True,
                ),
                indent=2,
            )
        )
        return
    if args.method == "soup":
        report = parameter_merge(cfg, "soup", output)
    else:
        from tcr_merging.baselines.replay import ReplayBaseline
        from tcr_merging.merging.engine import merge_pass

        cfg.update(passes=1, first_pass_weighting="none", output=str(output))
        if args.calibration_calls == "first" or cfg["rows_per_call"] != 10:
            cfg["strict_paper_budget"] = False  # the TCR row quotas do not apply
        flow = 0 if args.calibration_calls == "first" else None
        backend = ReplayBaseline(cfg, output, settings, calibration_flow=flow)
        report = merge_pass(cfg, 1, backend=backend)
    print(
        json.dumps(
            {
                "method": args.method,
                "output": str(output),
                "model_sha256": report.get("model_sha256"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
