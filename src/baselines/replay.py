"""RegMean++ plugged into the shared native pi0.5 traversal.

The merged network is replayed in dependency order (the same traversal as TCR), so each linear
module sees inputs produced by the already-merged upstream modules. Only the per-module solve
differs from TCR: RegMean++ has no prior-centered ridge, no expert masses and no trust region.
"""

from pathlib import Path

import torch
from tcr_merging.baselines.regmeanpp import (
    solve_original_regmeanpp_weight,
    solve_spectral_regmeanpp_weight,
)

SOLVERS = ("spectral", "original")


class ReplayBaseline:
    # Hooks consulted by the shared engine (kept for interface compatibility).
    teacher_only = False
    student_inputs = False

    def __init__(self, cfg, output, settings, *, calibration_flow=None):
        if calibration_flow not in (None, 0, 5, 9):
            raise ValueError("calibration_flow must be None (all calls) or one of 0, 5, 9")
        self.calibration_flow = calibration_flow
        if settings["method"] != "regmeanpp" or settings["regmeanpp_solver"] not in SOLVERS:
            raise ValueError("ReplayBaseline supports RegMean++ (spectral | original) only")
        self.cfg = cfg
        self.output = Path(output)
        self.settings = settings
        self.method = settings["method"]
        self.description = (
            "RegMean++ (alpha off-diagonal Gram scaling) with spectral smoothing; "
            "merged-prefix replay; mean interface biases; no ridge/cap"
            if settings["regmeanpp_solver"] == "spectral"
            else "Original RegMean++ (exact active-support solve); merged-prefix replay; "
            "mean interface biases; no ridge/cap"
        )

    @staticmethod
    def order_groups(groups):
        # A solved producer must be visible to every subsequent consumer.
        def rank(name):
            suffixes = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "self_attn.o_proj",
                "mlp.fc1",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.fc2",
                "mlp.down_proj",
            )
            return next(i for i, suffix in enumerate(suffixes) if name.endswith(suffix))

        return {
            group: {i: sorted(names, key=rank) for i, names in blocks.items()}
            for group, blocks in groups.items()
        }

    def solve(self, module, inputs, weights, prior, *, bias=False):
        names = list(inputs)
        if set(names) != set(self.cfg["experts"]):
            raise ValueError("Baseline inputs must contain exactly the configured experts")
        rows = {n: len(inputs[n]) for n in names}
        if len(set(rows.values())) != 1:
            raise ValueError("RegMean++ requires equal calibration rows per expert")
        weights = {n: w.to(prior.device) for n, w in weights.items()}
        inputs = {n: x.to(device=prior.device, dtype=torch.float32) for n, x in inputs.items()}
        # Interfaces arrive with an augmented constant column; RegMean++ solves the weight only
        # and uses the expert-mean bias.
        x = {n: v[:, :-1] if bias else v for n, v in inputs.items()}
        w = {n: v[:, :-1] if bias else v for n, v in weights.items()}
        reference = prior[:, :-1] if bias else prior
        if self.settings["regmeanpp_solver"] == "spectral":
            merged, info = solve_spectral_regmeanpp_weight(
                x, w, offdiag_scale=self.settings["offdiag_scale"], device=prior.device
            )
            mean = (sum(w.values()) / len(names)).float()
            losses = {
                n: float((x[n] @ (merged.to(prior.device) - w[n].float()).T).square().mean())
                for n in names
            }
            reference_losses = {
                n: float((x[n] @ (mean - w[n].float()).T).square().mean()) for n in names
            }
            before = sum(reference_losses.values()) / len(names)
            after = sum(losses.values()) / len(names)
            info.update(
                dense_regmean_loss=after,
                mean_reference_loss=before,
                improvement_vs_prior=(before - after) / max(before, 1e-12),
            )
        else:
            merged, info = solve_original_regmeanpp_weight(
                x, w, reference, offdiag_scale=self.settings["offdiag_scale"]
            )
        if bias:
            mean_bias = sum(v[:, -1] for v in weights.values()) / len(names)
            merged = torch.cat((merged, mean_bias.cpu()[:, None]), dim=1)
            info["bias_semantics"] = "uniform_expert_mean_not_augmented_regression"
        return merged, info
