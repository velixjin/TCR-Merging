"""Exact primary-source RegMean++ Linear solver for the pi0.5 port."""

from __future__ import annotations

import math
from typing import Any

import torch


def _task_loss(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> float:
    residual = x @ (weight - target).T
    return float((residual * residual).mean())


def solve_original_regmeanpp_weight(
    inputs: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    metric_reference: torch.Tensor,
    *,
    offdiag_scale: float,
    residual_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve ``sum_i Gtilde_i W_i^T`` with no prior or regularizer.

    The metric reference is the equal expert mean. It is only used to report
    the offline improvement and never enters the solved equation.
    """
    names = tuple(inputs)
    if len(names) < 2 or set(weights) != set(names):
        raise ValueError("RegMean++ requires at least two matched experts")
    if not 0.0 < offdiag_scale < 1.0:
        raise ValueError("offdiag_scale must be in (0, 1)")
    if not 0.0 < residual_tolerance < 1.0:
        raise ValueError("residual_tolerance must be in (0, 1)")

    device = metric_reference.device
    output_width, input_width = map(int, metric_reference.shape)
    prepared_inputs: dict[str, torch.Tensor] = {}
    prepared_weights: dict[str, torch.Tensor] = {}
    rows_by_expert: dict[str, int] = {}
    for name in names:
        x = inputs[name].to(device=device, dtype=torch.float64)
        weight = weights[name].to(device=device, dtype=torch.float64)
        if x.ndim != 2 or tuple(weight.shape) != (output_width, input_width):
            raise ValueError(f"{name}: RegMean++ input/weight shape differs")
        if int(x.shape[1]) != input_width or int(x.shape[0]) <= 0:
            raise ValueError(f"{name}: RegMean++ feature shape differs")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"{name}: RegMean++ input is non-finite")
        prepared_inputs[name] = x
        prepared_weights[name] = weight
        rows_by_expert[name] = int(x.shape[0])
    if len(set(rows_by_expert.values())) != 1:
        raise ValueError(f"Original RegMean++ requires equal rows: {rows_by_expert}")

    total_rows = sum(rows_by_expert.values())
    mean_weight_t = (
        torch.stack([prepared_weights[name] for name in names]).mean(dim=0).T.contiguous()
    )
    strategy = "exact_active_support_primal_general_solve"
    if input_width <= total_rows:
        matrix = torch.zeros((input_width, input_width), dtype=torch.float64, device=device)
        rhs = torch.zeros((input_width, output_width), dtype=torch.float64, device=device)
        for name in names:
            x = prepared_inputs[name]
            weight_t = prepared_weights[name].T
            gram = x.T @ x
            diagonal = torch.diagonal(gram).clone()
            gram.mul_(offdiag_scale)
            torch.diagonal(gram).add_((1.0 - offdiag_scale) * diagonal)
            matrix.add_(gram)
            rhs.add_(gram @ weight_t)
        support_diagonal = torch.diagonal(matrix)
        if not bool(torch.isfinite(support_diagonal).all()) or bool(
            torch.any(support_diagonal < 0.0)
        ):
            raise RuntimeError("Original RegMean++ support diagonal is invalid")
        active = support_diagonal > 0.0
        active_count = int(active.sum())
        solved_t = mean_weight_t.clone()
        if active_count:
            try:
                solved_t[active] = torch.linalg.solve(matrix[active][:, active], rhs[active])
            except RuntimeError as exc:
                raise RuntimeError(
                    "Original RegMean++ active-support system is singular; ridge/pinv is forbidden"
                ) from exc
        residual = matrix @ solved_t - rhs
        rowspace_shape = None
    else:
        strategy = "exact_active_support_diagonal_plus_woodbury_rowspace"
        diagonal_by_expert = {name: x.square().sum(dim=0) for name, x in prepared_inputs.items()}
        diagonal = (1.0 - offdiag_scale) * sum(diagonal_by_expert.values())
        if not bool(torch.isfinite(diagonal).all()) or bool(torch.any(diagonal < 0.0)):
            raise RuntimeError(
                "Original RegMean++ support diagonal is invalid; no fallback is allowed"
            )
        active = diagonal > 0.0
        active_count = int(active.sum())
        active_diagonal = diagonal[active]
        root_alpha = math.sqrt(offdiag_scale)
        z_parts = [root_alpha * prepared_inputs[name][:, active] for name in names]
        z = torch.cat(z_parts, dim=0)
        scaled = z * torch.rsqrt(active_diagonal).unsqueeze(0)
        kernel = scaled @ scaled.T
        torch.diagonal(kernel).add_(1.0)
        cholesky, info = torch.linalg.cholesky_ex(kernel)
        if bool(torch.any(info != 0)):
            raise RuntimeError("Original RegMean++ exact row-space kernel failed Cholesky")
        rhs = torch.zeros((active_count, output_width), dtype=torch.float64, device=device)
        for name, z_expert in zip(names, z_parts, strict=True):
            weight_t = prepared_weights[name].T[active]
            rhs.add_(z_expert.T @ (z_expert @ weight_t))
            rhs.add_(
                (1.0 - offdiag_scale) * diagonal_by_expert[name][active].unsqueeze(1) * weight_t
            )
        base = rhs / active_diagonal.unsqueeze(1)
        dual = torch.cholesky_solve(z @ base, cholesky)
        solved_active = base - (z.T @ dual) / active_diagonal.unsqueeze(1)
        solved_t = mean_weight_t.clone()
        solved_t[active] = solved_active
        residual = active_diagonal.unsqueeze(1) * solved_active + z.T @ (z @ solved_active) - rhs
        rowspace_shape = [int(z.shape[0]), int(z.shape[0])]

    relative_residual = float(torch.linalg.vector_norm(residual)) / max(
        float(torch.linalg.vector_norm(rhs)), torch.finfo(torch.float64).tiny
    )
    if not math.isfinite(relative_residual) or relative_residual > residual_tolerance:
        raise RuntimeError(f"Original RegMean++ residual differs: {relative_residual}")
    merged = solved_t.T.to(dtype=torch.float32)
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("Original RegMean++ solution is non-finite")

    reference = metric_reference.to(device=device, dtype=torch.float32)
    merged_losses = {
        name: _task_loss(prepared_inputs[name].float(), merged, prepared_weights[name].float())
        for name in names
    }
    reference_losses = {
        name: _task_loss(prepared_inputs[name].float(), reference, prepared_weights[name].float())
        for name in names
    }
    merged_loss = sum(merged_losses.values()) / len(names)
    reference_loss = sum(reference_losses.values()) / len(names)
    return merged.detach().cpu(), {
        "method": "original_regmean_pp",
        "strategy": strategy,
        "offdiag_scale": offdiag_scale,
        "solve_dtype": "torch.float64",
        "equal_expert_weighting": True,
        "raw_xtx_no_normalization": True,
        "prior_ridge": False,
        "pseudo_inverse": False,
        "relative_weighting": False,
        "correction_cap": False,
        "rejection_gate": False,
        "rows_by_expert": rows_by_expert,
        "input_width": input_width,
        "output_width": output_width,
        "relative_residual": relative_residual,
        "rowspace_kernel_shape": rowspace_shape,
        "active_support_columns": active_count,
        "inactive_zero_support_columns": input_width - active_count,
        "inactive_tie_break": "equal_expert_weight_mean",
        "expert_objective_weights": {name: 1.0 / len(names) for name in names},
        "prior_loss": reference_loss,
        "dense_regmean_loss": merged_loss,
        "prior_loss_by_expert": reference_losses,
        "dense_regmean_loss_by_expert": merged_losses,
        "improvement_vs_prior": (reference_loss - merged_loss) / max(reference_loss, 1e-12),
        "trust_scale": 1.0,
        "rejected_nonimproving": False,
    }


# --------------------------------------------------------------------------------------------
# Spectrally smoothed RegMean++ (the variant reported in the paper's main table)
# --------------------------------------------------------------------------------------------
#
# Centered at the arithmetic expert mean W_bar, RegMean++ solves  A delta = R  with
#   A = (1 - a) diag(sum_i X_i^T X_i) + a sum_i X_i^T X_i
#   R = sum_i [a X_i^T X_i + (1 - a) diag(X_i^T X_i)] (W_i - W_bar)^T
# and returns W_bar + delta^T.  The pi0.5 Gram matrices are numerically rank deficient, so the
# inverse is replaced by the smooth spectral filter  lambda / (lambda^2 + tau^2), i.e.
# delta = Re[(A + i tau I)^{-1} R], with tau = d * eps(float32) * lambda_max(A).  Directions with
# no feature energy keep the expert mean.  No ridge toward a prior, trust region or clipping.

_POWER_SEED = 20260922


def _relative(numerator: float, denominator: float) -> float:
    numerator, denominator = float(numerator), float(denominator)
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def _power_lambda_max(diagonal: torch.Tensor, z: torch.Tensor, iterations: int = 64):
    """Deterministic power-iteration estimate of lambda_max(diag(d) + Z^T Z) (no dense D x D)."""
    if not bool(torch.any(diagonal > 0)) and not bool(torch.any(z != 0)):
        return 0.0, 0
    generator = torch.Generator(device="cpu").manual_seed(_POWER_SEED)
    v = torch.randn(diagonal.numel(), generator=generator, dtype=torch.float64).to(diagonal.device)
    v /= torch.linalg.vector_norm(v)
    history: list[float] = []
    for _ in range(iterations):
        av = diagonal * v + z.T @ (z @ v)
        length = torch.linalg.vector_norm(av)
        if float(length) == 0:
            raise RuntimeError("Nonzero PSD operator annihilated the power-iteration start")
        v = av / length
        av = diagonal * v + z.T @ (z @ v)
        history.append(float(torch.dot(v, av)))
        if len(history) >= 4 and all(
            abs(history[-j] - history[-j - 1]) <= 1e-12 * history[-1] for j in (1, 2, 3)
        ):
            break
    if not math.isfinite(history[-1]) or history[-1] <= 0:
        raise RuntimeError("Invalid spectral scale estimate")
    return history[-1], len(history)


def _centered_equation(inputs, weights, offdiag_scale, device):
    names = list(inputs)
    if len(names) < 1 or set(names) != set(weights):
        raise ValueError("Input/expert identities differ")
    shape = tuple(weights[names[0]].shape)
    xs = {n: inputs[n].to(device=device, dtype=torch.float64) for n in names}
    ws = {n: weights[n].to(device=device, dtype=torch.float64) for n in names}
    counts = {n: int(xs[n].shape[0]) for n in names}
    if len(set(counts.values())) != 1 or not all(c > 0 for c in counts.values()):
        raise ValueError("RegMean++ requires equal positive row counts per expert")
    for n in names:
        if xs[n].ndim != 2 or xs[n].shape[1] != shape[1] or tuple(ws[n].shape) != shape:
            raise ValueError(f"{n}: RegMean++ input/weight shape differs")
        if not bool(torch.isfinite(xs[n]).all()) or not bool(torch.isfinite(ws[n]).all()):
            raise ValueError(f"{n}: RegMean++ input is non-finite")
    mean = sum(ws.values()) / len(ws)
    energy = {n: xs[n].square().sum(0) for n in names}
    diagonal = (1 - offdiag_scale) * sum(energy.values())
    z = math.sqrt(offdiag_scale) * torch.cat(list(xs.values()), dim=0)
    rhs = torch.zeros(shape[1], shape[0], device=device, dtype=torch.float64)
    for n in names:
        delta = (ws[n] - mean).T
        rhs.add_(offdiag_scale * (xs[n].T @ (xs[n] @ delta)))
        rhs.add_((1 - offdiag_scale) * energy[n].unsqueeze(1) * delta)
    return diagonal, z, rhs, mean, counts


def solve_spectral_regmeanpp_weight(
    inputs: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    *,
    offdiag_scale: float = 0.3,
    strategy: str = "auto",
    power_iterations: int = 64,
    output_chunk_size: int = 128,
    residual_tolerance: float = 1e-7,
    device=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean-centered RegMean++ with the smooth spectral filter (float64 / complex128 solve)."""
    if not 0.0 < offdiag_scale < 1.0:
        raise ValueError("offdiag_scale must be in (0, 1)")
    if strategy not in ("auto", "dense", "woodbury"):
        raise ValueError("strategy must be auto, dense or woodbury")
    device = device or next(iter(inputs.values())).device
    d, z, r, mean, counts = _centered_equation(inputs, weights, offdiag_scale, device)
    width, rows = d.numel(), z.shape[0]
    scale, iterations = _power_lambda_max(d, z, power_iterations)
    tau = width * torch.finfo(torch.float32).eps * scale
    chosen = ("dense" if width <= rows else "woodbury") if strategy == "auto" else strategy
    delta = torch.zeros_like(r)
    if scale > 0 and bool(torch.any(r != 0)):
        if tau <= 0 or not math.isfinite(tau):
            raise RuntimeError("Invalid spectral threshold")
        if chosen == "dense":
            a = (torch.diag(d) + z.T @ z).to(torch.complex128)
            a.diagonal().add_(complex(0, tau))
            lu, piv = torch.linalg.lu_factor(a)
            del a
            for start in range(0, r.shape[1], output_chunk_size):
                stop = min(start + output_chunk_size, r.shape[1])
                block = r[:, start:stop].to(torch.complex128)
                delta[:, start:stop] = torch.linalg.lu_solve(lu, piv, block).real
        else:  # Woodbury identity over the (rows x rows) kernel
            inverse_diagonal = 1 / (d.to(torch.complex128) + complex(0, tau))
            complex_z = z.to(torch.complex128)
            kernel = (complex_z * inverse_diagonal.unsqueeze(0)) @ complex_z.T
            kernel.diagonal().add_(1)
            lu, piv = torch.linalg.lu_factor(kernel)
            del kernel
            for start in range(0, r.shape[1], output_chunk_size):
                stop = min(start + output_chunk_size, r.shape[1])
                base = inverse_diagonal.unsqueeze(1) * r[:, start:stop]
                dual = torch.linalg.lu_solve(lu, piv, complex_z @ base)
                delta[:, start:stop] = (
                    base - inverse_diagonal.unsqueeze(1) * (complex_z.T @ dual)
                ).real

    def apply(v):
        return d.unsqueeze(1) * v + z.T @ (z @ v)

    equation_residual = apply(delta) - r
    smoothed = apply(equation_residual) + (tau * tau) * delta
    residual = _relative(torch.linalg.vector_norm(smoothed), torch.linalg.vector_norm(apply(r)))
    if not torch.isfinite(delta).all() or not math.isfinite(residual):
        raise RuntimeError("Non-finite smoothed RegMean++ solution")
    if residual > residual_tolerance:
        raise RuntimeError(f"Smoothed normal-equation residual too large: {residual}")
    merged = (mean + delta.T).to(torch.float32).detach().cpu()
    if not torch.isfinite(merged).all():
        raise RuntimeError("Non-finite float32 RegMean++ output")
    return merged, {
        "method": "regmeanpp_spectral_smoothing",
        "offdiag_scale": offdiag_scale,
        "strategy": chosen,
        "tau": tau,
        "tau_rule": "input_width * eps(float32) * lambda_max (power iteration)",
        "lambda_max_estimate": scale,
        "power_iterations": iterations,
        "rows_by_expert": counts,
        "input_width": width,
        "output_width": int(r.shape[1]),
        "smoothed_normal_equation_relative_residual": residual,
        "centered_equation_relative_residual": _relative(
            torch.linalg.vector_norm(equation_residual), torch.linalg.vector_norm(r)
        ),
        "zero_energy_columns": int((d == 0).sum()),
        "solve_dtype": "float64/complex128",
        "prior_ridge": False,
        "correction_cap": False,
    }
