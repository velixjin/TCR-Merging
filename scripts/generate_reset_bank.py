"""Generate a policy-independent procedural LIBERO reset bank (CPU only).

Usage (after installation with the ``libero`` extra):

    python scripts/generate_reset_bank.py --output banks/libero-procedural-clean-v1

The default ``full`` profile creates 40 states for each of the 40 tasks in the four
standard LIBERO suites.  State ``i`` of task ``t`` in suite ``s`` is the simulator
state after ``reset()`` with seed ``275000000 + s*100000 + t*1000 + i``, captured in a
fresh ``ControlEnv``.  Indices 0-29 form three ten-state evaluation repeats and 30-39
are reserves.  Every state is rejected if it is non-finite, has the wrong shape, starts
or becomes (after 10 no-op steps) successful, duplicates an official or earlier bank
state, or is not reproduced by a second fresh environment with the same seed.

The ``preflight`` profile generates a subset (``--task SUITE:ID``, fewer states, sampled
checks) for quick checks.  LIBERO's official ``.pruned_init`` files are never modified
and the bank is published with a single directory rename.
"""

import argparse
import inspect
import json
import multiprocessing
import os
import platform
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from tcr_merging.pi05.reset_bank import (
    BANK_ID,
    EVAL_SEEDS,
    NOOP_ACTION,
    NOOP_STEPS,
    REPEATS,
    SEED_BASE,
    SEED_FORMULA,
    SEED_SUITE_STRIDE,
    SEED_TASK_STRIDE,
    STATES_PER_TASK,
    SUITES,
    canonical_state,
    repeat_mapping,
    role_for,
    seed_for,
    sha256_file,
    sha256_state,
    task_key,
    write_manifest,
    write_selections,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "libero_reset_bank.json"
ENVIRONMENT_LIFETIME = {
    "capture": "fresh_control_env_per_state",
    "same_seed_replay": "separate_fresh_control_env_per_checked_state",
    "save_restore_noop": "capture_control_env_before_close",
}


def audit_indices(state_count, mode, sampled_count):
    if mode == "all":
        return set(range(state_count))
    if mode != "sampled":
        raise ValueError(f"Unsupported audit mode: {mode}")
    if sampled_count <= 0:
        raise ValueError("sampled_count must be positive")
    if state_count <= sampled_count:
        return set(range(state_count))
    return {int(round(v)) for v in np.linspace(0, state_count - 1, num=sampled_count)}


def parse_task_binding(raw):
    if ":" not in raw:
        raise argparse.ArgumentTypeError("Task binding must be SUITE:TASK_ID")
    suite, raw_id = raw.rsplit(":", 1)
    if suite not in SUITES:
        raise argparse.ArgumentTypeError(f"Unknown suite {suite!r}")
    try:
        task_id = int(raw_id)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid task ID {raw_id!r}") from exc
    if not 0 <= task_id < 10:
        raise argparse.ArgumentTypeError("LIBERO source-suite task ID must be in [0, 10)")
    return suite, task_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("full", "preflight"),
        default="full",
        help="full enforces four suites, ten tasks each, and 40 states per task",
    )
    parser.add_argument(
        "--task",
        action="append",
        type=parse_task_binding,
        default=[],
        metavar="SUITE:TASK_ID",
        help="preflight-only task filter; repeat for multiple tasks",
    )
    parser.add_argument("--states-per-task", type=int, default=STATES_PER_TASK)
    parser.add_argument("--determinism-check", choices=("all", "sampled"), default="all")
    parser.add_argument("--restore-check", choices=("all", "sampled"), default="all")
    parser.add_argument("--sampled-checks-per-task", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4, help="Task-level CPU workers in [1, 8]")
    return parser.parse_args(argv)


def _package_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _software_receipt():
    import mujoco
    import robosuite
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": getattr(mujoco, "__version__", _package_version("mujoco")),
        "robosuite": getattr(robosuite, "__version__", _package_version("robosuite")),
        "libero_distribution": _package_version("libero") or _package_version("hf-libero"),
        "platform": platform.platform(),
    }


def load_config(path):
    """Read the generator config and refuse values that differ from this code."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    generation = payload.get("state_generation", {})
    checks = {
        "schema_version": (payload.get("schema_version"), 1),
        "bank_id": (payload.get("bank_id"), BANK_ID),
        "suite_order": (payload.get("suite_order"), list(SUITES)),
        "seed_base": (generation.get("seed_base"), SEED_BASE),
        "suite_stride": (generation.get("suite_stride"), SEED_SUITE_STRIDE),
        "task_stride": (generation.get("task_stride"), SEED_TASK_STRIDE),
        "workers": (generation.get("workers"), 4),
        "worker_limit": (generation.get("worker_limit"), 8),
        "multiprocessing_start_method": (generation.get("multiprocessing_start_method"), "spawn"),
        "environment_lifetime": (generation.get("environment_lifetime"), ENVIRONMENT_LIFETIME),
        "state_indices": (generation.get("state_indices"), repeat_mapping(STATES_PER_TASK)),
        "reserve_selection": (generation.get("reserve_selection"), "smallest_unused_state_index"),
        "evaluation_seeds": (
            [payload.get("evaluation", {}).get(r, {}).get("seed") for r in REPEATS],
            list(EVAL_SEEDS),
        ),
    }
    mismatches = {name: seen for name, (seen, expected) in checks.items() if seen != expected}
    if mismatches:
        raise ValueError(f"Reset-bank config differs from code: {mismatches}")
    return payload


def _reset_state(env, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    env.seed(seed)
    env.reset()
    return canonical_state(env.get_sim_state())


def _validate_initial_state(state, expected_shape, official_hashes, bank_hashes, env):
    value = canonical_state(state)
    if value.shape != expected_shape:
        raise ValueError(f"State shape {value.shape} differs from official shape {expected_shape}")
    if bool(env.check_success()):
        raise ValueError("Procedural reset starts in a successful terminal state")
    state_hash = sha256_state(value)
    if state_hash in official_hashes:
        raise ValueError("Procedural reset duplicates an official initial state")
    if state_hash in bank_hashes:
        raise ValueError("Procedural reset duplicates an earlier state in this bank")
    return state_hash


def _verify_restore_and_noops(env, state, expected_hash):
    env.set_init_state(state.copy())
    restored_hash = sha256_state(canonical_state(env.get_sim_state()))
    if restored_hash != expected_hash:
        raise ValueError(
            f"save/restore changed simulator state: expected={expected_hash} "
            f"observed={restored_hash}"
        )
    if bool(env.check_success()):
        raise ValueError("Restored procedural state is initially successful")
    noop = np.asarray(NOOP_ACTION, dtype=np.float32)
    for _ in range(NOOP_STEPS):
        env.step(noop)
        if not bool(np.isfinite(canonical_state(env.get_sim_state())).all()):
            raise ValueError("Non-finite simulator state after noop settling")
    if bool(env.check_success()):
        raise ValueError(f"Procedural state becomes successful after {NOOP_STEPS} noops")
    return {"restored_raw_sha256": restored_hash, "noop_steps": NOOP_STEPS, "finite": True}


def _official_state_receipt(path):
    import torch

    array = np.asarray(torch.load(path, weights_only=False))  # trusted LIBERO asset
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Unexpected official init-state array at {path}: {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"Official init-state file contains non-finite values: {path}")
    hashes = [sha256_state(row) for row in array]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"Official init-state file contains duplicates: {path}")
    return array, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "state_count": int(array.shape[0]),
        "state_shape": list(array.shape[1:]),
        "state_dtype": array.dtype.str,
        "raw_state_sha256": hashes,
    }


def _task_specs(bindings):
    if bindings:
        unique = sorted(set(bindings), key=lambda item: (SUITES.index(item[0]), item[1]))
        if len(unique) != len(bindings):
            raise ValueError("Duplicate --task binding")
        return [(suite, SUITES.index(suite), task_id) for suite, task_id in unique]
    return [(suite, i, task_id) for i, suite in enumerate(SUITES) for task_id in range(10)]


@dataclass(frozen=True)
class TaskJob:
    suite: str
    suite_index: int
    task_id: int
    task_name: str
    language: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    bddl_path: str
    official_receipt: dict
    expected_shape: tuple
    states_per_task: int
    determinism_indices: tuple
    restore_indices: tuple

    @property
    def task_key(self):
        return task_key(self.suite, self.task_id)


def _new_control_env(bddl_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only workers
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
    )


def _generate_one_state(job, state_index, task_hashes, env_factory=_new_control_env):
    seed = seed_for(job.suite_index, job.task_id, state_index)
    where = f"{job.task_key} state_index={state_index} seed={seed}"
    capture_env = None
    try:
        capture_env = env_factory(job.bddl_path)
        state = _reset_state(capture_env, seed)
        state_hash = _validate_initial_state(
            state,
            job.expected_shape,
            set(job.official_receipt["raw_state_sha256"]),
            task_hashes,
            capture_env,
        )
        restore_checked = state_index in job.restore_indices
        restore_receipt = (
            _verify_restore_and_noops(capture_env, state, state_hash) if restore_checked else None
        )
    except Exception as exc:
        raise RuntimeError(f"{where}: capture/restore validation failed: {exc!r}") from exc
    finally:
        if capture_env is not None:
            capture_env.close()

    determinism_checked = state_index in job.determinism_indices
    if determinism_checked:
        replay_env = None
        try:
            # A second, freshly constructed ControlEnv is required: reusing the capture
            # env leaves task/model sampler state mutable across resets.
            replay_env = env_factory(job.bddl_path)
            repeated = _reset_state(replay_env, seed)
            if repeated.shape != job.expected_shape:
                raise ValueError(f"repeat shape={repeated.shape} expected={job.expected_shape}")
            if bool(replay_env.check_success()):
                raise ValueError("same-seed fresh replay starts successful")
            repeated_hash = sha256_state(repeated)
            if repeated_hash != state_hash:
                mismatched = int(np.count_nonzero(repeated != state))
                raise ValueError(
                    f"same seed in a fresh ControlEnv gave a different state: {state_hash} != "
                    f"{repeated_hash}; mismatched_coordinates={mismatched}"
                )
        except Exception as exc:
            raise RuntimeError(f"{where}: fresh-env determinism replay failed: {exc!r}") from exc
        finally:
            if replay_env is not None:
                replay_env.close()

    return state, {
        "index": state_index,
        "role": role_for(state_index),
        "seed": seed,
        "raw_sha256": state_hash,
        "same_seed_regeneration_checked": determinism_checked,
        "same_seed_regeneration_env": "fresh_control_env",
        "save_restore_noop_checked": restore_checked,
        "save_restore_env": "capture_control_env",
        "restore_receipt": restore_receipt,
    }


def _generate_task(job):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    task_hashes, states, records = set(), [], []
    for state_index in range(job.states_per_task):
        state, record = _generate_one_state(job, state_index, task_hashes)
        task_hashes.add(record["raw_sha256"])
        states.append(state)
        records.append(record)
    return job, np.ascontiguousarray(np.stack(states, axis=0)), records


def _prepare_task_jobs(args, task_specs):
    from libero.libero import benchmark, get_libero_path

    def indices(mode):
        return tuple(
            sorted(audit_indices(args.states_per_task, mode, args.sampled_checks_per_task))
        )

    determinism_indices = indices(args.determinism_check)
    restore_indices = indices(args.restore_check)
    jobs = []
    for suite, suite_index, task_id in task_specs:
        try:
            task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
            bddl_path = (
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ).resolve()
            official_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / Path(task.init_states_file).name
            ).resolve()
            if not bddl_path.is_file() or not official_path.is_file():
                raise FileNotFoundError("missing BDDL or official init file")
            official_states, official_receipt = _official_state_receipt(official_path)
        except Exception as exc:
            raise RuntimeError(f"{task_key(suite, task_id)}: task input check failed: {exc!r}")
        jobs.append(
            TaskJob(
                suite=suite,
                suite_index=suite_index,
                task_id=task_id,
                task_name=task.name,
                language=task.language,
                problem_folder=task.problem_folder,
                bddl_file=task.bddl_file,
                init_states_file=Path(task.init_states_file).name,
                bddl_path=str(bddl_path),
                official_receipt=official_receipt,
                expected_shape=tuple(official_states.shape[1:]),
                states_per_task=args.states_per_task,
                determinism_indices=determinism_indices,
                restore_indices=restore_indices,
            )
        )
    return jobs


def _run_task_jobs(jobs, workers):
    executor = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    futures = [executor.submit(_generate_task, job) for job in jobs]
    try:
        # Consume in task order even when workers finish out of order.
        results = [future.result() for future in futures]
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


def check_args(args):
    if not 0 < args.states_per_task <= STATES_PER_TASK:
        raise ValueError(f"states-per-task must lie in [1, {STATES_PER_TASK}]")
    if args.profile == "full":
        if args.task or args.states_per_task != STATES_PER_TASK:
            raise ValueError("full profile requires all four suites and 40 states per task")
        if args.determinism_check != "all" or args.restore_check != "all":
            raise ValueError("full profile requires all-state determinism and restore checks")
    elif not args.task:
        raise ValueError("preflight profile requires at least one explicit --task")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must lie in [1, 8]")


def _build_manifest(args, staging):
    from libero.libero.envs.env_wrapper import ControlEnv

    config_path = args.config.resolve()
    config = load_config(config_path)
    check_args(args)
    generator_path = Path(__file__).resolve()
    control_source = Path(inspect.getsourcefile(ControlEnv) or "").resolve()
    manifest = {
        "schema_version": 1,
        "bank_id": BANK_ID,
        "profile": args.profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_benchmark_bank": False,
        "prospective_policy_independent_generation": True,
        "claim_boundary": (
            "Custom prospective procedural reset bank generated before method ranking; "
            "not the canonical 50-state LIBERO benchmark bank."
        ),
        "generator": {
            "seed_formula": SEED_FORMULA,
            "seed_base": SEED_BASE,
            "suite_stride": SEED_SUITE_STRIDE,
            "task_stride": SEED_TASK_STRIDE,
            "suite_order": list(SUITES),
            "states_per_task": args.states_per_task,
            "workers": args.workers,
            "parallelism": "spawn_process_per_task_bounded_pool",
            "environment_lifetime": ENVIRONMENT_LIFETIME,
            "final_eval_seeds": list(EVAL_SEEDS),
            "reserve_policy": {
                "indices": list(range(30, min(args.states_per_task, 40))),
                "selection": "smallest_unused_state_index",
                "trigger": "predeclared_invalid_state_or_infrastructure_receipt_only",
                "forbidden_trigger": "policy_outcome_or_method_ranking",
            },
            "rejection_rules": [
                "non_finite",
                "shape_mismatch",
                "initial_success",
                "official_state_collision",
                "procedural_bank_collision",
                "same_seed_hash_mismatch",
                "save_restore_hash_mismatch",
                "non_finite_after_noops",
                "success_after_noops",
            ],
            "determinism_check": {
                "mode": args.determinism_check,
                "sampled_checks_per_task": args.sampled_checks_per_task,
            },
            "restore_check": {
                "mode": args.restore_check,
                "sampled_checks_per_task": args.sampled_checks_per_task,
                "noop_steps": NOOP_STEPS,
                "noop_action": list(NOOP_ACTION),
            },
            "repeat_mapping": repeat_mapping(args.states_per_task),
            "control_env": {
                "use_camera_obs": False,
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "gpu_required": False,
            },
        },
        "software": _software_receipt(),
        "code": {
            "generator_path": str(generator_path),
            "generator_sha256": sha256_file(generator_path),
            "frozen_config_path": str(config_path),
            "frozen_config_sha256": sha256_file(config_path),
            "control_env_source_path": str(control_source),
            "control_env_source_sha256": sha256_file(control_source),
        },
        "frozen_config": config,
        "suites": list(SUITES),
        "tasks": {},
    }

    jobs = _prepare_task_jobs(args, _task_specs(args.task))
    bank_hashes = set()
    for job, states, records in _run_task_jobs(jobs, args.workers):
        for record in records:
            if record["raw_sha256"] in bank_hashes:
                raise RuntimeError(
                    f"{job.task_key} state_index={record['index']} seed={record['seed']}: "
                    f"cross-task bank collision {record['raw_sha256']}"
                )
            bank_hashes.add(record["raw_sha256"])
        relative = Path("states") / job.suite / f"task-{job.task_id:02d}.npy"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.save(stream, states, allow_pickle=False)
        bddl_path = Path(job.bddl_path)
        manifest["tasks"][job.task_key] = {
            "suite": job.suite,
            "suite_index": job.suite_index,
            "task_id": job.task_id,
            "task_name": job.task_name,
            "language": job.language,
            "problem_folder": job.problem_folder,
            "bddl_file": job.bddl_file,
            "init_states_file": job.init_states_file,
            "bddl": {"path": str(bddl_path), "sha256": sha256_file(bddl_path)},
            "official_init": job.official_receipt,
            "states_file": relative.as_posix(),
            "states_file_sha256": sha256_file(destination),
            "state_shape": list(states.shape[1:]),
            "state_dtype": states.dtype.str,
            "states": records,
        }
        print(json.dumps({"task": job.task_key, "accepted": len(records)}), flush=True)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    check_args(args)
    load_config(args.config)
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing existing reset-bank output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        manifest = _build_manifest(args, staging)
        manifest_sha = write_manifest(staging, manifest)
        write_selections(staging, manifest, manifest_sha)
        # mkdtemp creates a 0700 directory; publish with ordinary umask permissions.
        umask = os.umask(0)
        os.umask(umask)
        staging.chmod(0o777 & ~umask)
        if output.exists():  # rename(2) would silently replace an empty directory
            raise FileExistsError(f"Output appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "manifest_sha256": manifest_sha,
                "tasks": len(manifest["tasks"]),
                "states": sum(len(row["states"]) for row in manifest["tasks"].values()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
