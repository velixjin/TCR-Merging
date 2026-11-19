"""Evaluate one checkpoint on LIBERO.

``main`` (``tcr evaluate``) uses LIBERO's stock initial states from an explicit offset;
``bank_main`` (``tcr evaluate-bank``) reproduces the paper protocol on a procedural
reset bank (see ``tcr_merging.pi05.reset_bank``).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def eval_arguments(policy, suite, output, episodes, seed, device):
    return [
        f"--policy.path={Path(policy).resolve()}",
        f"--policy.device={device}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        f"--env.task={suite}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={episodes}",
        f"--seed={seed}",
        f"--output_dir={Path(output).resolve()}",
    ]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-state-offset", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    from lerobot.scripts import lerobot_eval
    from tcr_merging.pi05.libero import runtime_compatibility

    previous = sys.argv
    sys.argv = [
        "lerobot-eval",
        *eval_arguments(
            args.policy, args.suite, args.output, args.episodes, args.seed, args.device
        ),
    ]
    try:
        with runtime_compatibility(args.init_state_offset):
            lerobot_eval.main()
    finally:
        sys.argv = previous


def bank_eval_arguments(policy, suite, output, episodes, seed, device):
    """LeRobot arguments of one paper reset-bank evaluation (one suite, one repeat)."""
    return [
        *eval_arguments(policy, suite, output, episodes, seed, device),
        "--env.init_states=true",
        "--env.hard_reset=true",
    ]


def selected_episodes(selection, suite):
    """Episodes per task for ``suite``; every one of its ten tasks must be bound equally."""
    tasks = selection.suite_tasks(suite)
    if set(tasks) != set(range(10)):
        raise ValueError(f"Selection must bind all ten tasks of {suite}")
    counts = {len(task.indices) for task in tasks.values()}
    if len(counts) != 1:
        raise ValueError(f"Selection binds unequal state counts across {suite} tasks")
    return counts.pop()


def _write_bank_receipt(output, selection, suite):
    from tcr_merging.pi05.reset_bank import sha256_file

    result = output / "eval_info.json"
    if not result.is_file():
        raise FileNotFoundError(f"LeRobot evaluation did not produce {result}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank_manifest_sha256": selection.bank_manifest_sha256,
        "selection": selection.selection_path.name,
        "selection_sha256": selection.selection_sha256,
        "repeat_id": selection.repeat_id,
        "eval_seed": selection.eval_seed,
        "eval_info_sha256": sha256_file(result),
        "tasks": {
            task.task_key: {
                "task_name": task.task_name,
                "state_indices": list(task.indices),
                "raw_state_sha256": list(task.raw_sha256),
            }
            for task in selection.suite_tasks(suite).values()
        },
    }
    receipt = output / "reset_bank_receipt.json"
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return receipt


def bank_main(argv):
    from tcr_merging.pi05.reset_bank import SUITES, load_selection, repeat_protocol

    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on one suite with a procedural reset bank "
        "(paper protocol: repeat r uses selections/repeat-0r.json and seed 27400r)."
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--bank", required=True, type=Path, help="Reset-bank directory")
    parser.add_argument("--repeat", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    selection_path, seed = repeat_protocol(args.bank, args.repeat)
    selection = load_selection(args.bank, selection_path, verify_source_files=True)
    if selection.eval_seed != seed:
        raise ValueError(f"{selection_path} is not paper repeat {args.repeat}")
    episodes = selected_episodes(selection, args.suite)
    from lerobot.scripts import lerobot_eval
    from tcr_merging.pi05.reset_bank import install_reset_bank
    from tcr_merging.pi05.runtime import visual_feature_compatibility

    previous = sys.argv
    sys.argv = [
        "lerobot-eval",
        *bank_eval_arguments(args.policy, args.suite, args.output, episodes, seed, args.device),
    ]
    try:
        with visual_feature_compatibility(), install_reset_bank(selection):
            lerobot_eval.main()
    finally:
        sys.argv = previous
    receipt = _write_bank_receipt(args.output.resolve(), selection, args.suite)
    print(json.dumps({"reset_bank_receipt": str(receipt)}), flush=True)
