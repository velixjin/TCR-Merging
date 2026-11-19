"""Procedural LIBERO reset banks: loading, validation and installation into LeRobot.

The paper evaluates every checkpoint on a custom, policy-independent reset bank
instead of LIBERO's stock ``.pruned_init`` files, which are left untouched.  A bank
directory holds ``manifest.json`` (+ ``manifest.sha256``), one ``states/<suite>/
task-XX.npy`` array per task and ``selections/repeat-0{1,2,3}.json``.  It is accepted
only when the manifest, every state array and state row, and (optionally) the current
BDDL and official init-state files all match their recorded SHA256 identities.

State ``i`` of task ``t`` in suite ``s`` was generated with seed
``275000000 + s*100000 + t*1000 + i``; indices 0-9, 10-19 and 20-29 form repeats 1-3
(evaluated with seeds 274001-274003) and 30-39 are reserves.
"""

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BANK_ID = "procedural-clean-v1"
BANK_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEED_BASE = 275_000_000
SEED_SUITE_STRIDE = 100_000
SEED_TASK_STRIDE = 1_000
SEED_FORMULA = "275000000 + suite_index*100000 + task_id*1000 + state_index"
STATES_PER_TASK = 40
TASKS_PER_SUITE = 10
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
EVAL_SEEDS = (274_001, 274_002, 274_003)
NOOP_STEPS = 10
NOOP_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


class ResetBankError(ValueError):
    """Raised when a reset bank or selection fails validation."""


def _seed(suite_index, task_id, state_index):
    return SEED_BASE + suite_index * SEED_SUITE_STRIDE + task_id * SEED_TASK_STRIDE + state_index


def seed_for(suite_index, task_id, state_index):
    """Simulator seed used to generate one bank state."""
    if suite_index < 0 or task_id < 0 or state_index < 0:
        raise ValueError("Seed coordinates must be nonnegative")
    if state_index >= STATES_PER_TASK:
        raise ValueError("state_index exceeds the seed-formula slot")
    return _seed(suite_index, task_id, state_index)


def role_for(index):
    if 0 <= index < 10:
        return "repeat-01"
    if index < 20:
        return "repeat-02"
    if index < 30:
        return "repeat-03"
    return "reserve"


def repeat_mapping(state_count):
    return {
        "repeat-01": list(range(0, min(state_count, 10))),
        "repeat-02": list(range(10, min(state_count, 20))),
        "repeat-03": list(range(20, min(state_count, 30))),
        "reserve": list(range(30, state_count)),
    }


def repeat_protocol(bank_root, repeat):
    """Selection file and evaluation seed for paper repeat 1, 2 or 3."""
    if repeat not in (1, 2, 3):
        raise ValueError("repeat must be 1, 2 or 3")
    return Path(bank_root) / "selections" / f"{REPEATS[repeat - 1]}.json", EVAL_SEEDS[repeat - 1]


def task_key(suite, task_id):
    return f"{suite}/{task_id:02d}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state(state):
    value = np.asarray(state)
    if value.ndim != 1:
        raise ResetBankError(f"Expected a flat simulator state, got shape={value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ResetBankError(f"Expected a floating simulator state, got dtype={value.dtype}")
    if not bool(np.isfinite(value).all()):
        raise ResetBankError("Simulator state contains non-finite values")
    # The dtype and shape are part of the hash domain, so the simulator dtype is kept.
    return np.ascontiguousarray(value)


def sha256_state(state):
    value = canonical_state(state)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResetBankError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResetBankError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_relative_path(root, raw, field):
    if not isinstance(raw, str) or not raw:
        raise ResetBankError(f"{field} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ResetBankError(f"Unsafe {field}: {raw!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResetBankError(f"{field} escapes bank root: {raw!r}") from exc
    return resolved


def _verify_manifest_sidecar(root, manifest_path):
    receipt = root / "manifest.sha256"
    if not receipt.is_file():
        raise ResetBankError(f"Missing manifest hash receipt: {receipt}")
    fields = receipt.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != "manifest.json":
        raise ResetBankError(f"Malformed manifest hash receipt: {receipt}")
    observed = sha256_file(manifest_path)
    if fields[0] != observed:
        raise ResetBankError(
            f"Bank manifest SHA256 mismatch: receipt={fields[0]} observed={observed}"
        )


FROZEN_GENERATOR_FIELDS = {
    "seed_formula": SEED_FORMULA,
    "seed_base": SEED_BASE,
    "suite_stride": SEED_SUITE_STRIDE,
    "task_stride": SEED_TASK_STRIDE,
    "suite_order": list(SUITES),
    "final_eval_seeds": list(EVAL_SEEDS),
}


def _check_header(manifest):
    if manifest.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ResetBankError("Unsupported reset-bank schema")
    if manifest.get("bank_id") != BANK_ID:
        raise ResetBankError(f"Unexpected bank_id; {BANK_ID} is required")
    if manifest.get("canonical_benchmark_bank") is not False:
        raise ResetBankError("Reset bank must explicitly declare itself non-canonical")
    if manifest.get("prospective_policy_independent_generation") is not True:
        raise ResetBankError("Reset bank lacks the policy-independent generation declaration")
    generator = manifest.get("generator")
    if not isinstance(generator, dict):
        raise ResetBankError("Reset bank lacks a generator record")
    for field, expected in FROZEN_GENERATOR_FIELDS.items():
        if generator.get(field) != expected:
            raise ResetBankError(f"Generator field differs for {field}: {generator.get(field)!r}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ResetBankError("Reset bank contains no task records")
    if manifest.get("profile") == "full":
        mapping = generator.get("repeat_mapping")
        if generator.get("states_per_task") != STATES_PER_TASK or mapping != repeat_mapping(40):
            raise ResetBankError("Full bank does not have the 30+10 state mapping")
        reserve = generator.get("reserve_policy", {})
        if (
            reserve.get("indices") != list(range(30, 40))
            or reserve.get("selection") != "smallest_unused_state_index"
        ):
            raise ResetBankError("Full bank reserve policy differs from the frozen rule")
        expected_tasks = {task_key(s, t) for s in SUITES for t in range(TASKS_PER_SUITE)}
        if set(tasks) != expected_tasks:
            raise ResetBankError("Full bank does not contain exactly four suites by ten tasks")


def _check_task(root, key, row, seen_state_hashes):
    if not isinstance(key, str) or not isinstance(row, dict):
        raise ResetBankError("Malformed task record")
    if row.get("suite") not in SUITES:
        raise ResetBankError(f"Unknown suite in task record {key}")
    if key != task_key(row.get("suite"), int(row.get("task_id"))) or row.get(
        "suite_index"
    ) != SUITES.index(row.get("suite")):
        raise ResetBankError(f"Task identity/index mismatch for {key}")
    states_path = _safe_relative_path(root, row.get("states_file"), "states_file")
    if not states_path.is_file():
        raise ResetBankError(f"Missing state array for {key}: {states_path}")
    if sha256_file(states_path) != row.get("states_file_sha256"):
        raise ResetBankError(f"State-array SHA256 mismatch for {key}")
    try:
        states = np.load(states_path, allow_pickle=False)
    except Exception as exc:
        raise ResetBankError(f"Cannot load state array for {key}: {exc}") from exc
    if states.ndim != 2 or list(states.shape[1:]) != row.get("state_shape"):
        raise ResetBankError(f"State-array shape mismatch for {key}: {states.shape}")
    if states.dtype.str != row.get("state_dtype"):
        raise ResetBankError(f"State-array dtype mismatch for {key}: {states.dtype.str}")
    records = row.get("states")
    if not isinstance(records, list) or len(records) != states.shape[0]:
        raise ResetBankError(f"State-record count mismatch for {key}")
    if [r.get("index") for r in records if isinstance(r, dict)] != list(range(states.shape[0])):
        raise ResetBankError(f"State indices are not contiguous in {key}")
    official = row.get("official_init", {})
    official_hashes = set(official.get("raw_state_sha256", []))
    if len(official_hashes) != official.get("state_count"):
        raise ResetBankError(f"Official-state hash record is incomplete for {key}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ResetBankError(f"Malformed state record for {key}/{index}")
        if record.get("seed") != _seed(int(row["suite_index"]), int(row["task_id"]), index):
            raise ResetBankError(f"Seed formula mismatch for {key}/{index}")
        if record.get("role") != role_for(index):
            raise ResetBankError(f"State role mismatch for {key}/{index}")
        state_hash = sha256_state(states[index])
        if state_hash != record.get("raw_sha256"):
            raise ResetBankError(f"Raw-state SHA256 mismatch for {key}/{index}")
        if state_hash in official_hashes:
            raise ResetBankError(f"Bank state collides with the official bank: {key}/{index}")
        if state_hash in seen_state_hashes:
            raise ResetBankError(f"Duplicate bank state hash: {key}/{index}")
        seen_state_hashes.add(state_hash)


def _check_source_files(key, row):
    from libero.libero import get_libero_path

    folder = str(row.get("problem_folder"))
    current = {
        "bddl": Path(get_libero_path("bddl_files")) / folder / str(row.get("bddl_file")),
        "official_init": Path(get_libero_path("init_states"))
        / folder
        / str(row.get("init_states_file")),
    }
    for name, path in current.items():
        record = row.get(name)
        if not isinstance(record, dict):
            raise ResetBankError(f"Missing {name} record for {key}")
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ResetBankError(f"Current LIBERO {name} file differs from the bank for {key}")


def load_bank_manifest(bank_root, verify_source_files=True):
    """Validate a bank directory and return its manifest.

    ``verify_source_files`` additionally checks that the installed LIBERO BDDL and
    official init-state files are the ones the bank was generated against.
    """
    root = Path(bank_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ResetBankError(f"Missing bank manifest: {manifest_path}")
    _verify_manifest_sidecar(root, manifest_path)
    manifest = _read_json(manifest_path)
    _check_header(manifest)
    seen_state_hashes = set()
    for key, row in manifest["tasks"].items():
        _check_task(root, key, row, seen_state_hashes)
        if verify_source_files:
            _check_source_files(key, row)
    return manifest


@dataclass(frozen=True)
class SelectedTask:
    task_key: str
    suite: str
    task_id: int
    task_name: str
    problem_folder: str
    bddl_file: str
    indices: tuple
    states: np.ndarray
    raw_sha256: tuple


@dataclass(frozen=True)
class Selection:
    bank_root: Path
    bank_manifest_sha256: str
    selection_path: Path
    selection_sha256: str
    repeat_id: str | None
    eval_seed: int | None
    tasks: dict  # (suite, task_id) -> SelectedTask

    def suite_tasks(self, suite):
        return {t: task for (s, t), task in self.tasks.items() if s == suite}


def load_selection(bank_root, selection_path, verify_source_files=True):
    """Validate a bank plus one selection file and return the selected state rows."""
    root = Path(bank_root).expanduser().resolve()
    manifest = load_bank_manifest(root, verify_source_files=verify_source_files)
    selection_file = Path(selection_path).expanduser().resolve()
    selection = _read_json(selection_file)
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ResetBankError("Unsupported reset-bank selection schema")
    bank_sha = sha256_file(root / "manifest.json")
    if selection.get("bank_id") != manifest["bank_id"]:
        raise ResetBankError("Selection bank_id differs from bank manifest")
    if selection.get("bank_manifest_sha256") != bank_sha:
        raise ResetBankError("Selection does not bind the current bank manifest SHA256")
    bindings = selection.get("tasks")
    if not isinstance(bindings, dict) or not bindings:
        raise ResetBankError("Selection contains no task bindings")

    repeat_id = selection.get("repeat_id")
    eval_seed = selection.get("eval_seed")
    if (repeat_id is None) != (eval_seed is None):
        raise ResetBankError("repeat_id and eval_seed must both be set or both be omitted")
    if repeat_id is not None:
        if repeat_id not in REPEATS:
            raise ResetBankError(f"Unsupported repeat_id: {repeat_id!r}")
        if type(eval_seed) is not int:
            raise ResetBankError("Selection eval_seed must be an integer")
        expected_seeds = dict(
            zip(REPEATS, manifest.get("generator", {}).get("final_eval_seeds", []), strict=True)
        )
        if eval_seed != expected_seeds.get(repeat_id):
            raise ResetBankError(f"Eval seed differs from the frozen mapping for {repeat_id}")

    selected = {}
    for key, raw_indices in bindings.items():
        if key not in manifest["tasks"]:
            raise ResetBankError(f"Selection references unknown task: {key}")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ResetBankError(f"Selection for {key} must be a non-empty index list")
        if any(type(index) is not int or index < 0 for index in raw_indices):
            raise ResetBankError(f"Selection for {key} contains an invalid index")
        if len(set(raw_indices)) != len(raw_indices):
            raise ResetBankError(f"Selection for {key} contains duplicate indices")
        row = manifest["tasks"][key]
        states_path = _safe_relative_path(root, row["states_file"], "states_file")
        states = np.load(states_path, allow_pickle=False)
        if any(index >= states.shape[0] for index in raw_indices):
            raise ResetBankError(f"Selection index is out of range for {key}")
        chosen = np.ascontiguousarray(states[raw_indices])
        hashes = tuple(row["states"][index]["raw_sha256"] for index in raw_indices)
        # Recheck the selected rows after advanced indexing/copying.
        for state, expected in zip(chosen, hashes, strict=True):
            if sha256_state(state) != expected:
                raise ResetBankError(f"Selected raw-state SHA256 mismatch for {key}")
        identity = (str(row["suite"]), int(row["task_id"]))
        if identity in selected:
            raise ResetBankError(f"Duplicate task identity in selection: {identity}")
        selected[identity] = SelectedTask(
            task_key=key,
            suite=identity[0],
            task_id=identity[1],
            task_name=str(row["task_name"]),
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(row["bddl_file"]),
            indices=tuple(raw_indices),
            states=chosen,
            raw_sha256=hashes,
        )
    return Selection(
        bank_root=root,
        bank_manifest_sha256=bank_sha,
        selection_path=selection_file,
        selection_sha256=sha256_file(selection_file),
        repeat_id=repeat_id,
        eval_seed=eval_seed,
        tasks=selected,
    )


def write_manifest(root, manifest):
    """Write ``manifest.json`` and its ``manifest.sha256`` receipt; return the digest."""
    path = Path(root) / "manifest.json"
    _write_json(path, manifest)
    digest = sha256_file(path)
    (Path(root) / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
    return digest


def write_selections(root, manifest, manifest_sha):
    """Write the three repeat selections (full bank) or one all-states selection."""
    selection_root = Path(root) / "selections"
    selection_root.mkdir(parents=True, exist_ok=False)
    base = {
        "schema_version": 1,
        "bank_id": manifest["bank_id"],
        "bank_manifest_sha256": manifest_sha,
    }
    if manifest["profile"] == "full":
        mapping = manifest["generator"]["repeat_mapping"]
        for repeat_id, seed in zip(REPEATS, EVAL_SEEDS, strict=True):
            payload = {
                **base,
                "repeat_id": repeat_id,
                "eval_seed": seed,
                "tasks": {key: mapping[repeat_id] for key in sorted(manifest["tasks"])},
            }
            _write_json(selection_root / f"{repeat_id}.json", payload)
    else:
        tasks = {
            key: list(range(len(row["states"]))) for key, row in sorted(manifest["tasks"].items())
        }
        _write_json(selection_root / "preflight-all.json", {**base, "tasks": tasks})
    receipts = {p.name: sha256_file(p) for p in sorted(selection_root.glob("*.json"))}
    _write_json(selection_root / "sha256.json", receipts)


@contextmanager
def install_reset_bank(selection, log=print):
    """Make LeRobot's ``LiberoEnv`` reset to the selected bank states, in order.

    Episode ``k`` of task ``t`` starts from ``selection.tasks[(suite, t)].states[k]``.
    Environments must use ``hard_reset=True``; running past the selected states raises
    instead of wrapping around.
    """
    from lerobot.envs import libero

    original_init = libero.LiberoEnv.__init__
    original_reset = libero.LiberoEnv.reset
    signature = inspect.signature(original_init)

    def init(env, *args, **kwargs):
        bound = signature.bind(env, *args, **kwargs)
        bound.apply_defaults()
        suite = str(bound.arguments["task_suite_name"])
        task_id = int(bound.arguments["task_id"])
        selected = selection.tasks.get((suite, task_id))
        if selected is None:
            raise ResetBankError(f"No reset-bank selection for LIBERO task {suite}/{task_id}")
        if not bool(bound.arguments["hard_reset"]):
            raise ResetBankError("Reset-bank evaluation requires hard_reset=True")
        task = bound.arguments["task_suite"].get_task(task_id)
        if (task.name, task.problem_folder, task.bddl_file) != (
            selected.task_name,
            selected.problem_folder,
            selected.bddl_file,
        ):
            raise ResetBankError(f"Current task metadata differs from bank for {suite}/{task_id}")
        # Keep the stock loader from reading the official .pruned_init file, then
        # install the selected rows immediately after normal construction.
        bound.arguments["init_states"] = False
        original_init(*bound.args, **bound.kwargs)
        env.init_states = True
        env._init_states = selected.states.copy()
        env.init_state_id = env.episode_index
        env._reset_bank_task = selected
        if log is not None:
            log(
                json.dumps(
                    {
                        "reset_bank": selection.bank_manifest_sha256,
                        "selection": selection.selection_sha256,
                        "task": selected.task_key,
                        "state_indices": list(selected.indices),
                        "raw_state_sha256": list(selected.raw_sha256),
                    },
                    sort_keys=True,
                )
            )

    def reset(env, *args, **kwargs):
        selected = getattr(env, "_reset_bank_task", None)
        if selected is not None and env.init_state_id >= len(env._init_states):
            raise ResetBankError(f"Selection exhausted for {selected.task_key}; refusing reuse")
        return original_reset(env, *args, **kwargs)

    libero.LiberoEnv.__init__ = init
    libero.LiberoEnv.reset = reset
    try:
        yield
    finally:
        libero.LiberoEnv.__init__ = original_init
        libero.LiberoEnv.reset = original_reset
