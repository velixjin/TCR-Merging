import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from tcr_merging.pi05 import reset_bank as rb
from tcr_merging.pi05.evaluation import bank_eval_arguments, selected_episodes

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_reset_bank", ROOT / "scripts/generate_reset_bank.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bank(root, state_dim=4):
    """Synthetic full-profile bank (40 tasks x 40 states) written with the real writers."""
    rng = np.random.default_rng(0)
    root.mkdir()
    tasks = {}
    for suite_index, suite in enumerate(rb.SUITES):
        for task_id in range(10):
            key = rb.task_key(suite, task_id)
            states = rng.standard_normal((rb.STATES_PER_TASK, state_dim))
            official = rng.standard_normal((3, state_dim))
            relative = f"states/{suite}/task-{task_id:02d}.npy"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / relative, states, allow_pickle=False)
            tasks[key] = {
                "suite": suite,
                "suite_index": suite_index,
                "task_id": task_id,
                "task_name": f"task_{key}",
                "problem_folder": suite,
                "bddl_file": f"{task_id}.bddl",
                "init_states_file": f"{task_id}.pruned_init",
                "official_init": {
                    "state_count": 3,
                    "raw_state_sha256": [rb.sha256_state(s) for s in official],
                },
                "states_file": relative,
                "states_file_sha256": rb.sha256_file(root / relative),
                "state_shape": [state_dim],
                "state_dtype": states.dtype.str,
                "states": [
                    {
                        "index": i,
                        "role": rb.role_for(i),
                        "seed": rb.seed_for(suite_index, task_id, i),
                        "raw_sha256": rb.sha256_state(s),
                    }
                    for i, s in enumerate(states)
                ],
            }
    manifest = {
        "schema_version": 1,
        "bank_id": rb.BANK_ID,
        "profile": "full",
        "canonical_benchmark_bank": False,
        "prospective_policy_independent_generation": True,
        "generator": {
            **rb.FROZEN_GENERATOR_FIELDS,
            "states_per_task": 40,
            "repeat_mapping": rb.repeat_mapping(40),
            "reserve_policy": {
                "indices": list(range(30, 40)),
                "selection": "smallest_unused_state_index",
            },
        },
        "tasks": tasks,
    }
    write_bank_files(root, manifest)
    return root


def write_bank_files(root, manifest):
    sha = rb.write_manifest(root, manifest)
    rb.write_selections(root, manifest, sha)
    return sha


def rewrite_manifest(root, edit):
    manifest = json.loads((root / "manifest.json").read_text())
    edit(manifest)
    rb.write_manifest(root, manifest)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    return make_bank(tmp_path_factory.mktemp("bank") / "bank")


@pytest.fixture
def bank_copy(tmp_path, bank):
    import shutil

    return Path(shutil.copytree(bank, tmp_path / "copy"))


def test_seed_formula_and_roles():
    assert rb.seed_for(0, 0, 0) == 275_000_000
    assert rb.seed_for(1, 2, 3) == 275_102_003
    assert rb.seed_for(3, 9, 39) == 275_309_039
    with pytest.raises(ValueError):
        rb.seed_for(0, 0, 40)
    with pytest.raises(ValueError):
        rb.seed_for(-1, 0, 0)
    assert [rb.role_for(i) for i in (0, 9, 10, 19, 20, 29, 30, 39)] == [
        "repeat-01", "repeat-01", "repeat-02", "repeat-02",
        "repeat-03", "repeat-03", "reserve", "reserve",
    ]  # fmt: skip


def test_repeat_protocol(tmp_path):
    for repeat in (1, 2, 3):
        path, seed = rb.repeat_protocol(tmp_path, repeat)
        assert path == tmp_path / "selections" / f"repeat-0{repeat}.json"
        assert seed == 274_000 + repeat
    with pytest.raises(ValueError):
        rb.repeat_protocol(tmp_path, 4)


@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_selection_maps_repeat_to_state_indices(bank, repeat):
    path, seed = rb.repeat_protocol(bank, repeat)
    selection = rb.load_selection(bank, path, verify_source_files=False)
    assert selection.repeat_id == f"repeat-0{repeat}" and selection.eval_seed == seed
    assert len(selection.tasks) == 40
    expected = tuple(range(10 * (repeat - 1), 10 * repeat))
    states = np.load(bank / "states/libero_goal/task-03.npy")
    for task in selection.tasks.values():
        assert task.indices == expected
    np.testing.assert_array_equal(
        selection.tasks[("libero_goal", 3)].states, states[list(expected)]
    )
    for suite in rb.SUITES:
        assert selected_episodes(selection, suite) == 10


def test_selection_files_and_receipts(bank):
    receipts = json.loads((bank / "selections/sha256.json").read_text())
    assert sorted(receipts) == ["repeat-01.json", "repeat-02.json", "repeat-03.json"]
    for name, digest in receipts.items():
        assert rb.sha256_file(bank / "selections" / name) == digest


def test_tampered_manifest_is_refused(bank_copy):
    path = bank_copy / "manifest.json"
    path.write_text(path.read_text().replace('"task_libero_goal/00"', '"task_libero_goal/0X"', 1))
    with pytest.raises(rb.ResetBankError, match="manifest SHA256 mismatch"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_tampered_state_file_is_refused(bank_copy):
    path = bank_copy / "states/libero_object/task-05.npy"
    states = np.load(path)
    states[12, 0] += 1e-12
    np.save(path, states)
    with pytest.raises(rb.ResetBankError, match="State-array SHA256 mismatch"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_tampered_state_row_is_refused_even_with_updated_file_hash(bank_copy):
    path = bank_copy / "states/libero_object/task-05.npy"
    states = np.load(path)
    states[12, 0] += 1e-12
    np.save(path, states)
    key = "libero_object/05"
    rewrite_manifest(
        bank_copy, lambda m: m["tasks"][key].update(states_file_sha256=rb.sha256_file(path))
    )
    with pytest.raises(
        rb.ResetBankError, match="Raw-state SHA256 mismatch for libero_object/05/12"
    ):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_wrong_seed_record_is_refused(bank_copy):
    rewrite_manifest(bank_copy, lambda m: m["tasks"]["libero_10/09"]["states"][7].update(seed=1))
    with pytest.raises(rb.ResetBankError, match="Seed formula mismatch for libero_10/09/7"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_official_state_collision_is_refused(bank_copy):
    def collide(manifest):
        row = manifest["tasks"]["libero_spatial/00"]
        row["official_init"]["raw_state_sha256"][0] = row["states"][4]["raw_sha256"]

    rewrite_manifest(bank_copy, collide)
    with pytest.raises(rb.ResetBankError, match="collides with the official bank"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_unsafe_states_path_is_refused(bank_copy):
    rewrite_manifest(
        bank_copy,
        lambda m: m["tasks"]["libero_spatial/00"].update(states_file="../outside.npy"),
    )
    with pytest.raises(rb.ResetBankError, match="Unsafe states_file"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_frozen_generator_fields_are_enforced(bank_copy):
    rewrite_manifest(bank_copy, lambda m: m["generator"].update(seed_base=0))
    with pytest.raises(rb.ResetBankError, match="seed_base"):
        rb.load_bank_manifest(bank_copy, verify_source_files=False)


def test_stale_selection_is_refused(bank_copy):
    path, _ = rb.repeat_protocol(bank_copy, 1)
    rewrite_manifest(bank_copy, lambda m: m.update(created_at="later"))
    with pytest.raises(rb.ResetBankError, match="does not bind the current bank manifest"):
        rb.load_selection(bank_copy, path, verify_source_files=False)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda s: s.update(eval_seed=274_003), "frozen mapping"),
        (lambda s: s.pop("eval_seed"), "both be set"),
        (lambda s: s["tasks"].update({"libero_10/00": [1, 1]}), "duplicate"),
        (lambda s: s["tasks"].update({"libero_10/00": [40]}), "out of range"),
        (lambda s: s["tasks"].update({"libero_99/00": [0]}), "unknown task"),
    ],
)
def test_invalid_selection_is_refused(bank_copy, edit, match):
    path, _ = rb.repeat_protocol(bank_copy, 2)
    selection = json.loads(path.read_text())
    edit(selection)
    path.write_text(json.dumps(selection))
    with pytest.raises(rb.ResetBankError, match=match):
        rb.load_selection(bank_copy, path, verify_source_files=False)


def test_partial_suite_selection_is_refused(bank_copy):
    path, _ = rb.repeat_protocol(bank_copy, 1)
    selection = json.loads(path.read_text())
    del selection["tasks"]["libero_goal/04"]
    path.write_text(json.dumps(selection))
    loaded = rb.load_selection(bank_copy, path, verify_source_files=False)
    assert selected_episodes(loaded, "libero_10") == 10
    with pytest.raises(ValueError, match="all ten tasks"):
        selected_episodes(loaded, "libero_goal")


def test_bank_eval_arguments_match_paper_protocol(tmp_path):
    args = bank_eval_arguments(
        tmp_path / "policy", "libero_goal", tmp_path / "out", 10, 274_002, "cuda"
    )
    for flag in (
        "--env.type=libero",
        "--env.task=libero_goal",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--eval.batch_size=1",
        "--eval.n_episodes=10",
        "--seed=274002",
        "--policy.n_action_steps=10",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--policy.device=cuda",
        f"--policy.path={(tmp_path / 'policy').resolve()}",
        f"--output_dir={(tmp_path / 'out').resolve()}",
    ):
        assert flag in args
    assert len(args) == 13


class FakeTask:
    def __init__(self, key):
        self.name, self.problem_folder, self.bddl_file = f"task_{key}", key.split("/")[0], ""


class FakeSuite:
    def __init__(self, suite):
        self.suite = suite

    def get_task(self, task_id):
        task = FakeTask(rb.task_key(self.suite, task_id))
        task.bddl_file = f"{task_id}.bddl"
        return task


class FakeLiberoEnv:
    """Mirrors the init-state handling of LeRobot's LiberoEnv (commit bf31dd79)."""

    def __init__(
        self, task_suite, task_id, task_suite_name, init_states=True, episode_index=0,
        n_envs=1, hard_reset=True,
    ):  # fmt: skip
        self.init_states = init_states
        self.episode_index = episode_index
        self._init_states = np.zeros((50, 4)) if init_states else None
        self._reset_stride = n_envs
        self.init_state_id = episode_index

    def reset(self, seed=None):
        if self.init_states and self._init_states is not None:
            state = self._init_states[self.init_state_id % len(self._init_states)]
            self.init_state_id += self._reset_stride
            return state
        return None


@pytest.fixture
def fake_lerobot(monkeypatch):
    libero = types.ModuleType("lerobot.envs.libero")
    libero.LiberoEnv = FakeLiberoEnv
    envs = types.ModuleType("lerobot.envs")
    envs.libero = libero
    lerobot = types.ModuleType("lerobot")
    lerobot.envs = envs
    for name, module in (
        ("lerobot", lerobot),
        ("lerobot.envs", envs),
        ("lerobot.envs.libero", libero),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return libero


def test_install_resets_to_selected_states_in_order(bank, fake_lerobot):
    selection = rb.load_selection(bank, rb.repeat_protocol(bank, 3)[0], verify_source_files=False)
    states = np.load(bank / "states/libero_object/task-02.npy")
    logs = []
    with rb.install_reset_bank(selection, log=logs.append):
        env = fake_lerobot.LiberoEnv(FakeSuite("libero_object"), 2, "libero_object")
        for index in range(20, 30):
            np.testing.assert_array_equal(env.reset(), states[index])
        with pytest.raises(rb.ResetBankError, match="exhausted"):
            env.reset()
        with pytest.raises(rb.ResetBankError, match="hard_reset"):
            fake_lerobot.LiberoEnv(FakeSuite("libero_object"), 2, "libero_object", hard_reset=False)
        with pytest.raises(rb.ResetBankError, match="metadata differs"):
            fake_lerobot.LiberoEnv(FakeSuite("libero_goal"), 2, "libero_object")
    assert json.loads(logs[0])["state_indices"] == list(range(20, 30))
    assert fake_lerobot.LiberoEnv.__init__ is FakeLiberoEnv.__init__
    assert fake_lerobot.LiberoEnv.reset is FakeLiberoEnv.reset


def test_generator_config_is_the_frozen_config():
    generator = load_generator()
    config = generator.load_config(ROOT / "configs/libero_reset_bank.json")
    assert config["bank_id"] == rb.BANK_ID
    assert config["state_generation"]["seed_formula"] == rb.SEED_FORMULA


def test_generator_refuses_altered_config(tmp_path):
    generator = load_generator()
    config = json.loads((ROOT / "configs/libero_reset_bank.json").read_text())
    config["state_generation"]["seed_base"] += 1
    (tmp_path / "c.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seed_base"):
        generator.load_config(tmp_path / "c.json")


def test_generator_profiles():
    generator = load_generator()
    generator.check_args(generator.parse_args(["--output", "x"]))
    for argv in (
        ["--states-per-task", "10"],
        ["--task", "libero_goal:1"],
        ["--determinism-check", "sampled"],
        ["--profile", "preflight"],
        ["--workers", "9"],
    ):
        with pytest.raises(ValueError):
            generator.check_args(generator.parse_args(["--output", "x", *argv]))


class FakeControlEnv:
    """Draws its 'scene' from the global NumPy RNG on reset, like LIBERO's samplers."""

    def __init__(self, bddl_path):
        self.state = None

    def seed(self, seed):
        self.seeded = seed

    def reset(self):
        self.state = np.random.random(4)

    def get_sim_state(self):
        return self.state.copy()

    def set_init_state(self, state):
        self.state = np.asarray(state, dtype=float).copy()

    def check_success(self):
        return False

    def step(self, action):
        pass

    def close(self):
        pass


def test_generator_state_uses_frozen_seed():
    generator = load_generator()
    job = generator.TaskJob(
        suite="libero_goal", suite_index=2, task_id=7, task_name="t", language="t",
        problem_folder="libero_goal", bddl_file="t.bddl", init_states_file="t.pruned_init",
        bddl_path="t.bddl", official_receipt={"raw_state_sha256": []}, expected_shape=(4,),
        states_per_task=40, determinism_indices=(13,), restore_indices=(13,),
    )  # fmt: skip
    state, record = generator._generate_one_state(job, 13, set(), env_factory=FakeControlEnv)
    seed = 275_000_000 + 2 * 100_000 + 7 * 1_000 + 13
    assert record["seed"] == seed and record["role"] == "repeat-02"
    np.testing.assert_array_equal(state, np.random.RandomState(seed).random_sample(4))
    assert record["raw_sha256"] == rb.sha256_state(state)
    assert record["same_seed_regeneration_checked"] and record["save_restore_noop_checked"]
    with pytest.raises(RuntimeError, match="duplicates an earlier state"):
        generator._generate_one_state(job, 13, {record["raw_sha256"]}, env_factory=FakeControlEnv)


@pytest.mark.skipif(not os.environ.get("TCR_RESET_BANK"), reason="set TCR_RESET_BANK=<bank dir>")
def test_released_bank(tmp_path):
    """Optional check of a real bank, e.g. the paper's libero-procedural-clean-v1."""
    bank = Path(os.environ["TCR_RESET_BANK"])
    manifest = rb.load_bank_manifest(bank, verify_source_files=False)
    assert manifest["profile"] == "full"
    for repeat in (1, 2, 3):
        path, seed = rb.repeat_protocol(bank, repeat)
        selection = rb.load_selection(bank, path, verify_source_files=False)
        assert selection.eval_seed == seed
        assert {t.indices for t in selection.tasks.values()} == {
            tuple(range(10 * (repeat - 1), 10 * repeat))
        }
    # The selection writer reproduces the published selection files byte for byte.
    rb.write_selections(tmp_path, manifest, rb.sha256_file(bank / "manifest.json"))
    for name in ("repeat-01.json", "repeat-02.json", "repeat-03.json", "sha256.json"):
        assert (tmp_path / "selections" / name).read_bytes() == (
            bank / "selections" / name
        ).read_bytes()
