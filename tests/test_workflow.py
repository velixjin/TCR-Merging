import json

import pytest
from tcr_merging.calibration.traces import validate_trace
from tcr_merging.cli import main
from tcr_merging.config import load_config


def config_file(tmp_path, **updates):
    cfg = {
        "base_model": "base",
        "output": "output",
        "strict_paper_budget": False,
        "experts": {
            n: {"checkpoint": n, "cache_a": f"a/{n}", "cache_b": f"b/{n}"} for n in ("x", "y")
        },
        **updates,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_paths_relative_to_config_not_cwd(tmp_path, monkeypatch):
    p = config_file(tmp_path)
    monkeypatch.chdir("/")
    cfg = load_config(p)
    assert cfg["experts"]["x"]["cache_a"] == str(tmp_path / "a/x")


@pytest.mark.parametrize(
    "updates",
    [
        {"unknown": 1},
        {"ridge_ratio": -1},
        {"rows_per_call": 0},
        {"variant": "final_call"},
        {"variant": "expert_prefix"},
        {"first_pass_weighting": "other"},
    ],
)
def test_invalid_settings_fail_early(tmp_path, updates):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, **updates))


def test_dry_run_does_not_need_weights(tmp_path, capsys):
    main(["merge", "--config", str(config_file(tmp_path)), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert len(result["commands"]) == 2
    assert not (tmp_path / "output").exists()


def test_merge_rejects_demo_inputs_and_unknown_fields(tmp_path):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, variant="demo"))
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, ridge_reference="pass1"))
    assert load_config(config_file(tmp_path, variant="demo"), baseline=True)["variant"] == "demo"


def test_trace_rejects_wrong_teacher_and_broken_flow_groups(tmp_path):
    cfg = {"variant": "full", "strict_paper_budget": False}
    m = {
        "task": "x",
        "calibration_policy": str(tmp_path),
        "sample_count": 3,
        "method": "pi05_full_vision_language_action_block_regmeanpp_replay_calibration",
        "samples": [
            {
                "index": i,
                "vision_count": 3,
                "prompt_signature": 5,
                "request_index": 0,
                "flow_index": f,
                "selected_request_slot": 0,
            }
            for i, f in enumerate((0, 5, 9))
        ],
    }
    validate_trace(m, tmp_path, "x", 2, cfg)
    with pytest.raises(ValueError):
        validate_trace(m, tmp_path / "wrong", "x", 2, cfg)
    m["samples"][1]["flow_index"] = 9
    with pytest.raises(ValueError):
        validate_trace(m, tmp_path, "x", 2, cfg)


def test_three_pass_config_and_output_layout(tmp_path, capsys):
    from tcr_merging.merging.engine import pass_dir

    experts = {
        n: {"checkpoint": n, "cache_a": f"a/{n}", "cache_b": f"b/{n}", "cache_c": f"c/{n}"}
        for n in ("x", "y")
    }
    cfg = load_config(config_file(tmp_path, passes=3, experts=experts))
    assert [pass_dir(cfg, i).name for i in (1, 2, 3)] == ["pass1", "pass2", "merged"]
    two = load_config(config_file(tmp_path))
    assert [pass_dir(two, i).name for i in (1, 2)] == ["pass1", "merged"]
    with pytest.raises(ValueError):  # cache_c missing
        load_config(config_file(tmp_path, passes=3))
    with pytest.raises(ValueError):  # reused cache
        bad = {n: {**e, "cache_c": e["cache_b"]} for n, e in experts.items()}
        load_config(config_file(tmp_path, passes=3, experts=bad))
    main(["merge", "--config", str(config_file(tmp_path, passes=3, experts=experts)), "--dry-run"])
    assert len(json.loads(capsys.readouterr().out)["commands"]) == 3


def test_staging_directory_follows_umask(tmp_path):
    import os

    from tcr_merging.pi05.checkpoints import make_staging

    old = os.umask(0o022)
    try:
        assert (make_staging(tmp_path, ".s-").stat().st_mode & 0o777) == 0o755
    finally:
        os.umask(old)
