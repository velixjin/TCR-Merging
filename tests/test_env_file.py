"""Machine-specific paths come from a local .env file, never from tracked configs."""

import json
import os
from pathlib import Path

import pytest
from tcr_merging.config import load_config, load_env_file

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("TCR_CHECKPOINT_ROOT", "TCR_CACHE_ROOT", "TCR_OUTPUT_ROOT")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in (*KEYS, "TCR_ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_tracked_configs_have_no_machine_paths():
    for path in (ROOT / "configs").glob("*.json"):
        text = path.read_text()
        assert "../" not in text and "/mnt/" not in text and "/home/" not in text, path.name


def test_env_example_lists_every_config_variable():
    example = (ROOT / ".env.example").read_text()
    for path in (ROOT / "configs").glob("*.json"):
        for key in KEYS:
            if "${" + key + "}" in path.read_text():
                assert f"\n{key}=" in example, (path.name, key)


def test_configs_resolve_from_env_file(clean_env):
    (clean_env / ".env").write_text(
        "# local\nTCR_CHECKPOINT_ROOT=/data/ckpt\nexport TCR_CACHE_ROOT='/data/cache'\nTCR_OUTPUT_ROOT=\"/data/out\"\n"
    )
    cfg = load_config(ROOT / "configs" / "pi05.json")
    assert cfg["base_model"] == "/data/ckpt/base"
    assert cfg["experts"]["goal"]["cache_b"] == "/data/cache/b/goal"
    assert cfg["output"] == "/data/out/main"


def test_shell_environment_takes_precedence(clean_env, monkeypatch):
    (clean_env / "local.env").write_text("TCR_OUTPUT_ROOT=/from/file\n")
    monkeypatch.setenv("TCR_OUTPUT_ROOT", "/from/shell")
    monkeypatch.setenv("TCR_ENV_FILE", str(clean_env / "local.env"))
    assert load_env_file() == clean_env / "local.env"
    assert os.environ["TCR_OUTPUT_ROOT"] == "/from/shell"


def test_missing_variable_is_an_error(clean_env):
    with pytest.raises(KeyError, match="TCR_CHECKPOINT_ROOT"):
        load_config(ROOT / "configs" / "pi05.json")


def test_missing_named_env_file_is_an_error(clean_env, monkeypatch):
    monkeypatch.setenv("TCR_ENV_FILE", str(clean_env / "absent.env"))
    with pytest.raises(FileNotFoundError):
        load_env_file()


def test_relative_paths_still_resolve_against_config(clean_env):
    config = clean_env / "c.json"
    experts = {
        n: {"checkpoint": f"ck/{n}", "cache_a": f"a/{n}", "cache_b": f"b/{n}"} for n in ("x", "y")
    }
    config.write_text(
        json.dumps(
            {
                "base_model": "ck/base",
                "experts": experts,
                "output": "out",
                "strict_paper_budget": False,
            }
        )
    )
    assert load_config(config)["output"] == str(clean_env / "out")
