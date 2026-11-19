# TCR: Trajectory-Calibrated Merging of Embodied Policies

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg)](https://pytorch.org/)

**[Project Page](https://velixjin.github.io/TCR-Merging/)**

Wenshuo Chen<sup>1\*</sup>, Honglei Jin<sup>1\*†</sup>, Le Yang<sup>2\*</sup>, Jianfei Song<sup>3</sup>, Shaofeng Liang<sup>1</sup>, Lei Wang<sup>4</sup>, Junze Zhu<sup>1</sup>, Kailong Wang<sup>1</sup>, Zhengxi Jin<sup>5</sup>, Haozhe Jia<sup>5</sup>, Honglin Chu<sup>3</sup>, Songning Lai<sup>1</sup>, Xiucheng Wang<sup>6</sup>, Yutao Yue<sup>1,7✉</sup>

<sup>1</sup>HKUST(GZ) &nbsp; <sup>2</sup>Sun Yat-sen University &nbsp; <sup>3</sup>LimX Dynamics &nbsp; <sup>4</sup>Griffith University &nbsp; <sup>5</sup>Shandong University &nbsp; <sup>6</sup>Xidian University &nbsp; <sup>7</sup>JITRI

<sub>\* Equal contribution &nbsp; † Project leader &nbsp; ✉ Corresponding author</sub>

## News

- **2026-11**: Initial code release (TCR, Model Soups and RegMean++ baselines, LIBERO reset-bank evaluation).

## Overview

Official implementation of **Trajectory-Calibrated Regression (TCR)** for merging robot policy
experts into one policy with fixed weights, the original inference graph and no expert router.

TCR records the inputs that every linear module sees while each frozen expert *executes its own
task*, including intermediate calls of the iterative action generator. It then fits merged
weights module by module with a ridge regression centered at the expert mean, while replaying
the partially merged network in forward order so that every module is calibrated on features
produced by the already-merged upstream modules. Sequential passes on separate execution caches
give the final policy; the default recipe uses three passes, and a two-pass configuration is
also provided.

This repository supports the paper's **π0.5 / LeRobot** setting (418 linear modules, 422 adapted
tensors) and the LIBERO benchmark, and includes the two reference baselines Model Soups and
RegMean++.

Model weights and calibration caches are not released yet; see
[Reproducing the paper](docs/reproduction.md) for what is needed and for the
equivalence checks of this code against the code that produced the paper checkpoints.

## Installation

Python 3.12, PyTorch 2.7.1 (CUDA 12.8) and LeRobot at commit
`bf31dd794ffb4f87380aba3912f64421e8352d3c`. The block replay uses version-specific
LeRobot/Transformers internals, so use the pinned versions. Full instructions, including LIBERO
and the PaliGemma tokenizer: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -c requirements/constraints.txt -e '.[pi05,libero,test]'
python scripts/check_env.py --cuda --libero
pytest -q
```

## Usage

### 1. Prepare experts

A shared dense base and at least two dense experts fine-tuned from it. Each directory needs
`model.safetensors`, `config.json`, a local `tokenizer/`, and the matching LeRobot
pre/postprocessor files. Standard PEFT LoRA experts are materialized with:

```bash
tcr export-adapter --base-model checkpoints/base \
  --adapter checkpoints/spatial_adapter --output checkpoints/spatial
```

The expert training recipe used in the paper is described in [docs/reproduction.md](docs/reproduction.md).

### 2. Collect three execution caches per expert

Cache A keeps five reservoir-selected requests per task; caches B and C each keep five requests
spread across a separate episode. All retain generation calls 0, 5 and 9 of the ten
flow-matching steps.

```bash
tcr collect --name spatial --policy checkpoints/spatial --suite libero_spatial \
  --selection reservoir --init-state-offset 0 --seed 271001 \
  --noise-seed 272001 --output caches/a/spatial
tcr collect --name spatial --policy checkpoints/spatial --suite libero_spatial \
  --selection quantiles --init-state-offset 1 --seed 271101 \
  --noise-seed 272101 --output caches/b/spatial
tcr collect --name spatial --policy checkpoints/spatial --suite libero_spatial \
  --selection quantiles --init-state-offset 2 --seed 271201 \
  --noise-seed 272201 --output caches/c/spatial
```

Repeat for every expert. Keep calibration episodes disjoint from evaluation episodes. For other
simulators or real robots, use `ReplayCollector` in your own rollout loop (see its docstring in
[src/calibration/collection.py](src/calibration/collection.py)).

### 3. Merge

Machine-specific locations go in a local `.env` file, which is git-ignored and never committed. The configs refer
to them as `${TCR_CHECKPOINT_ROOT}`, `${TCR_CACHE_ROOT}` and `${TCR_OUTPUT_ROOT}`:

```bash
cp .env.example .env   # then set absolute paths; shell variables override the file
```

Relative paths in a config still resolve against the config file. Then:

```bash
tcr check --config configs/pi05_3pass.json
tcr merge --config configs/pi05_3pass.json --dry-run
tcr merge --config configs/pi05_3pass.json
```

Pass 1 (cache A) starts from the expert mean with relative-error expert masses (10 rows per
generation call and view); pass 2 (cache B) starts from pass 1 with uniform masses, 16 rows per
request and module, and inherits pass 1's per-module ridge values; pass 3 (cache C) repeats pass 2
starting from pass 2. The merged policy is written to `outputs/main_3pass/merged/` (intermediate
passes in `pass1/` and `pass2/`), each with a `tcr_manifest.json` recording input hashes,
per-module row counts, numerical settings and the output hash. For a two-pass policy, use
`configs/pi05.json` (caches A and B; written to `outputs/main/merged/`).
Existing outputs are never overwritten. One 80 GB GPU is enough for four experts (peak ≈ 34 GB
per pass).

### 4. Evaluate

The paper evaluates on a procedural LIBERO reset bank (40 tasks × 40 states, disjoint from the
calibration episodes): three repeats, each running ten bank states per task (repeat *r* uses
states 10(r−1)…10r−1 and seed 27400*r*).

```bash
python scripts/generate_reset_bank.py --config configs/libero_reset_bank.json --output banks/clean-v1
tcr evaluate-bank --policy outputs/main_3pass/merged --suite libero_spatial \
  --bank banks/clean-v1 --repeat 1 --output outputs/eval/spatial-r1
```

The bank records hashes of the LIBERO BDDL and init files it was built from; `evaluate-bank`
refuses to run against a different LIBERO installation. A quick check on stock LIBERO init states:

```bash
tcr evaluate --policy outputs/main_3pass/merged --suite libero_spatial \
  --episodes 10 --seed 274001 --init-state-offset 20 --output outputs/eval/spatial
```

### Baselines

```bash
tcr baseline --config configs/pi05.json --method soup --output outputs/soup
tcr baseline --config configs/regmeanpp.json --method regmeanpp --output outputs/regmeanpp
```

RegMean++ uses α = 0.3 off-diagonal Gram scaling and spectral smoothing of the rank-deficient
Gram matrices (`--regmeanpp-solver original` gives the unsmoothed exact solve). As in the paper,
it is calibrated on demonstration observations (`tcr collect-demo`) with 16 rows per sample; see
[docs/baselines.md](docs/baselines.md).

## Repository Structure

```text
src/                 # installed as the Python package `tcr_merging`
  merging/           # prior-centered ridge regression, row sampling, sequential multi-pass engine
  pi05/              # dense checkpoints, LoRA export, native block replay, LIBERO entry points
  calibration/       # execution / demonstration capture and trace validation
  baselines/         # Model Soups and RegMean++ (spectral smoothing / original solver)
  cli.py             # `tcr` command
  config.py          # portable JSON configuration
configs/             # TCR recipe, RegMean++ baseline, LIBERO reset bank
scripts/             # environment checker, reset-bank generator
tests/               # numerical and workflow tests (CPU + small GPU)
```

## Citation

```bibtex
@misc{chen2026trajectorycalibratedmerging,
  title         = {Trajectory-Calibrated Merging of Embodied Policies},
  author        = {Wenshuo Chen and Honglei Jin and Le Yang and Jianfei Song and Shaofeng Liang and Lei Wang and Junze Zhu and Kailong Wang and Zhengxi Jin and Haozhe Jia and Honglin Chu and Songning Lai and Xiucheng Wang and Yutao Yue},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). LeRobot, Transformers, PEFT, LIBERO,
model weights and datasets keep their own licenses.

## Acknowledgements

The native block replay in
`src/pi05/replay.py` re-implements the forward computation of LeRobot's π0.5 / PaliGemma layers
(Apache-2.0), which is itself a port of Physical Intelligence's OpenPI. We thank the authors of
[LeRobot](https://github.com/huggingface/lerobot), [OpenPI](https://github.com/Physical-Intelligence/openpi)
and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for releasing their code.
