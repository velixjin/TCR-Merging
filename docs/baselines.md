# Baselines and demonstration inputs

`tcr baseline` builds the two baselines shipped with this repository from the same expert
checkpoints as TCR.

| Method | Calibration input | Config |
| --- | --- | --- |
| Model Soups | none (uniform weight average) | `configs/pi05.json` |
| RegMean++ | demonstration-observation cache A, first generation call, 16 rows per request | `configs/regmeanpp.json` |

```bash
tcr baseline --config configs/pi05.json --method soup --output outputs/soup
tcr baseline --config configs/regmeanpp.json --method regmeanpp --output outputs/regmeanpp
```

RegMean++ runs one independent pass with each expert's own preceding blocks as the feature
source, α = 0.3 off-diagonal Gram scaling, and spectral smoothing of the rank-deficient Gram
matrices (`--regmeanpp-solver original` gives the unsmoothed exact solve). `--dry-run` prints
the resolved settings without loading weights.

## Demonstration observations

RegMean++ is calibrated on demonstration observations rather than on expert executions.
Prepare a JSON list of prespecified demonstration episode indices, one per task, from your
training/calibration split (not the evaluation split). Provide the local LeRobot dataset and the
matching execution cache as a noise template:

```bash
tcr collect-demo --name spatial --policy checkpoints/spatial \
  --dataset data/libero --episodes data/spatial_demo_episodes_a.json \
  --template caches/a/spatial --output caches/demo_a/spatial
```

Repeat for every expert and place the caches under `${TCR_CACHE_ROOT}/demo_a` (set in `.env`).
The collector uses five observation positions per demonstration, runs native expert generation,
and pairs initial generation noise with the template's five requests. Demonstration action
labels are never passed to the model. Frame/episode identities and template hashes are recorded.
Demonstration caches carry `variant: "demo"` in their config; `tcr merge` accepts only
frozen-expert execution caches (`variant: "full"`).
