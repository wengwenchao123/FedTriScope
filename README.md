# FedTriScope

## Installation

Use Python 3.9 or newer. Install a PyTorch build appropriate for your GPU first,
then run `pip install -r requirements.txt`. CPU is supported via `--device cpu`.
The historical experiments used a single NVIDIA RTX 5060 Ti; timings and exact
floating-point results can vary across hardware and dependency versions.

## Data

The datasets can be obtained through the download link in the Data section of
the [ST-GFSL repository](https://github.com/RobinLu1209/ST-GFSL#data).
Please follow its data usage and citation instructions. Datasets and checkpoints
are not bundled with this repository. Prepare the processed `dataset.npy` files
under `data/<city>/`; PEMS-BAY additionally requires `pems-bay.h5` with its original
timestamps for calendar alignment.
`--data_path` specifies the parent directory containing all four city folders.
It defaults to this repository's `data/` directory. Explicit relative paths are
resolved against the current working directory.

## Train each target city

Run from this directory:

```bash
python run.py --config configs/metr-la.json
python run.py --config configs/pems-bay.json
python run.py --config configs/shenzhen.json
python run.py --config configs/chengdu.json
```

Each task uses the other three cities as sources. To use an existing data folder:

```bash
python run.py --config configs/metr-la.json --data_path /path/to/data --device cuda:0
```

For PyCharm, select `run.py` and put `--config configs/metr-la.json` in Parameters;
set the working directory to the repository root. Change the city config as needed.
An explicit `--log_dir` must be empty to prevent accidental result overwrite.

## Evaluate

```bash
python run.py --mode test --checkpoint runs/metr-la/RUN_ID/final.pth --data_path /path/to/data
```

Architecture and target-city settings are restored from the checkpoint. Only
load trusted checkpoints. This is evaluation, not optimizer-state resume.

## Released protocol

| Setting | Value |
|---|---|
| History / forecast | 12 / 6 steps |
| Common feature/identity dimension | 8 for LA/BAY targets; 64 for Shenzhen/Chengdu targets |
| Input | Standardized speed relative to the final historical observation |
| Backbone | Three residual point-wise MLP blocks, dropout 0.15 |
| Shared target/source step sizes | 0.0002 / 0.000001 |
| Private optimizer | Adam, 0.001 |
| Target adaptation | 50 full epochs; fresh shared Adam at 0.001, retained private Adam state |
| Target training | First 864 observations (3 x 288) |
| Target test | Final 20% of the complete series |
| Source training | Complete source-city series |

864 observations correspond to **3 days for LA/BAY and 6 days for Shenzhen/Chengdu**.
Target observations between the training prefix and the test split are not used
for gradient training. Training always runs for the configured number of rounds
and adaptation epochs.

All four cities use the same dimension within a task. Sources are averaged equally
before per-parameter-tensor conflict projection against the target gradient.
Private embeddings are updated locally with Adam; shared weights use a direct
weighted gradient step, followed by complete replacement at every city.
Target adaptation updates both shared and private target parameters.

After training, the final model is evaluated once on the test split.
`results.json` records `final_test` and the split boundaries; `metrics.json`
also includes configuration, training history and dataset metadata. `curve.json`
contains training losses and update counts only. `final.pth` is the sole saved
checkpoint; there is no metric-based checkpoint selection or early stopping.
MAE/RMSE are computed separately per forecast step across all test windows and
nodes, then averaged over the six steps.

Optional mechanism controls retain the same architecture and training budget:

```bash
python run.py --config configs/metr-la.json --tsb_projection none
python run.py --config configs/metr-la.json --tsb_source_lr 0
```

The second control is Target Updates Only; source calculations still execute for
matched control behavior. It is not the historical 100-epoch Target-only protocol.
