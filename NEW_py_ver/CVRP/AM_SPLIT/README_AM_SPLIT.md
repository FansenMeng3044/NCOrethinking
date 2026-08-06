# Attention Model + giant tour + Split for CVRP

This directory is a clone of Wouter Kool's official
`attention-learn-to-route` repository with an additional `am_split` problem.
The upstream commit is `c9abf41ac2f878a55b20dc7e829bc942bb999631`.

## Method

The policy receives depot/customer coordinates only. The depot is encoded as
context and permanently masked from the action space. Customer demand and
remaining capacity are not policy inputs. AM generates a permutation of all
customers and the exact Bellman Split decoder places capacity-feasible depot
returns without changing that order.

Two training objectives are available:

- `--reward-mode split` (default): train with the exact hard-capacity Split cost,
  matching `POMO_SPLIT`.
- `--reward-mode raw`: train with the unsplit giant-tour length and use Split only
  for evaluation, matching `POMO_SPLIT_EVALONLY`.

Evaluation always uses hard-capacity Split. It supports greedy decoding, sampled
best-of-width decoding, and 1x/8x geometric augmentation. Sampling width 100 plus
8x augmentation is the closest candidate-budget analogue of POMO=100 plus 8x,
although AM sampling and POMO multi-start are not identical search procedures.

## Environment

The independent environment is installed at `/home/ubuntu/anaconda3/envs/am`:

```bash
conda activate am
cd /root/mfs/pomo/NEW_py_ver/CVRP/AM_SPLIT
```

`environment-am.yml` records the environment dependencies.

## Training

Default CVRP100 Split-reward training uses 10,000 generated instances per epoch,
8,100 total epochs, the official AM rollout baseline, and saves after every 100
completed epochs as `epoch-100.pt`, `epoch-200.pt`, and so on:

```bash
python train_am_split.py --reward-mode split
```

To reproduce the EVALONLY training objective:

```bash
python train_am_split.py --reward-mode raw
```

A quick CPU integration run is:

```bash
python train_am_split.py --smoke
```

## Evaluation

The default dataset is the same fixed CVRP100 seed-1234 `.pt` file used by the
POMO evaluation. The default search is sampling width 100 with 8x augmentation:

```bash
python eval_am_split.py \
  --model outputs/am_split_100/<run-directory> \
  --decode-strategy sampling --width 100 --augmentation 8
```

Greedy evaluation without augmentation:

```bash
python eval_am_split.py \
  --model outputs/am_split_100/<run-directory> \
  --decode-strategy greedy --augmentation 1
```

Pass `--output result.json` to persist the metric summary. Evaluation reports
both `no_aug_split_score` and `aug_split_score`.

## Tests

```bash
python -m unittest discover -s tests -v
```
