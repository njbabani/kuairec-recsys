# KuaiRec Recommender

An end-to-end short-video recommender built on [KuaiRec](https://kuairec.com/), with a
**simulated A/B testing framework** that uses real user reactions instead of made-up ones.

KuaiRec comes from Kuaishou, a TikTok-style app. Besides a normal logged feed, it contains a
*fully observed* matrix: 1,411 users × 3,327 videos with 99.6% of cells filled. For those users
and videos we know how each user reacted to (almost) every video, so an offline A/B test can
show each arm different videos from that set and look up the real reaction.

> Status: **Phase 2 complete** (data pipeline, EDA, debiased labels, evaluation harness,
> tracked baselines). See the
> [roadmap](#roadmap).

## What this project demonstrates

| Area | How | Status |
|---|---|---|
| Reproducible data pipeline | DVC stages, checksummed + retrying download, `params.yaml` as single source of truth | Done |
| Data contracts & quality | pandera schemas, train/eval leakage guard, referential integrity, duplicate audit, summary metrics | Done |
| Engineering hygiene | `uv` lockfile, ruff (incl. bandit + timezone rules), pytest (80% coverage gate), GitHub Actions CI | Done |
| Exploratory analysis | [EDA notebook](notebooks/01_eda.ipynb): duration bias, exposure bias, collection windows | Done |
| Bias-aware labeling | Duration-debiased relevance labels (log-spaced length buckets), fitted on train only, with a residual-bias metric | Done |
| Evaluation design | Temporal train/valid split; fully observed matrix split by user into tune (selection) and test | Done |
| Classical recsys | Random, popularity (views vs. engagement with empirical-Bayes smoothing), ALS matrix factorisation with grid search | Done |
| Deep learning | Two-tower retrieval model in PyTorch (Apple MPS) | Planned |
| Learning to rank | LightGBM LambdaRank re-ranker + SHAP explanations | Planned |
| Offline evaluation | Per-user NDCG / precision / recall / MAP@K with bootstrap CIs, paired model comparisons, coverage, Gini, popularity bias | Done |
| Experiment tracking | Every run logged to MLflow (local) and Weights & Biases (offline by default), W&B Sweeps config for Bayesian HPO | Done |
| A/B testing | Power analysis, A/A tests, SRM check, CUPED, sequential testing | Planned |
| Demo | Streamlit app: user lookup, model leaderboard, interactive A/B lab | Planned |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you), ~1.5 GB free disk
and ~5 GB RAM for the `prepare` stage (it peaks at ~4 GB).

```bash
git clone <this-repo> && cd recommender
make setup   # create the environment from uv.lock
make data    # download KuaiRec (432 MB), validate, label and split
make eda     # re-run the EDA notebook and refresh reports/figures
make experiments  # ALS search + baselines, logged to MLflow and W&B (~3 min)
make results      # refresh the results notebook and figures
make test    # run the test suite
```

Run `make` to list every command.

## Pipeline

The pipeline is defined in [`dvc.yaml`](dvc.yaml); `dvc repro` re-runs only the stages whose
code, parameters or inputs changed, and `dvc.lock` pins the exact hash of every output.

```
download ──> prepare ──> split ──┬──> als_search    (ALS grid on the tune users)
                                 └──> baselines     (all baselines + paired comparison)
```

| Stage | Does | Output |
|---|---|---|
| `download` | Streams `KuaiRec.zip` from Zenodo with retries, verifies its MD5, writes atomically; reuses a verified local copy | `data/raw/KuaiRec.zip` |
| `prepare` | Cleans and types six tables, checks train/eval leakage and referential integrity, validates schemas | `data/processed/*.parquet`, [`reports/data_summary.json`](reports/data_summary.json) |
| `split` | Temporal train/valid split, user-hash tune/test split, fits debiased labels on train, labels all splits | `data/splits/{train,valid,tune,test}.parquet`, [`reports/split_summary.json`](reports/split_summary.json) |
| `als_search` | Fits an ALS grid on train, scores it on the tune users, logs every trial | [`reports/metrics/als_search.json`](reports/metrics/als_search.json) |
| `baselines` | Fits every baseline, evaluates on the tune users, paired comparison with the reference | [`reports/metrics/baselines.json`](reports/metrics/baselines.json) |

No DVC remote is configured: the raw data is public, so `dvc repro` rebuilds everything from
Zenodo, and `dvc.lock` records the hash of every output so any drift shows up in `dvc status`.

## Key findings from the EDA

Full analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

**1. `watch_ratio` mostly measures how short a video is.** People watch a median of ~7.5
seconds whatever the video's length, so a naive "watch ratio ≥ 2 = liked" label is positive for
two thirds of the shortest videos' views and almost none of the longest. A model trained on it
learns "recommend short videos". Instead, a view is positive when its watch ratio is above the
80th percentile of *training views of similar-length videos*, using log-spaced length buckets at
most 15% wide (the duration-bucketing idea of D2Q, Zhan et al., KDD 2022). Measured on 10%-wide
length slices, the positive rate now stays within roughly 9–25% instead of 0.5–62%; the split
summary tracks this so a regression shows up.

![Positive rate by video length: naive vs debiased label](reports/figures/03_label_positive_rate_by_length.png)

**2. In the logs, *how often* a video was shown says nothing about its appeal; *how well it was
received* does.** Per video, logged view counts are uncorrelated with the positive rate when
everyone sees it, while the logged positive rate is clearly predictive. Phase 2 tests the
implication: popularity by count should be a weak baseline, popularity by engagement rate a
strong one.

![Logged exposure and engagement vs appeal in the fully observed matrix](reports/figures/05_exposure_vs_preference.png)

**3. The logs come from three collection windows** (Jul 5–12, Aug 1–10, Aug 27–Sep 5), and
the last 7 days are the validation set. Validation differs from the fully observed test set: it
has re-watches (13.7% of views) and videos never seen in training (12.5%), the test set has
neither. So validation is used for training-time checks, and model selection uses a held-out
set of fully observed users instead.

![Daily views with the validation window](reports/figures/06_daily_views_and_split.png)

## Baseline results (tune users)

Details and charts in [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb). Each model
ranks every candidate video for each of the 299 tune users; metrics are per user, then averaged.
These are **selection-set scores**: the same users chose ALS's hyperparameters (see below for
the correction); final numbers will come from the untouched test users.

| Model | NDCG@10 [95% CI] | Coverage@10 | Popularity of recommendations |
|---|---|---|---|
| random | 0.164 [0.143, 0.185] | 59% | 50th percentile |
| popularity by views | 0.184 [0.162, 0.206] | 0.3% | 100th percentile |
| popularity by engagement | 0.217 [0.195, 0.240] | 0.3% | 66th percentile |
| **ALS (tuned)** | **0.274** [0.246, 0.302] | 22% | 81st percentile |

- **The EDA hypothesis holds:** ranking by view count barely beats random, ranking by engagement
  rate is the strongest non-personalized baseline.
- **Paired comparisons are the right test.** Separate intervals for ALS and engagement
  popularity overlap, but on the same users ALS is better by **+0.057 NDCG@10
  [+0.031, +0.083]**: pairing removes each user's own base rate, the biggest source of variance
  (the same idea behind paired designs in A/B tests). The gain is uneven: ALS wins for 57% of
  users. Intervals cover user sampling only, not seeds or the choice of candidate videos.
- **Winner's curse, corrected.** ALS's settings were picked from 36 trials on these same users.
  A bootstrap over users (pick the best trial in-sample, score it on the users left out)
  estimates **0.015 NDCG@10** of selection optimism, so ~0.259 is the fairer expectation for new
  users, still well above every popularity baseline.
- **Accuracy vs. popularity bias.** Across the ALS trials, the more accurate settings recommend
  more popular videos; the chosen one draws its top 10 from around the 81st popularity
  percentile. The top settings are within 0.001 of each other (a flat plateau).
- **What the evaluation actually tests.** Tune users have *no* training views of the 3,327
  candidate videos, so ALS must transfer taste learned from their other videos (median ~33
  positive views) to new ones. ALS uses Hu et al.'s confidence `1 + alpha·count`; users with no
  positive training views fall back to the average user vector.

![Paired comparison with the strongest popularity baseline](reports/figures/09_baselines_paired_comparison.png)

![ALS trials: accuracy vs popularity of recommendations](reports/figures/10_als_accuracy_vs_popularity.png)

## Experiment tracking

Every training run is logged to both trackers through one small interface
([`tracking.py`](src/recsys/tracking.py)):

- **MLflow** (local, `mlflow.db`): `make mlflow-ui`, then open http://127.0.0.1:5000.
- **Weights & Biases** runs *offline* by default, so nothing is uploaded and no account is
  needed. To publish: `wandb login` once, then either run with `WANDB_MODE=online` or upload past
  runs with `wandb sync wandb/offline-run-*`.
- **W&B Sweeps** (Bayesian search, needs a login): `make sweep-als`, then start the
  `wandb agent ...` command it prints. The local grid (`make experiments`) needs no account.
  Sweeps also select on the tune users, so re-check any sweep winner before trusting it.
- MLflow runs are tagged with the git commit (and whether the tree was dirty). Note that
  `wandb sync` uploads run metadata such as hostname and file paths.

## Evaluation design and limitations

| Split | Source | Used for |
|---|---|---|
| train | logged feed, before Aug 30 | fitting models and label cut-offs |
| valid | logged feed, Aug 30 onwards | training-time checks (early stopping) |
| tune | fully observed matrix, 20% of users (stable SHA-256 hash) | model selection and EDA |
| test | fully observed matrix, other 80% of users | final report and A/B simulation only |

What this design does and doesn't guarantee:

- **No training on the future, for validation only.** The fully observed matrix spans the same
  weeks as training. It measures preferences rather than forecasting, and none of its
  (user, video) pairs appear in training (enforced in `prepare`).
- **"Fully observed" has a scope.** It covers 3,327 videos (about a third of the catalogue,
  mostly frequently shown ones) and heavy users, with one reaction per pair recorded in a feed.
  The A/B simulation can only recommend from these videos, and its results describe these users.
- **Users differ a lot in how much they watch**, even after the duration fix, so pooled metrics
  like overall AUC mostly measure *which users* engage. Ranking metrics are computed per user.
- **The test users are never opened** during analysis; the EDA uses the tune users only.

## Data notes

Findings from building the `prepare` stage (full numbers in `reports/data_summary.json`):

- **No train/eval leakage.** None of the 4.68M (user, video) pairs in the evaluation matrix
  appear in the training matrix. The pipeline enforces this and refuses to write if it breaks.
- **Referential integrity.** Every user and video in both interaction matrices exists in the
  user and video tables; also enforced by the pipeline.
- **Duplicate logs.** 968k rows (7.7%) of the training matrix are duplicate log entries and are
  dropped; ~2.2k of them are the same event filed under two daily partitions (the source
  `date` column is a log-partition label, not the event date). Re-watches (same pair,
  different time) are kept.
- **Extreme watch ratios.** Median `watch_ratio` is ~0.73 but the max is 573 (videos left
  looping). Kept as-is in the data layer; capped during modelling.
- **Missing timestamps.** 3.9% of evaluation rows have no time; the schema allows this there
  and forbids it in training data.
- **Timezone.** Source times are Asia/Shanghai local; stored as a timezone-aware `event_time`
  that reproduces the source `time` column exactly.
- **Not ingested (yet).** `social_network.csv` (472 users with friend lists) is skipped until a
  model needs it.

## Project structure

```
├── dvc.yaml / dvc.lock      # pipeline definition + pinned output hashes
├── params.yaml              # every tunable setting, read by DVC and recsys.config
├── src/recsys/
│   ├── config.py            # typed, validated access to params.yaml
│   ├── io.py                # shared parquet / JSON writers
│   ├── hashing.py           # stable SHA-256 user assignment (tune/test, A/B arms)
│   ├── data/
│   │   ├── download.py      # checksummed, atomic download
│   │   ├── prepare.py       # clean, leakage-check, validate, write parquet
│   │   ├── schemas.py       # pandera data contracts
│   │   ├── labels.py        # duration-debiased relevance labels
│   │   ├── split.py         # train/valid/tune/test split + labeling stage
│   │   └── categories.py    # English names for the Chinese category labels
│   ├── tracking.py          # MLflow + W&B behind one interface
│   ├── evaluation/
│   │   ├── ranking.py       # per-user metrics, bootstrap CIs, paired comparisons
│   │   └── concentration.py # Gini, Lorenz curve, top-k share
│   ├── models/              # random, popularity, ALS (common Recommender protocol)
│   ├── experiments/         # baselines + ALS search (DVC stages), W&B sweep trial
│   └── viz/
│       └── style.py         # chart style with a colorblind-safe palette
├── notebooks/               # jupytext .py sources + executed .ipynb
├── sweeps/                  # W&B sweep configs
├── tests/                   # synthetic fixtures; no network or real data needed
└── reports/                 # git-tracked pipeline metrics and figures
```

## Roadmap

0. **Setup & data pipeline** (done)
1. **EDA, debiased labels and train/validation split** (done)
2. **Evaluation harness, popularity and ALS baselines, MLflow + W&B tracking** (done)
3. Two-tower neural retrieval model
4. LightGBM re-ranker with feature engineering and SHAP
5. A/B testing framework on the fully observed matrix
6. Streamlit demo app

## License

Code: [MIT](LICENSE). The KuaiRec data is not redistributed here; it is downloaded from Zenodo
under its own license (below).

## Dataset & citation

KuaiRec is released under [CC BY 4.0](https://zenodo.org/records/18164998). If you use it,
cite the original paper:

> Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He,
> Jiaxin Mao, Tat-Seng Chua. *KuaiRec: A Fully-observed Dataset and Insights for Evaluating
> Recommender Systems.* CIKM 2022. [doi:10.1145/3511808.3557220](https://doi.org/10.1145/3511808.3557220)
