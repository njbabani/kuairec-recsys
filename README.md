# KuaiRec Recommender

An end-to-end short-video recommender built on [KuaiRec](https://kuairec.com/), with a
**simulated A/B testing framework** that uses real user reactions instead of made-up ones.

KuaiRec comes from Kuaishou, a TikTok-style app. Besides a normal logged feed, it contains a
*fully observed* matrix: 1,411 users × 3,327 videos with 99.6% of cells filled. For those users
and videos we know how each user reacted to (almost) every video, so an offline A/B test can
show each arm different videos from that set and look up the real reaction.

> Status: **Phase 5 complete** (data pipeline, EDA, debiased labels, evaluation harness,
> tracked baselines, two-tower neural model, two-stage re-ranking with SHAP, final test
> evaluation, A/B testing framework checked against known true effects). See the
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
| Classical recsys | Random, popularity (views vs. engagement with empirical-Bayes smoothing), category affinity, ALS matrix factorisation with grid search | Done |
| Feature pipeline | Label-free user and video feature tables (profile, categories, length) as their own DVC stage | Done |
| Deep learning | Two-tower model in PyTorch (CPU or Apple MPS): pointwise vs. in-batch softmax with logQ correction, feature towers that can score unseen videos, multi-seed selection and seed ensembles | Done |
| Training infrastructure | Atomic, resumable checkpoints (training state every epoch, search progress per scored epoch, `weights_only` loading); trained models versioned as DVC outputs | Done |
| Learning to rank | Two-stage pipeline: two-tower retrieval, then a LightGBM LambdaRank re-ranker over ~20 features, explained with TreeSHAP | Done |
| Offline evaluation | Per-user NDCG / precision / recall / MAP@K with bootstrap CIs, paired model comparisons, coverage, Gini, popularity bias; one final evaluation on untouched test users | Done |
| Experiment tracking | Every run logged to MLflow (local) and Weights & Biases (offline by default), W&B Sweeps config for Bayesian HPO | Done |
| A/B testing | Power analysis (MDE, sample size with its uncertainty), hash-based assignment checked over 2,000 salts, Welch t-test, CUPED, SRM and covariate-balance checks, guardrail metric, ship rule, A/A tests, peeking and always-valid p-values (mSPRT); every method checked against the true effect over 2,000 re-randomised experiments | Done |
| Demo | Streamlit app: user lookup, model leaderboard, interactive A/B lab | Planned |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you), ~1.5 GB free disk
and ~5 GB RAM for the `prepare` stage (it peaks at ~4 GB). On macOS, LightGBM needs the OpenMP
runtime: `brew install libomp`.

```bash
git clone <this-repo> && cd recommender
make setup   # create the environment from uv.lock
make data    # download KuaiRec (432 MB), validate, label and split
make eda     # re-run the EDA notebook and refresh reports/figures
make experiments  # model searches, re-ranker, test evaluation, A/B (~30 min on an M4)
make results      # refresh the results notebooks and figures
make test    # run the test suite
```

Run `make` to list every command.

## Pipeline

The pipeline is defined in [`dvc.yaml`](dvc.yaml); `dvc repro` re-runs only the stages whose
code, parameters or inputs changed, and `dvc.lock` pins the exact hash of every output.

```
download ──> prepare ──┬──> split ────┬──> als_search        (ALS grid on the tune users)
                       │              ├──> two_tower_search  (two-tower grid, scored every epoch)
                       └──> features ─┴──> leaderboard       (every model + paired comparisons)
                                                 │
                                                 └──> retrieval ──> reranker  (stage 2 + SHAP)
                                                                       │
                                                  final_evaluation <───┘  (test users, once)
                                                         │
                                                         └──> ab_test  (simulated A/B tests)
```

`two_tower_search`, `leaderboard` and `final_evaluation` read both the splits and the features.
`retrieval` scores every pair the later stages need with the saved two-tower models, so
`reranker` and `final_evaluation` (LightGBM) never load PyTorch: on macOS the two libraries'
OpenMP runtimes crash when they share a process.

| Stage | Does | Output |
|---|---|---|
| `download` | Streams `KuaiRec.zip` from Zenodo with retries, verifies its MD5, writes atomically; reuses a verified local copy | `data/raw/KuaiRec.zip` |
| `prepare` | Cleans and types six tables, checks train/eval leakage and referential integrity, validates schemas | `data/processed/*.parquet`, [`reports/data_summary.json`](reports/data_summary.json) |
| `split` | Temporal train/valid split, user-hash tune/test split, fits debiased labels on train, labels all splits | `data/splits/{train,valid,tune,test}.parquet`, [`reports/split_summary.json`](reports/split_summary.json) |
| `features` | User features (profile categories, log counts) and video features (categories, log length); nothing derived from reactions | `data/features/{users,videos}.parquet` |
| `als_search` | Fits an ALS grid on train, scores it on the tune users, logs every trial | [`reports/metrics/als_search.json`](reports/metrics/als_search.json) |
| `two_tower_search` | Trains each two-tower configuration once, scores it on tune after every epoch, logs a training curve per configuration | [`reports/metrics/two_tower_search.json`](reports/metrics/two_tower_search.json) |
| `leaderboard` | Fits every model with its chosen settings, evaluates on the tune users, paired comparisons with engagement popularity and ALS; saves the trained two-tower members | [`reports/metrics/leaderboard.json`](reports/metrics/leaderboard.json), `models/two_tower/` |
| `retrieval` | Scores every (user, video) pair of the validation week and the tune/test matrices with the saved two-tower ensemble | `data/retrieval/two_tower_scores.parquet` |
| `reranker` | Trains the LightGBM re-ranker on the validation week (tuned on held-out validation users), evaluates the two-stage pipeline on the tune users, explains it with TreeSHAP | [`reports/metrics/reranker.json`](reports/metrics/reranker.json), `models/reranker/` |
| `final_evaluation` | Scores every frozen model once on the 1,112 untouched test users, with paired comparisons; writes each policy's 10-video session for every test user | [`reports/metrics/test_evaluation.json`](reports/metrics/test_evaluation.json), `data/ab/sessions.parquet` |
| `ab_test` | Looks up what each test user did with every policy's session, splits users into arms by hash, runs the power analysis, the planned experiments and the A/A, peeking and logging-bug checks | [`reports/metrics/ab_test.json`](reports/metrics/ab_test.json), `data/ab/user_outcomes.parquet` |

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
the correction); final numbers come from the untouched test users
([final results](#final-results-test-users)).

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

## Two-tower results (tune users)

Details and charts in [`notebooks/03_two_tower.ipynb`](notebooks/03_two_tower.ipynb). Same users
and protocol as above; ALS and the two-tower model were both tuned on these users.

| Model | NDCG@10 [95% CI] | Recall@50 | Coverage@10 | Popularity of recommendations |
|---|---|---|---|---|
| category affinity | 0.199 [0.175, 0.223] | 0.020 | 6% | 59th percentile |
| ALS (tuned) | 0.274 [0.246, 0.302] | 0.030 | 22% | 81st percentile |
| **two-tower (3-seed ensemble)** | **0.298** [0.271, 0.325] | **0.035** | 11% | **49th percentile** |

- **The objective matters more than the architecture.** The pointwise loss (*will this user like
  this video, given it was shown?*) beats the in-batch softmax with logQ correction (*which
  video did they like?*) in every configuration: 0.287 vs 0.252 seed-averaged. The pointwise loss
  asks the same question as the evaluation and also learns from views that were not positive.
  (With the same epochs, the softmax loss gets ~5x fewer updates, so this compares the two as
  configured here.)
- **One training run is a noisy measurement.** The first search picked a single-seed peak of
  0.294 that retrained to 0.278. Across seeds a single run varies by ±0.003–0.006, as much as
  the gaps being compared, so the search now trains every configuration with 3 seeds and
  selects on the average (best: 0.287 ± 0.003, 0.277 after selection optimism of 0.010).
  On CPU, training is deterministic: the leaderboard's retrained members match the search to
  every digit.
- **Against ALS, on the same users:** the ensemble gains **+0.024 NDCG@10 [−0.000, +0.048]**
  and ranks better for 56% of users. That is borderline on the tune users; the untouched test
  users settle it (+0.017 [+0.005, +0.029]). Against engagement popularity the gain is clear:
  **+0.081 [+0.058, +0.103]**, better for 67% of users.
- **Accurate without leaning on popularity.** The two-tower model's top 10 sits at the 49th
  popularity percentile (random is 50th) versus the 81st for ALS, and it has the best
  recall@50.
- **Category taste alone does not help.** Re-weighting engagement popularity by each user's
  per-category lift *lowers* NDCG@10 (−0.018 [−0.034, −0.002]): first-level categories are too
  coarse to describe taste, and the lifts add noise. Personalisation needs finer signals (ids,
  embeddings).
- **Overfitting is real and fast.** Every configuration peaks around epochs 5–7 and then
  declines (the 64-dimensional pointwise model falls from 0.285 at epoch 5 to 0.260 at epoch
  10, seed-averaged), which is why every epoch is scored.

![Leaderboard](reports/figures/11_leaderboard.png)

![Training curves of the two-tower search](reports/figures/13_two_tower_training_curves.png)

![Accuracy vs popularity of recommendations, every model](reports/figures/14_accuracy_vs_popularity.png)

## Two-stage results: retrieval + re-ranking (tune users)

Details and charts in [`notebooks/04_reranker.ipynb`](notebooks/04_reranker.ipynb). Stage 1 (the
two-tower ensemble) keeps each user's top 200 of ~3,300 candidates; stage 2, a LightGBM
LambdaRank model over 19 features, re-orders them. The re-ranker learns from the validation week
(logged views from Aug 30, one per pair, every feature computed from data before it) of 4,248
users outside the fully observed matrix; 533 more stop its boosting and pick its configuration,
and 497 are kept only for reporting. The 1,411 tune and test users play no part in training it.

| Model | NDCG@10 [95% CI] | Recall@50 | Popularity of recommendations |
|---|---|---|---|
| ALS | 0.274 [0.246, 0.302] | 0.030 | 81st percentile |
| two-stage (two-tower + re-ranker) | 0.293 [0.267, 0.318] | **0.036** | 56th percentile |
| two-tower alone | **0.298** [0.271, 0.325] | 0.035 | 49th percentile |

Each comparison below is paired on the same users, against ordering by the two-tower score:

| Where | Δ NDCG@10 [95% CI] | Better for |
|---|---|---|
| untouched validation users (logged views) | **+0.043** [+0.026, +0.061] | 61% of users |
| tune users, top-200 shortlist re-ranked | −0.005 [−0.023, +0.014] | 48% |
| tune users, every candidate re-ranked | **−0.040** [−0.060, −0.018] | 42% |
| tune users, re-ranker cross-fitted on other tune users (upper bound) | +0.002 [−0.019, +0.024] | 48% |

- **A clear gain on logged data, a smaller one on real preferences.** On validation users it
  never saw, the re-ranker beats two-tower ordering by +0.043. The 299 tune users show no
  detectable gain, but the 1,112 untouched test users, opened once at the end, do: **+0.017
  [+0.006, +0.027]** ([final results](#final-results-test-users)). *Correction:* this section
  first read the tune result as "no gain on real preferences". That comparison was too small,
  and it favoured the two-tower model, which was selected on those same users.
- **Part of the logged-data gain does not carry over.** The gain on logged views (+0.043) is
  larger than on the fully observed test users (+0.017): some of what the ranker learns is
  specific to the logged feed. Cross-fitting it on the fully observed shortlists of other tune
  users (a diagnostic only) gives +0.002 [−0.019, +0.024], an interval too wide to separate
  "no gain" from the test gain.
- **The shortlist protects the re-ranker.** Asked to order all ~3,300 candidates, including a
  long tail unlike the views it learned from, it does clearly worse (−0.040). Retrieval keeps it
  on familiar ground, which is one practical reason two-stage systems shortlist first.
- **What reorders a user's list (TreeSHAP, centred within each user):** engagement popularity
  first, then video length (shorter videos move up: both the validation week and the fully
  observed users like short videos slightly more often, as the labels are length-neutral only on
  the training weeks), the two-tower and ALS scores, and video age. Platform-wide like, share
  and comment rates matter little.
- **Retrieval is the ceiling.** The top 200 hold 11% of a tune user's liked videos on average;
  a perfect shortlist would hold 58% (most users like more than 200 of the candidates, so not
  all of their liked videos fit).
- **Fairness.** ALS and the two-tower model were tuned on the tune users and the re-ranker was
  not, which favours them slightly; a single two-tower run also varies by ±0.006. The untouched
  test users give the final comparison, below.

![Re-ranking gains: logged vs fully observed users](reports/figures/15_two_stage_paired.png)

![What reorders a user's shortlist (within-user TreeSHAP)](reports/figures/18_reranker_shap_beeswarm.png)

## Final results (test users)

The 1,112 test users were opened once, after every model was frozen. The protocol is the same
as above (each model ranks all 3,327 candidate videos for each user). Details in
[`notebooks/05_ab_testing.ipynb`](notebooks/05_ab_testing.ipynb).

| Model | NDCG@10 [95% CI] | Recall@50 | Coverage@10 | Popularity of recommendations |
|---|---|---|---|---|
| random | 0.149 [0.139, 0.159] | 0.015 | 97% | 50th percentile |
| popularity by engagement | 0.208 [0.196, 0.220] | 0.026 | 0.4% | 65th percentile |
| ALS | 0.271 [0.256, 0.286] | 0.036 | 35% | 81st percentile |
| two-tower (3-seed ensemble) | 0.288 [0.274, 0.303] | **0.043** | 16% | 49th percentile |
| **two-stage (two-tower + re-ranker)** | **0.306** [0.290, 0.320] | 0.042 | 13% | 56th percentile |

| Paired on the same users | Δ NDCG@10 [95% CI] | Better for |
|---|---|---|
| two-stage vs two-tower | **+0.017** [+0.006, +0.027] | 52% of users |
| two-tower vs ALS | **+0.017** [+0.005, +0.029] | 54% |
| ALS vs engagement popularity | **+0.063** [+0.050, +0.076] | 58% |

- **The test users settle both close calls.** The two-tower model beats ALS (borderline on the
  tune users), and re-ranking its shortlist adds a real gain (none was detectable on the tune
  users; see the correction above).
- **Small, real, uneven gains.** Each step up is worth about +0.017 NDCG@10 and helps only
  slightly more users than it hurts (52–54%). Paired comparisons are what make gains this small
  detectable on 1,112 users.
- **Compare models on the same users, not scores across user sets.** Every model except the
  two-stage pipeline scores lower on test than on tune, even random (0.164 → 0.149), because the
  test users like fewer of the candidates (15.2% against 16.2%). Score levels move with the users; paired differences
  on the same users do not.
- **Accurate without leaning on popularity.** The two neural pipelines recommend around the
  49th–56th popularity percentile, ALS around the 81st.

![Final test leaderboard](reports/figures/19_test_leaderboard.png)

## A/B testing on real reactions (test users)

Details in [`notebooks/05_ab_testing.ipynb`](notebooks/05_ab_testing.ipynb), which explains each
idea from scratch.

**The setup.** An A/B test splits users at random. Control sees the current recommender,
treatment sees the new one, and the difference in a metric between the groups estimates the new
one's effect. Here each policy builds a 10-video session for every test user (its top 10). The
matrix is fully observed, so we know what every user would have done with *every* policy's
session:

- **Primary metric:** videos liked per session.
- **Guardrail:** seconds watched per session (a launch must not hurt it).
- **Assignment:** a salted hash of the user id splits users about 50/50 (534 control, 578
  treatment), and the same user always lands in the same arm.

The analysis sees only each user's assigned arm, as in a real test. Because both outcomes are
known, the **true effect** is known too: the average over all 1,112 users of treatment minus
control. Each method is checked against it by re-randomising the split 2,000 times.

| Policy | random | popularity | ALS | two-tower | two-stage |
|---|---|---|---|---|---|
| Videos liked per session (all test users) | 1.48 | 2.07 | 2.65 | 2.85 | 3.00 |

**1. Power analysis, before running anything.** Per user, videos liked varies with a standard
deviation of 2.37. With 534 and 578 users, the smallest effect the test detects 80% of the time
(the *minimum detectable effect*, MDE) is **0.40 liked videos per session**, 15% of ALS's 2.65.
CUPED (below) lowers it to 0.31.

**2. Three planned experiments.**

| Experiment (control → treatment) | True effect | Estimate [95% CI] | With CUPED | Decision | Power (plain / CUPED) | Users for 80% power (both arms) |
|---|---|---|---|---|---|---|
| popularity → ALS | +0.58 | +0.39 [+0.14, +0.64] | +0.46 [+0.27, +0.65] | ship | 99.9% / 100% | 427 (295–675) |
| ALS → two-tower | +0.21 | −0.02 [−0.30, +0.26] | +0.05 [−0.18, +0.27] | inconclusive | 30% / 43% | 4,216 (1,778–19,906) |
| two-tower → two-stage | +0.14 | −0.11 [−0.40, +0.18] | −0.04 [−0.26, +0.19] | inconclusive | 15% / 20% | 9,253 (3,248–94,975) |

*Power* is the share of the 2,000 re-randomised experiments that detect the true effect. The
users needed are for the plain test (no CUPED), with each arm's own spread. The range in
brackets covers the 95% interval of the effect *new* users could show: these users pin it down
only so far. The decision uses the CUPED estimate, the guardrail and the sample-ratio check.

- **Only the large effect is detected, and that is the right answer.** Both newer models really
  are better (+0.21 and +0.14 liked videos per session), but both effects are below the MDE.
  "Inconclusive" means *not enough evidence*, not *no effect*. Detecting them reliably would
  take about 4,200 and 9,300 users, and planning is less certain than one number suggests:
  for the smallest effect anywhere from 3,200 to 95,000.
- **Why offline evaluation sees what this A/B test cannot.** The offline comparison scores every
  model on the *same* users, so each user's own appetite for liking videos cancels out. An A/B
  test compares *different* users, and their liked counts vary far more than the effects being
  measured. That is why teams screen ideas offline (or with interleaving) and spend A/B traffic
  on the decisions that need it.
- **CUPED cuts variance by 35–44%**, as if there were 1.5–1.8× more users. It subtracts the part
  of each user's outcome predicted by their behaviour *before* the experiment (their positive
  rate in the training weeks, correlation about 0.6 with liked videos). That behaviour cannot have
  been changed by the treatment, so the estimate stays unbiased: across re-randomisations its
  average matches the truth to within 0.003.
- **The guardrail only catches harms large enough to see.** Compared with popularity, ALS
  really lowers watch time by 1.2 s per session. The test cannot see a change that small
  (−3.0 s [−9.9, +3.8]), so the experiment ships.

**3. Checking the machinery.**

- **A/A test.** The two-tower model is compared with itself over 2,000 random splits. 4.65% of
  the splits come out "significant" (4.75% with CUPED), matching the 5% false-alarm rate the test
  promises, and the p-values are spread evenly.
- **Is the hash fair?** The simulations flip a coin per user, but the real split is a hash. The
  same A/A comparison split by 2,000 other salts gives 5.3% false alarms, 4.2% pre-experiment
  imbalance in the covariate and 5.3% sample-ratio alarms, all close to the 5% a fair coin
  gives. The salt actually used splits 534/578 (SRM p = 0.19) with balanced covariates (p = 0.41).
- **Interval coverage.** The 95% intervals contain the true effect 96–99% of the time. This is
  slightly conservative, as expected when the users are fixed and only the assignment is random.
- **Peeking.** Checking after every tenth of the users and stopping at the first p < 0.05
  raises false alarms from 5% to **20%**. Always-valid p-values (mSPRT) may be checked at every
  look and keep them below 5% (1% here: the guarantee is conservative). The price is power: for
  the real ALS → two-tower effect it falls from 28% (one look at the end) to 8%, partly because
  the test was tuned to effects the size of the MDE (0.40), about twice this one.
- **Sample ratio mismatch (SRM).** A simulated logging bug: the new version only logs a session
  once something is liked, so the least engaged treatment users vanish. Both arms run the same
  policy, so any lift is fake. The analysis is the full pipeline (CUPED, SRM check, decision);
  CUPED cannot undo this bias, because users are lost for how they reacted *during* the
  experiment.

  | Treatment users lost | Estimate (CUPED) [95% CI] | SRM p-value | Decision |
  |---|---|---|---|
  | none | −0.16 [−0.39, +0.07] | 0.19 | inconclusive |
  | 5% (29) | −0.08 [−0.31, +0.16] | 0.65 | inconclusive |
  | 10% (58) | +0.02 [−0.22, +0.25] | 0.67 | inconclusive |
  | 20% (116) | +0.24 [−0.01, +0.49] | 0.02 | inconclusive |
  | 30% (173) | +0.47 [+0.22, +0.73] | 0.00003 | **invalid: sample ratio mismatch** |

  At 30% the bug fakes a significant lift (+0.63 over the bug-free estimate, p = 0.0003) and the
  SRM check rightly blocks it. Smaller losses fake smaller lifts and slip through: the hash had
  already given treatment 44 extra users, so losing 58 of them brings the arms *closer* to 50/50.
  An SRM check is a test with limited power, so a p-value below 0.05 deserves a look even when
  it is above the strict 0.001 alarm level.

![Experiment readouts against the true effects](reports/figures/20_ab_readouts.png)

![A/A p-values and power](reports/figures/21_aa_and_power.png)

![Peeking inflates false alarms; always-valid p-values do not](reports/figures/22_peeking.png)

![A logging bug fakes a lift; the SRM check catches only the large loss](reports/figures/23_srm_demo.png)

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

## Checkpointing and saved models

Long training is resumable, and finished models are kept:

- **Training checkpoints.** `TwoTowerRecommender.fit(..., checkpoint_path=...)` saves the
  weights, optimizer state and shuffling state after every epoch; rerunning with the same
  settings and data continues after the last saved epoch (and reproduces an uninterrupted run
  exactly on CPU). A checkpoint made for other settings or data is ignored.
- **Search progress.** `two_tower_search` records every scored (configuration, seed, epoch);
  if it is interrupted, `dvc repro` picks up where it stopped instead of redoing finished work.
  The resume state lives in `checkpoints/` and is deleted once the stage succeeds.
- **Saved models.** The leaderboard saves each trained two-tower member to `models/two_tower/`
  (a DVC output); `TwoTowerEnsemble.load("models/two_tower")` restores it, features included.
- **Safe by construction.** Files are written to a temporary name and renamed (a crash never
  leaves a half-written checkpoint), and read with `torch.load(weights_only=True)`, so opening
  a checkpoint cannot run code.

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
- **Training noise is part of the uncertainty.** One two-tower training run varies by about
  ±0.006 NDCG@10 between seeds, which the user-bootstrap intervals do not include. Searches
  select on seed averages and the leaderboard reports the seed spread.
- **Selection optimism.** ALS and the two-tower model were tuned on the tune users; the
  searches estimate how much that flatters them, and the test users give the final numbers.
- **User profile features** (activity level, follower counts) come with the dataset and may
  summarise the whole collection period, including weeks that overlap the evaluation matrix.
  They are user-level, not (user, video) reactions, so they cannot leak a label, but they are
  not strictly "as of" the training cut-off.
- **The re-ranker learns from logged views.** Only part of its gain on them (+0.043) carries
  over to the fully observed test users (+0.017). Exposure bias and a time shift between the
  validation week and the earlier weeks may both contribute; they were not separated.
- **The A/B tests are a replay, not a live experiment.** Each user's reaction to a video was
  recorded once, in a real feed. A session built by a new policy would change what the user saw
  next (position, fatigue, novelty), and a replay cannot capture that. Results describe these
  1,112 heavy users and these 3,327 videos.
- **Cold-start videos are not evaluated.** The video tower can score a video it never saw in
  training (from its categories and length), but every tune/test candidate has training views,
  so that ability is tested on synthetic data only.

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
│   │   ├── features.py      # label-free user and video feature tables
│   │   ├── download.py      # checksummed, atomic download
│   │   ├── prepare.py       # clean, leakage-check, validate, write parquet
│   │   ├── schemas.py       # pandera data contracts
│   │   ├── labels.py        # duration-debiased relevance labels
│   │   ├── split.py         # train/valid/tune/test split + labeling stage
│   │   └── categories.py    # English names for the Chinese category labels
│   ├── tracking.py          # MLflow + W&B behind one interface
│   ├── checkpointing.py     # fingerprints, atomic writes, resumable progress logs
│   ├── evaluation/
│   │   ├── ranking.py       # per-user metrics, bootstrap CIs, paired comparisons
│   │   └── concentration.py # Gini, Lorenz curve, top-k share
│   ├── models/              # random, popularity, category affinity, ALS (Recommender protocol)
│   │   └── two_tower/       # encoding, towers + losses, training, checkpoints, seed ensemble
│   ├── reranking/           # ranking features, LightGBM ranker, two-stage pipeline
│   ├── abtest/              # Welch, CUPED, power, SRM, mSPRT; sessions; simulated experiments
│   ├── experiments/         # searches, leaderboard, retrieval, re-ranker, test evaluation, A/B (DVC stages)
│   └── viz/
│       └── style.py         # chart style with a colorblind-safe palette
├── notebooks/               # jupytext .py sources + executed .ipynb
├── sweeps/                  # W&B sweep configs
├── tests/                   # synthetic fixtures; no network or real data needed
├── reports/                 # git-tracked pipeline metrics and figures
├── models/                  # trained models (DVC outputs, not in git)
└── checkpoints/             # resume state of interrupted training (temporary, not in git)
```

## Roadmap

0. **Setup & data pipeline** (done)
1. **EDA, debiased labels and train/validation split** (done)
2. **Evaluation harness, popularity and ALS baselines, MLflow + W&B tracking** (done)
3. **Two-tower neural model, category-affinity baseline, feature pipeline** (done)
4. **Two-stage pipeline: LightGBM re-ranker with feature engineering and SHAP** (done)
5. **Final test evaluation and A/B testing framework on the fully observed matrix** (done)
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
