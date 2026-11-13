# NIFTY 50 ORB Logistic Regression V1 — IS only

1624 events; 1350 chronological OOF predictions; 274 initial events used only as training warm-up. Final model fits all IS events. No OOS file opened or predicted; existing project files remain unchanged.

Target: 1 for the original 1.5R target exit before stop/session exit, otherwise 0. Verified against pipeline target construction, IS ledger exit category, target_hit and entry + direction × 1.5 × actual risk. Net returns are used only for retrospective probability-group descriptions.
Prediction is at completed breakout-candle confirmation, one minute after its start, before next-open execution under the original sequencing convention. Original full-session quality screening is retained and introduces retrospective sample-selection limitations.

## Features and model

- `feature_or_width_points`: 09:15–09:29 candle highs/lows, completed by 09:30.
- `feature_or_width_pct`: OR width / 09:15 open; known by 09:30.
- `feature_or_direction`: Sign of 09:29 close minus 09:15 open; known by 09:30.
- `feature_or_return`: 09:29 close / 09:15 open − 1; known by 09:30.
- `feature_breakout_direction`: Completed signal close strictly above/below OR boundary.
- `feature_breakout_minutes_since_0915`: Signal start minute + 1 − 09:15, derived clock feature at confirmation.
- `feature_breakout_distance_points`: Completed signal close beyond the relevant OR boundary.

Five historical fields with missing records are excluded as entire columns: prior ATR-relative OR width, gap, previous-day return/range, and historical volatility. History-availability flag is excluded. All 1,624 rows remain; no imputation. Selection is structural and is not based on labels or subgroup profitability. Raw dates/timestamps, fill-dependent fields, target labels, realized R and exits never enter X.
Pipeline: StandardScaler → LogisticRegression. C=1.0, class_weight=None, default solver/regularization/tolerance; max_iter=1000 is a fixed convergence cap; random_state=20261014. No hyperparameter, threshold, feature-performance, calibration or profitability optimization. Each fold scaler and estimator fit only that fold training data. Final scaler/model fit all IS separately; the final model produces no evaluation predictions.

## Chronology, information windows and baseline

Five expanding TimeSeriesSplit folds use contiguous, equal-sized event-count test blocks; dates are irregular trading events, so folds do not have equal calendar duration. Training precedes validation. Training labels whose conservative exit-candle-end timestamp is not strictly before first validation confirmation are purged. Original events end intraday and there is at most one event per day: zero actual overlaps/purges, hence zero additional embargo days. Shared backward-looking historical observations alone are not future-label overlap. No random split.
| fold | train_count | test_count | train_first_date | train_last_date | test_first_date | test_last_date | training_prevalence |
|---|---|---|---|---|---|---|---|
| 1 | 274 | 270 | 2015-01-09 | 2016-03-29 | 2016-03-30 | 2017-06-05 | 0.335766 |
| 2 | 544 | 270 | 2015-01-09 | 2017-06-05 | 2017-06-06 | 2018-07-13 | 0.266544 |
| 3 | 814 | 270 | 2015-01-09 | 2018-07-13 | 2018-07-16 | 2019-09-12 | 0.264128 |
| 4 | 1084 | 270 | 2015-01-09 | 2019-09-12 | 2019-09-13 | 2020-11-11 | 0.257380 |
| 5 | 1354 | 270 | 2015-01-09 | 2020-11-11 | 2020-11-12 | 2021-12-31 | 0.261448 |

The comparison baseline predicts only each fold training-label prevalence, fixed throughout that fold test block. Its probabilities vary between folds, so its pooled ROC-AUC need not equal 0.5. Per-fold baseline ROC-AUC is 0.5; PR average precision equals that fold test prevalence. Fold-specific and pooled metrics are saved. Primary PR-AUC is average precision (step-weighted precision–recall area); trapezoidal PR area is separately labeled and can differ, especially for tied constant predictions.

## OOF results

| metric | logistic | constant_baseline |
|---|---|---|
| positive_prevalence | 0.244444 | 0.244444 |
| roc_auc | 0.553948 | 0.466310 |
| pr_auc_average_precision | 0.274273 | 0.228825 |
| pr_auc_trapezoidal | 0.271635 | 0.288239 |
| log_loss | 0.576749 | 0.564308 |
| brier_score | 0.190037 | 0.188089 |
| accuracy_threshold_0_5 | 0.750370 | 0.755556 |
| balanced_accuracy_threshold_0_5 | 0.505793 | 0.500000 |

Model confusion matrix (rows=true 0,1; columns=predicted 0,1): `[[1004, 16], [321, 9]]`. Baseline: `[[1020, 0], [330, 0]]`. Threshold is fixed at 0.5; high accuracy with the majority negative class is not evidence of useful target detection.
Pooled OOF model beats baseline on: ROC-AUC=True, PR average precision=True, log loss=False, Brier=False. These are descriptive point estimates, not significance claims or external generalization evidence.

The model improves pooled discrimination but does not beat the training-prevalence baseline on either log loss or Brier score. There is no demonstrated overall probability-forecast advantage.

Calibration by OOF probability decile is in calibration_deciles.csv for the model and baseline_calibration.csv for the baseline. Ties remain together and may reduce the number of bins; empty intervals are omitted. Boundaries, counts, mean predicted probability and observed target rate are saved. No calibrator was trained.

## Descriptive net R ranking

| group | probability_lower | probability_upper | boundary | n | mean_probability | observed_target_rate | mean_realized_net_r |
|---|---|---|---|---|---|---|---|
| 1 | 0.000000 | 0.142043 | [lower,upper] | 270 | 0.091371 | 0.159259 | -0.202566 |
| 2 | 0.142043 | 0.193340 | (lower,upper] | 270 | 0.167853 | 0.270370 | -0.210957 |
| 3 | 0.193340 | 0.246278 | (lower,upper] | 270 | 0.219073 | 0.233333 | -0.366226 |
| 4 | 0.246278 | 0.310976 | (lower,upper] | 270 | 0.274575 | 0.307407 | -0.339240 |
| 5 | 0.310976 | 0.605234 | (lower,upper] | 270 | 0.392552 | 0.251852 | -0.488370 |

These quintiles use only held-out probabilities and retrospective realized net R. Boundaries are pooled IS OOF quantiles, not live entry thresholds. Target likelihood and net R are different objectives: the binary target excludes profitable session exits and cost/R varies. These groups neither constitute a backtest of a traded classifier nor select filters, sizing or thresholds; no executable performance claim is made.

All 5 probability groups have negative descriptive mean net R. Lowest-probability quintile: -0.202566R; highest-probability quintile: -0.488370R. Observed target frequencies and net R need not be monotonic in estimated target probability.

## Reproduction

From the project directory: `python orb_logistic_v1/train.py`, then `python -m unittest discover -s orb_logistic_v1 -p "test_*.py" -v`. Version pins are in requirements.txt. final_model.joblib includes scaler and classifier; fold models permit checking held-out prediction/scaler provenance. Load with the saved feature order; future missing values require a separately reviewed data contract. No future evaluation is performed here.
Feature list, exclusions/timing evidence, verification snippets, input hashes, environment, warm-up IDs, fold indices, training prevalences, metrics and OOF outputs are saved. Existing input hashes are checked before/after execution.

Official references: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
