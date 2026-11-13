# ORB directional Logistic V2 — IS only

All 1,624 original IS trades reconcile into 759 long and 865 short trades. Existing pipeline, labels, execution, data eligibility, V1 models and its 1,350 OOF predictions remain unchanged. No OOS files were accessed, no hyperparameters or rules tuned, and no thresholds/filters selected.

## Directional diagnostics

| direction | n | target_hit_rate | gross_expectancy_r | net_expectancy_r | gross_r_ci_low | gross_r_ci_high | net_r_ci_low | net_r_ci_high | gross_profit_factor_currency | net_profit_factor_currency | mean_fees_r | mean_slippage_r | stop_pct | target_pct | session_exit_pct | mean_or_width_points | median_or_width_points | mean_initial_risk_points | total_net_pnl | bootstrap_replicates | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 759 | 0.231884 | -0.031605 | -0.372466 | -0.105101 | 0.047156 | -0.448430 | -0.293183 | 0.988939 | 0.563926 | 0.284051 | 0.056810 | 49.407115 | 23.188406 | 27.404480 | 43.262978 | 34.900000 | 47.641612 | -92821.039414 | 2000 | 20261014 |
| short | 865 | 0.284393 | 0.125577 | -0.204874 | 0.053407 | 0.195395 | -0.277769 | -0.134640 | 1.198873 | 0.691212 | 0.275376 | 0.055075 | 39.884393 | 28.439306 | 31.676301 | 44.219942 | 35.300000 | 49.030364 | -69899.747561 | 2000 | 20261014 |

Gross/net profit factors use original currency sizing; R uses original adverse entry-to-stop risk. Fees and slippage are separated. Stop percentage includes stop gaps. OR width is in points. Confidence intervals use 2,000 whole trading-day cluster percentile bootstrap replicates, fixed seed 20261014 and deterministic streams. There is one trade per day, so day resampling reduces to row resampling for this ledger. This assumes exchangeable days and does not correct serial dependence or retrospective sample selection.
Direct long-minus-short contrasts (joint trading-day bootstrap):

| basis | mean_difference | ci_low | ci_high |
|---|---|---|---|
| gross_r | -0.157182 | -0.263084 | -0.052002 |
| net_r | -0.167592 | -0.273181 | -0.060489 |

The long-minus-short net difference is -0.167592R per trade. This is economically material relative to mean friction of 0.335656R, but both directions have negative net mean R. The direct contrast CI excludes zero and provides evidence of a directional mean difference under the exchangeable-day bootstrap assumptions. It does not establish robustness to serial dependence/regime shifts or justify a profitable directional rule; the conclusion is not based merely on separate CI overlap or point estimates.

## Independent models and validation

Six predictors are the exact V1 feature list minus constant feature_breakout_direction. feature_or_direction is retained because it describes the opening-range candle move and is not constant breakout direction. All are known at completed signal-candle confirmation. Missing historical fields remain excluded; no imputation or new features.
StandardScaler → LogisticRegression(C=1.0, class_weight=None), V1 defaults and fixed max_iter=1000/seed retained. Five chronological expanding event-count folds per direction; each scaler/model fits only the training portion. Training events whose conservative label-end is not strictly before validation start are purged; current intraday events do not overlap, so actual purges and additional embargo are zero. Fold class counts, prevalences, indices and label maturities are saved in metrics. Final models fit all events of their own direction; final models are not used for OOF scoring.

| model | n | positive_prevalence | roc_auc | pr_auc_average_precision | brier_score | log_loss | balanced_accuracy_threshold_0_5 |
|---|---|---|---|---|---|---|---|
| ORB_LONG_LOGISTIC | 630 | 0.220635 | 0.583803 | 0.293486 | 0.172207 | 0.545794 | 0.500000 |
| ORB_SHORT_LOGISTIC | 720 | 0.266667 | 0.518239 | 0.279970 | 0.206611 | 0.608511 | 0.496686 |

| model | n | positive_prevalence | roc_auc | pr_auc_average_precision | brier_score | log_loss | balanced_accuracy_threshold_0_5 |
|---|---|---|---|---|---|---|---|
| ORB_LONG_LOGISTIC_constant_baseline | 630 | 0.220635 | 0.518462 | 0.226981 | 0.173307 | 0.531293 | 0.500000 |
| ORB_SHORT_LOGISTIC_constant_baseline | 720 | 0.266667 | 0.444602 | 0.241405 | 0.200842 | 0.591996 | 0.500000 |

Primary PR-AUC is average precision; trapezoidal PR area is separately saved. Constant baselines use each direction-fold training prevalence, never validation labels. Pooled baseline ROC-AUC can differ from 0.5 because probabilities vary across folds. Confusion matrices use fixed 0.5 only for classifier diagnostics, not trading selection:

- ORB_LONG_LOGISTIC: `[[491, 0], [139, 0]]`; baseline `[[491, 0], [139, 0]]` (true classes as rows; predicted classes as columns).
- ORB_SHORT_LOGISTIC: `[[497, 31], [182, 10]]`; baseline `[[528, 0], [192, 0]]` (true classes as rows; predicted classes as columns).

Fold counts and class balance:

| direction | fold | train_count | train_positive_count | train_negative_count | test_count | test_positive_count | test_negative_count | train_first_date | train_last_date | test_first_date | test_last_date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 1 | 129 | 37 | 92 | 126 | 29 | 97 | 2015-01-12 | 2016-04-22 | 2016-04-27 | 2017-07-11 |
| long | 2 | 255 | 66 | 189 | 126 | 27 | 99 | 2015-01-12 | 2017-07-11 | 2017-07-13 | 2018-08-17 |
| long | 3 | 381 | 93 | 288 | 126 | 31 | 95 | 2015-01-12 | 2018-08-17 | 2018-08-20 | 2019-12-17 |
| long | 4 | 507 | 124 | 383 | 126 | 30 | 96 | 2015-01-12 | 2019-12-17 | 2019-12-18 | 2020-12-23 |
| long | 5 | 633 | 154 | 479 | 126 | 22 | 104 | 2015-01-12 | 2020-12-23 | 2020-12-24 | 2021-12-31 |
| short | 1 | 145 | 54 | 91 | 144 | 28 | 116 | 2015-01-09 | 2016-03-10 | 2016-03-14 | 2017-05-03 |
| short | 2 | 289 | 82 | 207 | 144 | 39 | 105 | 2015-01-09 | 2017-05-03 | 2017-05-04 | 2018-06-13 |
| short | 3 | 433 | 121 | 312 | 144 | 34 | 110 | 2015-01-09 | 2018-06-13 | 2018-06-14 | 2019-07-18 |
| short | 4 | 577 | 155 | 422 | 144 | 43 | 101 | 2015-01-09 | 2019-07-18 | 2019-07-19 | 2020-09-30 |
| short | 5 | 721 | 198 | 523 | 144 | 48 | 96 | 2015-01-09 | 2020-09-30 | 2020-10-09 | 2021-12-24 |

## Fair comparison with unchanged pooled V1

Directional OOF warm-up boundaries differ from V1. Exact paired comparisons use the intersection of stored V1 OOF and each V2 OOF event set, and score both on identical event IDs. V2 events without a V1 OOF prediction cannot be evaluated against unchanged V1 on all V2 events: they are explicitly listed and excluded from both paired scores. No in-sample/final-model prediction is substituted. Differing training populations and refit dates remain methodological differences even after matching test events.

| direction | v2_total_oof | exact_matched_events | unmatched_v2_events |
|---|---|---|---|
| long | 630 | 630 | 0 |
| short | 720 | 714 | 6 |

### Long — exact matched events

| model | n | positive_prevalence | roc_auc | pr_auc_average_precision | brier_score | log_loss | balanced_accuracy_threshold_0_5 |
|---|---|---|---|---|---|---|---|
| v1_same_events | 630 | 0.220635 | 0.594177 | 0.290309 | 0.170669 | 0.533818 | 0.502579 |
| v2_same_events | 630 | 0.220635 | 0.583803 | 0.293486 | 0.172207 | 0.545794 | 0.500000 |

V2 improvement flags (descriptive, not significance tests): `{"roc_auc": false, "pr_auc_average_precision": true, "log_loss": false, "brier_score": false}`.

### Short — exact matched events

| model | n | positive_prevalence | roc_auc | pr_auc_average_precision | brier_score | log_loss | balanced_accuracy_threshold_0_5 |
|---|---|---|---|---|---|---|---|
| v1_same_events | 714 | 0.267507 | 0.519035 | 0.279559 | 0.207871 | 0.616261 | 0.506602 |
| v2_same_events | 714 | 0.267507 | 0.518565 | 0.281185 | 0.206827 | 0.609091 | 0.497497 |

V2 improvement flags (descriptive, not significance tests): `{"roc_auc": false, "pr_auc_average_precision": true, "log_loss": true, "brier_score": true}`.

### Combined — exact matched events

| model | n | positive_prevalence | roc_auc | pr_auc_average_precision | brier_score | log_loss | balanced_accuracy_threshold_0_5 |
|---|---|---|---|---|---|---|---|
| v1_same_events | 1344 | 0.245536 | 0.556004 | 0.276445 | 0.190433 | 0.577616 | 0.505747 |
| v2_same_events | 1344 | 0.245536 | 0.558855 | 0.278118 | 0.190599 | 0.579420 | 0.500359 |

V2 improvement flags (descriptive, not significance tests): `{"roc_auc": true, "pr_auc_average_precision": true, "log_loss": false, "brier_score": false}`.

## Probability ranking and realized net R

Quintiles use only held-out probabilities, never labels to define boundaries. Full-direction and identical-event V1/V2 quintile comparisons are in directional_probability_quintiles.csv; features remain prediction-time only, realized outcomes are descriptive labels. Quantile boundaries are retrospective OOF summaries, not executable thresholds. Within a paired sample models can assign different events to their own quintiles: this compares ranking, not identical quintile constituents.

| model | sample | quintile | n | lower_probability | upper_probability | boundary | mean_probability | actual_target_rate | mean_gross_r | mean_net_r | mean_fees_r | mean_slippage_r | mean_costs_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ORB_LONG_LOGISTIC | full_direction_oof | 1 | 126 | 0.000106 | 0.104496 | [lower,upper] | 0.062763 | 0.150794 | -0.084954 | -0.287596 | 0.168868 | 0.033774 | 0.202642 |
| ORB_LONG_LOGISTIC | full_direction_oof | 2 | 126 | 0.104496 | 0.159774 | (lower,upper] | 0.134652 | 0.222222 | 0.018032 | -0.286687 | 0.253932 | 0.050786 | 0.304718 |
| ORB_LONG_LOGISTIC | full_direction_oof | 3 | 126 | 0.159774 | 0.221176 | (lower,upper] | 0.190030 | 0.190476 | -0.168400 | -0.500817 | 0.277014 | 0.055403 | 0.332417 |
| ORB_LONG_LOGISTIC | full_direction_oof | 4 | 126 | 0.221176 | 0.287675 | (lower,upper] | 0.256082 | 0.246032 | -0.017318 | -0.433808 | 0.347075 | 0.069415 | 0.416490 |
| ORB_LONG_LOGISTIC | full_direction_oof | 5 | 126 | 0.287675 | 0.452728 | (lower,upper] | 0.343677 | 0.293651 | -0.024613 | -0.504086 | 0.399560 | 0.079912 | 0.479473 |
| ORB_SHORT_LOGISTIC | full_direction_oof | 1 | 144 | 0.000000 | 0.179213 | [lower,upper] | 0.119330 | 0.215278 | 0.078320 | -0.090473 | 0.140661 | 0.028132 | 0.168793 |
| ORB_SHORT_LOGISTIC | full_direction_oof | 2 | 144 | 0.179213 | 0.241284 | (lower,upper] | 0.211270 | 0.256944 | 0.070754 | -0.192135 | 0.219074 | 0.043815 | 0.262889 |
| ORB_SHORT_LOGISTIC | full_direction_oof | 3 | 144 | 0.241284 | 0.296192 | (lower,upper] | 0.269403 | 0.298611 | 0.109272 | -0.257058 | 0.305275 | 0.061055 | 0.366330 |
| ORB_SHORT_LOGISTIC | full_direction_oof | 4 | 144 | 0.296192 | 0.358503 | (lower,upper] | 0.322944 | 0.312500 | 0.161710 | -0.271162 | 0.360726 | 0.072145 | 0.432872 |
| ORB_SHORT_LOGISTIC | full_direction_oof | 5 | 144 | 0.358503 | 0.630331 | (lower,upper] | 0.449106 | 0.250000 | 0.009934 | -0.438866 | 0.374000 | 0.074800 | 0.448800 |

Paired ranking summaries (Spearman rank correlations of probability with target/net R and top-minus-bottom quintile net R):

| direction | model | n | probability_target_spearman | probability_net_r_spearman | lowest_quintile_net_r | highest_quintile_net_r | top_minus_bottom_quintile_net_r |
|---|---|---|---|---|---|---|---|
| long | POOLED_V1 | 630 | 0.135283 | -0.241333 | -0.287960 | -0.443484 | -0.155524 |
| long | ORB_LONG_LOGISTIC | 630 | 0.120382 | -0.218448 | -0.287596 | -0.504086 | -0.216490 |
| short | POOLED_V1 | 714 | 0.029189 | -0.240165 | -0.104231 | -0.553873 | -0.449642 |
| short | ORB_SHORT_LOGISTIC | 714 | 0.028468 | -0.238033 | -0.091456 | -0.434164 | -0.342708 |

These are descriptive, not formal model comparison tests; nonmonotonic bins and binary target success do not establish cost-adjusted profitability. No capital-performance backtest of a classifier is performed.

## Diagnostic conclusions

On exact matched long events, V2 worsens ROC-AUC, log loss and Brier score versus pooled V1, with a small average-precision improvement. On matched short events, ROC-AUC is nearly unchanged/slightly worse, average precision slightly improves, and log loss/Brier improve modestly. Combined matched-event log loss/Brier are slightly worse. These point estimates do not establish a general advantage from splitting.
Against each directional training-prevalence baseline, both models improve ROC-AUC and average precision but both have worse log loss. Long Brier is slightly better than its baseline; short Brier is worse. Thus directional specialization does not consistently improve probability forecasting.
Probability-to-net-R Spearman correlations remain negative for both directions and models. The highest-minus-lowest probability quintile net R is negative throughout: it becomes more negative for long V2 and less negative for short V2. This is no demonstrated positive net-R ranking relationship or executable strategy advantage. No thresholds or trading filters are selected.

## Reproduction

`python orb_logistic_directional_v2/run.py` then `python -m unittest discover -s orb_logistic_directional_v2 -p "test_*.py" -v`. Read-only V1 functions are reused through importlib without running V1 main. input_manifest.json records protected input hashes, feature timing proof, versions and settings. Fold models, final scaler/classifier pipelines, folds, OOF predictions and all requested diagnostics are saved under this new directory. Existing file hashes are checked before/after.
