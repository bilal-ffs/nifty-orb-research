# Research methodology and interpretation

## Baseline

The opening range uses exactly 09:15-09:29 IST minute starts. The first completed
09:30-11:59 candle closing strictly outside the range creates one daily event.
Its close is observed before next-open execution; final permitted entry is
12:00. Entry/close boundary equality denotes sequencing, not measured latency.
Stop is the opposite range boundary; target is 1.5 times actual adverse-fill
entry-to-stop risk. Stops win collisions; stop gaps fill at adverse open.
Remaining positions exit at 15:29 open. Only one daily event is attempted.

Historical features use shifted prior observations. Outcome, fill and predictor
columns have separate contracts. Tests mutate future observations to check that
prediction-time values and prior outputs remain unchanged.

Each period resets capital independently; the default is fixed 10% initial
notional, fractional units, with no compounding sizing. Daily account returns,
R outcomes and trade cash P&L are different quantities. The reports disclose
which denominator is used. Fees and adverse slippage reconcile separately.

## Chronological ML

Pooled V1 and directional V2 fit StandardScaler and untuned LogisticRegression
inside each training fold. Five expanding folds precede their validation events.
Any training label ending at or after validation start is purged. The baseline
predicts training-fold prevalence. OOF predictions, not final-model in-sample
predictions, are evaluated. V2 comparisons use identical matched events.
Classification target success is not equivalent to positive net trading P&L.
Probability quintiles are retrospective descriptions, not executable thresholds.

The cost-adjusted Vervoort experiment uses common calendar validation windows,
its own labels and ten fully reported multipliers. Comparisons are not corrected
for multiple testing. No best multiplier is presented as a validated winner.

## Data quality and execution limits

Duplicate/invalid/incomplete sessions are screened; zero volume is unavailable
information. Screening an entire day's completeness uses later information and
can bias the sample. Absent whole sessions are not detected without a verified
calendar. Candle prices require within-bar execution assumptions.

Index prices do not establish futures executability. Real broker fills, funding,
rolls, basis, capacity, margin and lots are unmodeled. Closing-equity drawdown
is not intraday maximum drawdown. A flat or small fixed allocation affects
account risk metrics; it does not make the underlying trade expectancy positive.

## Research selection and holdout status

The baseline's saved 2022-2025 test is kept separate from 2015-2021 development.
However, the legacy study explores 2015-2025, including those later dates.
Consolidation creates no new independent evidence. Any strategy proposed after
reading this repository needs fresh validation. Positive before-cost legacy
results cannot override negative cost-adjusted results under different rules.

Original reports preserve the terminology and historical claims of their runs.
This document supplies the repository-wide interpretation and takes precedence
for claims about untouched holdouts, publication contents and current validation.
