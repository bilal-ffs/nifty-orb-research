# NIFTY 50 ORB research baseline

No optimization, additional signal filters, or ML training. Source CSV remains unchanged.

## Data and rules

Source: 975,317 rows; SHA-256 `2f1ff6dd6a23db752097d8d969c009f32fdc04a14103f3fe0c89018ebb76f223`. Coverage: 2015-01-09 09:15:00+05:30 to 2025-07-25 15:29:00+05:30.
Observed regular sessions: 2602; eligible: 2571; excluded: 31. Outside-hours rows excluded only in memory: 612. Zero volume is unavailable and never used as a signal.
IST candle starts 09:15–15:29. Regular hours follow the requested 09:15–15:30 convention ([NSE market timings](https://www.nseindia.com/static/market-data/market-timings)). No external calendar applied; absent whole sessions cannot be detected. All incomplete observed regular sessions are excluded to avoid simulating through missing execution bars. This retrospective quality screen can cause selection bias and is not deployable as a morning filter.
OR uses exactly 09:15–09:29. First completed candle starting 09:30–11:59 closing strictly outside the range creates the sole daily event. Prediction is at its close, one minute after candle start. Entry is the next available open at or before 12:00 (inclusive); complete-session screening makes it the immediate next minute in this run. A gap back inside the range is allowed if actual entry-to-stop R stays positive; nonpositive R skips the event without searching again.
Stop is opposite OR boundary; target is actual adverse entry fill plus/minus 1.5R. Stops take precedence on stop/target collisions. Stop gaps fill at adverse candle open; targets receive no favorable gap improvement. Process entry candle after filling at its open. Remaining positions exit at 15:29 open, before the session end. Intrabar exit timestamps denote the candle containing execution, not an invented exact second; precision and candle-end fields identify uncertainty.
Costs: 5 bps per side on adverse entry/exit notional. Slippage: 1 bps adverse per fill, separately itemized. Gross P&L uses reference entry/exit prices for the same realized path; slippage and costs reconcile to net. It is not a separate zero-slippage backtest, since slippage changes R and target levels.

## Account methodology

Each split starts with INR 1,000,000. Fractional proxy units = initial capital * 0.1 / entry fill; fixed initial-capital allocation per trade, not compounding sizing. Default allocation is 10%, leaving 90% cash initially. Daily equity = initial capital + cumulative net P&L. Daily account return = net P&L / prior observed-session equity; excluded and no-trade observed dates contribute zero. Sharpe uses sample standard deviation, sqrt(252), and zero risk-free return. Drawdown uses closing account equity including initial capital. No CAGR calculated. No intraday drawdown or futures margin/lot sizing modeled. Index prices are not executable futures prices.
An initial diagnostic at 100% fixed initial-capital notional exhausted IS equity, making account-return Sharpe undefined. The reported capital illustration uses 10% fixed allocation consistently in both splits. This is a disclosed account assumption, not an optimized ORB parameter; trade events, fills, point costs and R outcomes are identical across allocations.

## Separate factual results

| Metric | IS 2015–2021 | OOS 2022–last date |
|---|---:|---:|
| number_of_trades | 1624 | 852 |
| long_trades | 759 | 394 |
| short_trades | 865 | 458 |
| win_rate_net | 0.405788 | 0.415493 |
| average_winner_currency | 419.580537 | 347.871290 |
| average_loser_currency | -455.154778 | -418.801522 |
| profit_factor_net | 0.629527 | 0.590452 |
| expectancy_net_r | -0.283201 | -0.298271 |
| gross_pnl | 32139.100470 | 16824.126849 |
| net_pnl | -162720.786975 | -85416.721346 |
| transaction_cost_pnl | 162383.239270 | 85200.706689 |
| slippage_pnl | 32476.648175 | 17040.141506 |
| maximum_drawdown_pct | 0.169352 | 0.086553 |
| maximum_drawdown_currency | 170292.114901 | 86617.563496 |
| sharpe_daily_252_rf0 | -2.705145 | -3.539919 |

Monthly/yearly P&L and compounded account returns are in each metrics JSON and separate performance CSVs. Win rate and average winners/losers use net P&L; target success is a different label.
IS evaluated through 2021-12-31; OOS evaluated through 2025-07-25. Under the stated assumptions, net P&L is negative in IS and negative in OOS. Transaction costs and slippage are explicitly deducted; these results do not justify changing rules using OOS outcomes.

## ML event contract and leakage prevention

One row per first eligible daily breakout; no subsequent events are considered after the first signal. `feature_*` fields and breakout/prediction timestamps are the only predictor inputs. `fill_*`, `target_*`, `outcome_*`, and execution status are separated by prefixes and excluded from predictors. Prediction is at signal-candle completion, before its next-open execution; prices of the entry and future candles are never features. Same clock time for close/next-open is a sequencing convention: observe close, then execute open; real latency is not modeled.
Feature percentages are decimal fractions. OR width percentage uses 09:15 open. Gap uses current 09:15 open versus prior observed regular close. Previous-day return is close-to-close. ATR is a simple mean of 14 historical true ranges; historical volatility is the sample standard deviation of 20 historical daily close returns. Both are shifted by one session. These are ML features only, never trading filters. Incomplete prior regular days leave NaNs and interrupt rolling windows. History flows from IS into OOS as already-known past data, never backward. Evening observations are not part of the regular daily history. No imputation, normalization, scaler fitting, or OOS aggregate features. Early warm-up rows retain missing history.
`target_reached_1_5r` is 1 only for target exits; conservative stop-first collisions are 0. `target_net_return_r` is realized adverse-fill/cost-adjusted points divided by actual entry-stop risk. Skipped fills retain feature rows with null targets; exclude these from future supervised training rather than assigning artificial failures. Labels are known only after exit. Future validation must split chronologically by date, purge any overlapping outcome horizons and fit all preprocessing on training folds only.

## Reproduction

Run `python -m pip install -r requirements.txt`, `python -m unittest discover -s tests -v`, then `python orb_pipeline.py`. Configurable flags: `--capital`, `--notional-fraction`, `--cost-bps`, `--slippage-bps`, `--source`, `--output`. Rules remain fixed; no parameter search. Source hash is checked before/after processing. Environment versions and detailed session exclusions are in the audit JSON.
This is a descriptive index-price baseline, not evidence of executable futures profitability. OOS was evaluated once under the same fixed rules.
