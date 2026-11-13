# NIFTY ORB + Vervoort ATR: Python backtest results

Input period: 2015-01-09 through 2025-07-25. All results before costs, one index unit.
Opening range: 09:15-09:29. Five-minute close signals; ATR(5) x 3.5; target 1.5 x ORB width; one trade/day; 15:00 square-off.

## Minute execution results

Five-minute signals and trailing-stop updates, with minute OHLC used to resolve bracket execution.

| Metric | Long | Short |
|---|---:|---:|
| Trades | 1,105 | 1,061 |
| Net points | 1,340.98 | 4,628.11 |
| Net R | +5.6271 | +53.2619 |
| Average R/trade | +0.00509 | +0.05020 |
| Win rate % | 42.71 | 42.04 |
| Profit factor (points) | 1.0631 | 1.2143 |
| Profit factor (R) | 1.0168 | 1.1585 |
| Maximum closed-trade drawdown R | 22.9481 | 26.3080 |

Shorts have the stronger historical before-cost R expectancy. Longs are close to flat on a risk-normalized basis. This is descriptive performance on the supplied data, not an out-of-sample validation.

## Five-minute execution comparison

| Side | Net R, 5-minute execution | Net R, minute execution |
|---|---:|---:|
| long | 5.9952 | 5.6271 |
| short | 53.1554 | 53.2619 |

Both models use identical entries. Execution changed realized R on 23 trades. No executed bracket bar touched both stop and target in either run. Minute bars still require inferred within-minute ordering and do not provide tick-level certainty.

## Data checks and TradingView comparison

- 975,321 source rows; 2,602 regular-session days; 2,571 eligible days.
- 31 regular-session days excluded for missing minutes through 14:59 or conflicting duplicate timestamps. Four duplicated timestamps occurred across three days. See audit JSON for details.
- The original CSV is unchanged. First duplicate observations are retained for ATR continuity; affected days are excluded from trading.
- On 2,148 overlapping trade days with the original TradingView export, 1,984 entry candle timestamps match; 1,797 entry prices match within 0.10 points. This is not an exact reproduction of that export.
- Source price differences, indicator warm-up/history, and export settings can affect reconciliation. Their individual contributions have not been isolated.
- Original TradingView trade results are not mixed into the new Python backtest. Its 2010-2026 coverage also differs from this input.
- Focused tests passed for Wilder ATR seed, gap fills, within-bar order, entry-bar lookahead prevention, trailing-stop timing, square-off, and trade R.
- Both complete exports passed row-count, one-trade-per-day, finite-value, and R calculation checks.

## Files

- orb_backtest.py: executable strategy translation; default runs both models.
- check_orb_backtest.py: focused validation cases.
- PY_ORB_METHOD.md: rules, data handling, timestamp semantics, and limitations.
- PY_ORB_minute_LONG.csv / SHORT.csv: separate side trades using minute execution.
- PY_ORB_pine5m_LONG.csv / SHORT.csv: separate side trades using five-minute execution.
- Each mode also has ALL, SUMMARY, YEAR, ENTRY_TIME, and ORB_WIDTH CSVs.
- PY_ORB_EXECUTION_COMPARISON.csv: paired execution results.
- PY_ORB_VS_TRADINGVIEW.csv: daily reconciliation with the earlier export.
- PY_ORB_RUN_AUDIT.json: data audit, parameters and results.
- PY_ORB_5MIN_INDICATORS.csv: aggregated candles and indicator values.

Costs, slippage and tick rounding are excluded. Point P&L is not futures lot cash P&L. Net R sums trades at one initial-risk unit each; it is not compounded account return. Entry-window summaries use candle opening labels, five minutes before actual fills. ORB quartiles use the full sample and are exploratory.
