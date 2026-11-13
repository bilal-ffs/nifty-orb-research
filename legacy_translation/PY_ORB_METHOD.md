# Python implementation of NIFTY ORB + Vervoort ATR

Run `python orb_backtest.py` from this folder. Requires Python, pandas and numpy.
Use `--mode pine5m` or `--mode minute` to run one execution model. Default: both.
Run `python check_orb_backtest.py` for focused execution and ATR checks.

## Rules translated from the supplied Pine Script

- Assume source timestamps are minute candle opening times in Asia/Kolkata.
- Restrict input to the regular 09:15–15:29 session and aggregate to 5-minute
  candles labeled by their opening time.
- ORB high/low use 09:15 through 09:29 inclusive. No entries during this range.
- At confirmed 5-minute closes, enter long if close > ORB high and Vervoort
  stop < close; enter short if close < ORB low and Vervoort stop > close.
  These are persistent conditions, not a requirement for a fresh crossover.
- Entry candle opening times must be >= 09:30 and < 12:30, matching Pine's
  session test. Therefore actual entry fills range from 09:35 to 12:30.
- Fill entry at that 5-minute close (`process_orders_on_close=true`). Do not
  apply any earlier price movement within the entry candle to the new trade.
- Allow one trade total per day, either long or short. Separate side files are
  splits of this shared strategy, not independently rerun long-only/short-only
  strategies. Pyramiding is disabled.
- ATR(5) uses Wilder smoothing, seeded with the first five true ranges.
  Vervoort ATR multiplier is 3.5. Indicator state carries across days, including
  regular-session observations on days excluded from trading.
- Initial stop is the Vervoort stop calculated on the entry candle. Target is
  entry +/- 1.5 * ORB width. Target stays fixed. This is not a fixed 1.5R target.
- Update trailing stops only at subsequent confirmed 5-minute closes, never
  loosening the stop. A new stop cannot act on earlier prices from that bar.
- Process active bracket orders during the 14:55 bar, then close anything
  still open at its 15:00 close, canceling outstanding orders.
- One unit, no commissions, slippage, taxes, spreads, or capital/margin limits.
  Output P&L is NIFTY points, not rupee profit for a particular futures lot.
  Order prices are retained at computed precision without tick rounding.

## Execution models

`pine5m` uses five-minute OHLC for bracket execution. If an order is crossed at
the next opening price, fill at the open. If both stop and target are reachable
within a candle, assume the extreme closer to the open happens first; ties use
low first. This approximates TradingView's default broker emulator.

`minute` uses exactly the same five-minute signals and stop updates but walks
the constituent minute OHLC candles for bracket fills. Within a minute it still
uses the same inferred OHLC path. This adds price resolution but is not tick
data or a guarantee of obtainable live fills. It does not enable minute-level
ATR recalculation or earlier signal entries.

TradingView's documented execution assumptions:
https://www.tradingview.com/pine-script-docs/concepts/strategies/#broker-emulator

## Data handling and limitations

Do not overwrite or modify the source CSV. Sort timestamps. For duplicated
timestamps retain the first observation for indicator continuity and exclude
the entire affected day from trading. List the conflicting observations in
`PY_ORB_RUN_AUDIT.json`. Retaining the first observation may affect subsequent
ATR state; it is an explicit source-data assumption, not a repair of the feed.

Exclude trading days missing any regular minute between 09:15 and 14:59.
Do not forward-fill OHLC or invent square-off prices. Keep available regular
bars in ATR history; incomplete five-minute bars and excluded days are reported
in the audit. Special sessions outside normal hours are not traded.

ATR begins at the start of this dataset; the first bars differ from an indicator
initialized on an earlier history. The earlier TradingView export covers a
different date range, and its prices/signals need not equal this data source.
The results are a backtest on the supplied feed, not a certified exact match to
TradingView. No position sizing or risk-based compounding is simulated.

## Output definitions

- Initial risk = abs(entry - initial stop), in index points.
- Trade R = signed realized P&L points / initial risk points.
- Net R = sum of trade R, before costs. Avg R = net R / trade count.
- Point profit factor = positive P&L points / absolute negative P&L points.
- R profit factor = positive trade R / absolute negative trade R.
- Max drawdown R uses cumulative closed-trade R, starting at zero, in time
  order; it does not include unrealized intratrade drawdowns.
- Entry and square-off times are actual five-minute closing timestamps.
  Stop/target exit times identify the containing execution candle's opening
  timestamp; their exact intra-candle execution time is unknown. Consequently
  holding minutes are approximate for bracket exits.
- Entry-time summaries use entry candle OPEN labels to match the previous
  TradingView CSV analysis. Actual entry fills occur five minutes later.
- ORB-width quartiles are computed separately by side on the entire sample.
  They describe this sample and are not validated trading filters. Raw points
  are not comparable volatility units across index levels; width_pct is also
  retained in each trade row for later normalized analysis.

For each model, ALL, LONG and SHORT contain one row per closed trade, with
initial stop, risk, ORB width, fixed target, final stop, R and exit reason.
SUMMARY, YEAR, ENTRY_TIME and ORB_WIDTH contain grouped comparisons.
The 5MIN_INDICATORS file supports spot-checking OHLC, ATR and Vervoort levels.
