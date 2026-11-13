# Data provenance, source history and third-party notices

## Local source projects

This publication copy consolidates `ORB_NIFTY_50` with selected source code and
reports from `ORB strategy nifty50`. Originals were retained unchanged. The
original-source manifest records pre-publication SHA-256 values and relative
paths. Publication support files and opt-in test decorators are subsequent edits.
The manifest does not claim that edited publication files have the original hash.

## Market data

The historical baseline report records 975,317 cleaned rows, covering
2015-01-09 09:15 through 2025-07-25 15:29 IST. Its SHA-256 is
`2f1ff6dd6a23db752097d8d969c009f32fdc04a14103f3fe0c89018ebb76f223`.
The older translation records 975,321 source rows, before duplicate cleaning.
Those sources are not interchangeable for exact historical reconciliation.

The original dataset URL, release/version, acquisition date and redistribution
license have not been verified. Raw data, duplicate backups, bar-level audits,
feature rows, trade ledgers and serialized estimators are deliberately absent.
Only source code, explanatory reports, charts and selected aggregate tables are
included. Dataset-specific reports do not grant rights to the underlying data.
No claim is made that the included summaries constitute an independent replication.

A local input has columns `date,open,high,low,close,volume`. Dates are minute
OPEN labels formatted `DD-MM-YYYY HH:MM`, interpreted as Asia/Kolkata. Zero
index volume is unavailable information. Do not forward-fill missing prices.

## Rule attribution

The Vervoort recurrence came from a supplied reference rule. The available
project record does not establish its exact original URL, author attribution,
version, or applicable source-code license. No third-party Pine source or
TradingView trade export is bundled. Before claiming exact source parity or
adding upstream source, record a verified citation/version and comply with that
source's license. The Python reports state the implemented recurrence explicitly.
The MIT license covers original project work only and grants no rights to data
or third-party materials.

## Software and public references

NumPy, pandas, scikit-learn, joblib and Matplotlib remain under their respective
licenses and are installed as dependencies, not vendored here. Useful primary
references for the implementation are:

- NSE market hours: https://www.nseindia.com/static/market-data/market-timings
- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- TradingView strategy execution concepts: https://www.tradingview.com/pine-script-docs/concepts/strategies/

These are reference links retained for readers; they do not verify the origin
or licensing of the specific supplied historical data or rule.
