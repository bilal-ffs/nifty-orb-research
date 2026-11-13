"""Reproducible index-price ORB research; no model fitting or source mutation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    capital: float = 1_000_000.0
    cost_bps: float = 5.0
    slippage_bps: float = 1.0
    target_r: float = 1.5
    notional_fraction: float = 0.1


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_and_audit(path):
    raw = pd.read_csv(path)
    expected = ['date', 'open', 'high', 'low', 'close', 'volume']
    if raw.columns.tolist() != expected:
        raise ValueError('Unexpected CSV schema')
    audit = {'source_sha256': sha256(path), 'source_rows': len(raw),
             'columns': expected, 'source_dtypes': raw.dtypes.astype(str).to_dict(),
             'date_format': '%d-%m-%Y %H:%M', 'timezone': 'Asia/Kolkata',
             'timestamp_convention': 'one-minute candle start; close known one minute later',
             'regular_candle_starts': '09:15 through 15:29 inclusive',
             'session_policy': 'Exclude entire observed regular session if any minute missing, duplicate, or invalid OHLCV. Retrospective data-quality exclusion, not a live signal filter.',
             'sessions': []}
    frame = raw.copy()
    frame['timestamp'] = pd.to_datetime(frame.date, format=audit['date_format'], errors='coerce').dt.tz_localize('Asia/Kolkata')
    numeric = expected[1:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='coerce')
    invalid = (~np.isfinite(frame[numeric]).all(axis=1) |
               (frame[['open', 'high', 'low', 'close']] <= 0).any(axis=1) |
               (frame.volume < 0) |
               (frame.high < frame[['open', 'low', 'close']].max(axis=1)) |
               (frame.low > frame[['open', 'high', 'close']].min(axis=1)))
    frame['invalid'] = invalid
    audit['invalid_timestamp_rows'] = (np.flatnonzero(frame.timestamp.isna()) + 2).tolist()
    audit['invalid_ohlcv_source_lines'] = (np.flatnonzero(invalid) + 2).tolist()
    audit['residual_duplicate_rows'] = int(frame.timestamp.dropna().duplicated(keep=False).sum())
    audit['zero_volume_as_unavailable_rows'] = int((frame.volume == 0).sum())
    frame.loc[frame.volume == 0, 'volume'] = np.nan
    minute = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
    regular = frame[minute.between(555, 929)].copy()
    outside = frame[frame.timestamp.notna() & ~minute.between(555, 929)]
    audit['outside_regular_rows_excluded'] = len(outside)
    audit['outside_regular_by_date'] = outside.groupby(outside.timestamp.dt.strftime('%Y-%m-%d')).size().to_dict()
    regular['session'] = regular.timestamp.dt.strftime('%Y-%m-%d')
    sessions = {}
    daily = []
    for date, g in regular.groupby('session', sort=True):
        g = g.sort_values('timestamp', kind='stable').reset_index(drop=True)
        wanted = pd.date_range(date + ' 09:15', periods=375, freq='min', tz='Asia/Kolkata')
        missing = wanted.difference(pd.DatetimeIndex(g.timestamp))
        missing_or = wanted[:15].difference(pd.DatetimeIndex(g.timestamp))
        duplicate = int(g.timestamp.duplicated(keep=False).sum())
        reasons = []
        if len(missing_or): reasons.append('incomplete_opening_range')
        if len(missing): reasons.append('missing_regular_minutes')
        if duplicate: reasons.append('duplicate_timestamp')
        if g.invalid.any(): reasons.append('invalid_ohlcv')
        eligible = not reasons
        audit['sessions'].append({'date': date, 'rows': len(g), 'eligible': eligible,
                                  'exclusion_reasons': reasons, 'missing_minutes': missing.strftime('%H:%M').tolist(),
                                  'duplicate_rows': duplicate, 'invalid_rows': int(g.invalid.sum())})
        sessions[date] = g if eligible else None
        daily.append({'date': date, 'open': g.open.iloc[0] if eligible else np.nan,
                      'high': g.high.max() if eligible else np.nan,
                      'low': g.low.min() if eligible else np.nan,
                      'close': g.close.iloc[-1] if eligible else np.nan})
    audit['eligible_sessions'] = sum(g is not None for g in sessions.values())
    audit['excluded_sessions'] = len(sessions) - audit['eligible_sessions']
    audit['coverage_start'] = str(frame.timestamp.min())
    audit['coverage_end'] = str(frame.timestamp.max())
    audit['calendar_limit'] = 'Only observed regular-session dates audited; absent whole trading days cannot be identified without an exchange calendar. Weekend special sessions retained when complete.'
    return sessions, pd.DataFrame(daily).set_index('date'), audit


def historical_features(daily):
    """Every historical statistic is shifted before joining the event date."""
    prev = daily.close.shift(1)
    returns = daily.close / prev - 1
    tr = pd.concat([daily.high - daily.low, (daily.high - prev).abs(),
                    (daily.low - prev).abs()], axis=1).max(axis=1, skipna=False)
    # First observed session has no previous close: use its own range for TR.
    if len(tr): tr.iloc[0] = daily.high.iloc[0] - daily.low.iloc[0]
    return pd.DataFrame({'prev_close': prev, 'prev_day_return': returns.shift(1),
                         'prev_day_range': (daily.high - daily.low).shift(1),
                         'prev_atr14': tr.rolling(14, min_periods=14).mean().shift(1),
                         'historical_vol20': returns.rolling(20, min_periods=20).std(ddof=1).shift(1)})


def first_event(g, history):
    opening = g[g.timestamp.dt.hour.mul(60).add(g.timestamp.dt.minute).between(555, 569)]
    if len(opening) != 15 or opening.timestamp.nunique() != 15:
        return None
    hi, lo = float(opening.high.max()), float(opening.low.min())
    for row in g.itertuples():
        i = row.Index
        minute = row.timestamp.hour * 60 + row.timestamp.minute
        # 11:59 close can execute at 12:00; 12:00 close is too late.
        if not 570 <= minute <= 719: continue
        direction = 1 if row.close > hi else -1 if row.close < lo else 0
        if not direction: continue
        prediction = row.timestamp + pd.Timedelta(minutes=1)
        width = hi - lo
        event = {'event_id': row.timestamp.strftime('%Y-%m-%d'),
                 'prediction_timestamp': prediction.isoformat(),
                 'signal_candle_start': row.timestamp.isoformat(),
                 'feature_or_width_points': width,
                 'feature_or_width_pct': width / opening.open.iloc[0],
                 'feature_or_width_prev_atr14': width / history.prev_atr14 if pd.notna(history.prev_atr14) and history.prev_atr14 > 0 else np.nan,
                 'feature_gap_pct': opening.open.iloc[0] / history.prev_close - 1,
                 'feature_prev_day_return': history.prev_day_return,
                 'feature_prev_day_range_points': history.prev_day_range,
                 'feature_or_direction': int(np.sign(opening.close.iloc[-1] - opening.open.iloc[0])),
                 'feature_or_return': opening.close.iloc[-1] / opening.open.iloc[0] - 1,
                 'feature_breakout_direction': direction,
                 'feature_breakout_minutes_since_0915': minute + 1 - 555,
                 'feature_breakout_distance_points': row.close - hi if direction == 1 else lo - row.close,
                 'feature_historical_vol20': history.historical_vol20,
                 'feature_history_available': bool(history.notna().all())}
        return i, direction, hi, lo, event
    return None


def simulate_trade(g, signal_i, direction, hi, lo, config):
    prediction = g.iloc[signal_i].timestamp + pd.Timedelta(minutes=1)
    candidates = g.index[g.timestamp >= prediction]
    if not len(candidates): return None, 'no_next_available_open'
    i = int(candidates[0])
    if g.iloc[i].timestamp.hour * 60 + g.iloc[i].timestamp.minute > 720:
        return None, 'entry_after_1200'
    base_entry = float(g.iloc[i].open)
    entry = base_entry * (1 + direction * config.slippage_bps / 10000)
    stop = lo if direction == 1 else hi
    risk = direction * (entry - stop)
    if risk <= 0: return None, 'nonpositive_entry_risk'
    target = entry + direction * config.target_r * risk
    for row in g.iloc[i:].itertuples():
        # Liquidate at 15:29 open, before the last candle develops.
        if row.timestamp.hour * 60 + row.timestamp.minute >= 929:
            base_exit, reason, precision = float(row.open), 'session_exit', 'exact_open'
        elif (direction == 1 and row.open <= stop) or (direction == -1 and row.open >= stop):
            base_exit, reason, precision = float(row.open), 'stop_gap', 'exact_open'
        else:
            stop_hit = row.low <= stop if direction == 1 else row.high >= stop
            target_hit = row.high >= target if direction == 1 else row.low <= target
            if stop_hit:
                base_exit, reason, precision = stop, 'stop', 'within_candle_unknown'
            elif target_hit:
                # No favorable gap improvement: target-level fill.
                base_exit, reason, precision = target, 'target', 'within_candle_unknown'
            else: continue
        exit_price = base_exit * (1 - direction * config.slippage_bps / 10000)
        units = config.capital * config.notional_fraction / entry
        gross_points = direction * (base_exit - base_entry)
        slipped_points = direction * (exit_price - entry)
        cost_points = (entry + exit_price) * config.cost_bps / 10000
        return {'date': row.timestamp.strftime('%Y-%m-%d'), 'direction': 'long' if direction == 1 else 'short',
                'prediction_timestamp': prediction.isoformat(), 'signal_candle_start': g.iloc[signal_i].timestamp.isoformat(),
                'entry_timestamp': g.iloc[i].timestamp.isoformat(), 'entry_reference_open': base_entry,
                'entry_price': entry, 'stop_price': stop, 'target_price': target, 'risk_points': risk,
                'exit_timestamp': row.timestamp.isoformat(), 'exit_timestamp_precision': precision,
                'exit_candle_end': (row.timestamp + pd.Timedelta(minutes=1)).isoformat(),
                'exit_reference_price': base_exit, 'exit_price': exit_price, 'exit_reason': reason,
                'units': units, 'gross_points': gross_points, 'slippage_points': gross_points - slipped_points,
                'transaction_cost_points': cost_points, 'net_points': slipped_points - cost_points,
                'gross_pnl': units * gross_points, 'slippage_pnl': units * (gross_points - slipped_points),
                'transaction_cost_pnl': units * cost_points, 'net_pnl': units * (slipped_points - cost_points),
                'gross_r': gross_points / risk, 'net_r': (slipped_points - cost_points) / risk,
                'target_hit': int(reason == 'target')}, None
    return None, 'missing_session_exit'


TRADE_COLUMNS = ['date','direction','prediction_timestamp','signal_candle_start','entry_timestamp','entry_reference_open','entry_price','stop_price','target_price','risk_points','exit_timestamp','exit_timestamp_precision','exit_candle_end','exit_reference_price','exit_price','exit_reason','units','gross_points','slippage_points','transaction_cost_points','net_points','gross_pnl','slippage_pnl','transaction_cost_pnl','net_pnl','gross_r','net_r','target_hit']


def run_split(sessions, history, start, end, config):
    trades, events, dates, skips = [], [], [], []
    for date, g in sessions.items():
        if not start <= date <= end: continue
        dates.append(date)  # excluded observed dates remain flat in account series
        if g is None: continue
        event = first_event(g, history.loc[date])
        if event is None: continue
        i, direction, hi, lo, features = event
        trade, reason = simulate_trade(g, i, direction, hi, lo, config)
        features['execution_status'] = reason or 'filled'
        if trade:
            trades.append(trade)
            features.update({'fill_entry_timestamp': trade['entry_timestamp'], 'fill_entry_price': trade['entry_price'],
                             'fill_risk_points': trade['risk_points'], 'target_reached_1_5r': trade['target_hit'],
                             'target_net_return_r': trade['net_r'], 'outcome_exit_reason': trade['exit_reason']})
        else:
            skips.append({'date': date, 'reason': reason})
            features.update(dict.fromkeys(['fill_entry_timestamp','fill_entry_price','fill_risk_points','target_reached_1_5r','target_net_return_r','outcome_exit_reason']))
        events.append(features)
    return pd.DataFrame(trades, columns=TRADE_COLUMNS), pd.DataFrame(events), dates, skips


def account_metrics(trades, dates, config):
    daily = pd.DataFrame(index=pd.to_datetime(dates))
    daily.index.name = 'date'
    for col in ['gross_pnl','net_pnl','transaction_cost_pnl','slippage_pnl']:
        values = trades.groupby('date')[col].sum() if len(trades) else pd.Series(dtype=float)
        values.index = pd.to_datetime(values.index)
        daily[col] = values.reindex(daily.index, fill_value=0).astype(float)
    daily['equity'] = config.capital + daily.net_pnl.cumsum()
    prior = daily.equity.shift(1, fill_value=config.capital)
    daily['account_return'] = daily.net_pnl / prior
    peak = daily.equity.cummax().clip(lower=config.capital)
    daily['drawdown_pct'] = daily.equity / peak - 1
    daily['drawdown_currency'] = daily.equity - peak
    wins = trades.loc[trades.net_pnl > 0] if len(trades) else trades
    losses = trades.loc[trades.net_pnl < 0] if len(trades) else trades
    sigma = daily.account_return.std(ddof=1)
    def aggregates(freq):
        result = []
        for key, g in daily.groupby(daily.index.to_period(freq)):
            result.append({'period': str(key), 'gross_pnl': float(g.gross_pnl.sum()), 'net_pnl': float(g.net_pnl.sum()),
                           'transaction_cost': float(g.transaction_cost_pnl.sum()), 'slippage': float(g.slippage_pnl.sum()),
                           'account_return': float((1 + g.account_return).prod() - 1) if (g.equity > 0).all() and (prior.loc[g.index] > 0).all() else None})
        return result
    metrics = {'number_of_trades': len(trades), 'long_trades': int((trades.direction == 'long').sum()),
               'short_trades': int((trades.direction == 'short').sum()),
               'win_rate_net': len(wins) / len(trades) if len(trades) else None,
               'average_winner_currency': float(wins.net_pnl.mean()) if len(wins) else None,
               'average_loser_currency': float(losses.net_pnl.mean()) if len(losses) else None,
               'average_winner_r': float(wins.net_r.mean()) if len(wins) else None,
               'average_loser_r': float(losses.net_r.mean()) if len(losses) else None,
               'profit_factor_net': float(wins.net_pnl.sum() / -losses.net_pnl.sum()) if len(losses) else None,
               'expectancy_net_r': float(trades.net_r.mean()) if len(trades) else None,
               'gross_pnl': float(daily.gross_pnl.sum()), 'net_pnl': float(daily.net_pnl.sum()),
               'transaction_cost_pnl': float(daily.transaction_cost_pnl.sum()), 'slippage_pnl': float(daily.slippage_pnl.sum()),
               'maximum_drawdown_pct': float(-daily.drawdown_pct.min()) if len(daily) else None,
               'maximum_drawdown_currency': float(-daily.drawdown_currency.min()) if len(daily) else None,
               'sharpe_daily_252_rf0': float(np.sqrt(252) * daily.account_return.mean() / sigma) if sigma > 0 and (prior > 0).all() and (daily.equity > 0).all() else None,
               'account_equity_positive_throughout': bool((daily.equity > 0).all()),
               'account': {'initial_capital_inr': config.capital, 'sizing': 'Fractional index proxy units = initial capital * notional_fraction / adverse entry fill; default 10% fixed initial-capital notional, no compounding of position sizing.',
                           'equity': 'Initial capital plus cumulative net INR proxy P&L, independently reset in each split; daily closing equity, no intraday mark-to-market.',
                           'daily_return': 'Daily net P&L / previous observed-session closing equity. All observed regular dates included, including excluded/no-trade dates as zero. No holidays/weekends added. No risk-free return; annualization sqrt(252).',
                           'not_executable': 'Index-price research proxy; fractional units are not futures contracts; futures basis, margin, lot sizes and liquidity are not modeled.'},
               'config': asdict(config), 'observed_daily_return_count': len(daily), 'monthly': aggregates('M'), 'yearly': aggregates('Y')}
    return metrics, daily


def write_report(root, results, audit, config):
    lines = ['# NIFTY 50 ORB research baseline', '',
             'No optimization, additional signal filters, or ML training. Source CSV remains unchanged.', '',
             '## Data and rules', '',
             f"Source: {audit['source_rows']:,} rows; SHA-256 `{audit['source_sha256']}`. Coverage: {audit['coverage_start']} to {audit['coverage_end']}.",
             f"Observed regular sessions: {len(audit['sessions'])}; eligible: {audit['eligible_sessions']}; excluded: {audit['excluded_sessions']}. Outside-hours rows excluded only in memory: {audit['outside_regular_rows_excluded']}. Zero volume is unavailable and never used as a signal.",
             'IST candle starts 09:15–15:29. Regular hours follow the requested 09:15–15:30 convention ([NSE market timings](https://www.nseindia.com/static/market-data/market-timings)). No external calendar applied; absent whole sessions cannot be detected. All incomplete observed regular sessions are excluded to avoid simulating through missing execution bars. This retrospective quality screen can cause selection bias and is not deployable as a morning filter.',
             'OR uses exactly 09:15–09:29. First completed candle starting 09:30–11:59 closing strictly outside the range creates the sole daily event. Prediction is at its close, one minute after candle start. Entry is the next available open at or before 12:00 (inclusive); complete-session screening makes it the immediate next minute in this run. A gap back inside the range is allowed if actual entry-to-stop R stays positive; nonpositive R skips the event without searching again.',
             'Stop is opposite OR boundary; target is actual adverse entry fill plus/minus 1.5R. Stops take precedence on stop/target collisions. Stop gaps fill at adverse candle open; targets receive no favorable gap improvement. Process entry candle after filling at its open. Remaining positions exit at 15:29 open, before the session end. Intrabar exit timestamps denote the candle containing execution, not an invented exact second; precision and candle-end fields identify uncertainty.',
             f'Costs: {config.cost_bps:g} bps per side on adverse entry/exit notional. Slippage: {config.slippage_bps:g} bps adverse per fill, separately itemized. Gross P&L uses reference entry/exit prices for the same realized path; slippage and costs reconcile to net. It is not a separate zero-slippage backtest, since slippage changes R and target levels.', '',
             '## Account methodology', '',
             f'Each split starts with INR {config.capital:,.0f}. Fractional proxy units = initial capital * {config.notional_fraction:g} / entry fill; fixed initial-capital allocation per trade, not compounding sizing. Default allocation is 10%, leaving 90% cash initially. Daily equity = initial capital + cumulative net P&L. Daily account return = net P&L / prior observed-session equity; excluded and no-trade observed dates contribute zero. Sharpe uses sample standard deviation, sqrt(252), and zero risk-free return. Drawdown uses closing account equity including initial capital. No CAGR calculated. No intraday drawdown or futures margin/lot sizing modeled. Index prices are not executable futures prices.',
             'An initial diagnostic at 100% fixed initial-capital notional exhausted IS equity, making account-return Sharpe undefined. The reported capital illustration uses 10% fixed allocation consistently in both splits. This is a disclosed account assumption, not an optimized ORB parameter; trade events, fills, point costs and R outcomes are identical across allocations.', '',
             '## Separate factual results', '',
             '| Metric | IS 2015–2021 | OOS 2022–last date |', '|---|---:|---:|']
    a, b = results['is'], results['oos']
    for key in ['number_of_trades','long_trades','short_trades','win_rate_net','average_winner_currency','average_loser_currency','profit_factor_net','expectancy_net_r','gross_pnl','net_pnl','transaction_cost_pnl','slippage_pnl','maximum_drawdown_pct','maximum_drawdown_currency','sharpe_daily_252_rf0']:
        def fmt(v): return 'N/A' if v is None else str(v) if isinstance(v, int) else f'{v:.6f}'
        lines.append(f'| {key} | {fmt(a[key])} | {fmt(b[key])} |')
    lines += ['', 'Monthly/yearly P&L and compounded account returns are in each metrics JSON and separate performance CSVs. Win rate and average winners/losers use net P&L; target success is a different label.',
              f"IS evaluated through {a['split']['last_observed_date']}; OOS evaluated through {b['split']['last_observed_date']}. Under the stated assumptions, net P&L is {'negative' if a['net_pnl'] < 0 else 'nonnegative'} in IS and {'negative' if b['net_pnl'] < 0 else 'nonnegative'} in OOS. Transaction costs and slippage are explicitly deducted; these results do not justify changing rules using OOS outcomes.", '',
              '## ML event contract and leakage prevention', '',
              'One row per first eligible daily breakout; no subsequent events are considered after the first signal. `feature_*` fields and breakout/prediction timestamps are the only predictor inputs. `fill_*`, `target_*`, `outcome_*`, and execution status are separated by prefixes and excluded from predictors. Prediction is at signal-candle completion, before its next-open execution; prices of the entry and future candles are never features. Same clock time for close/next-open is a sequencing convention: observe close, then execute open; real latency is not modeled.',
              'Feature percentages are decimal fractions. OR width percentage uses 09:15 open. Gap uses current 09:15 open versus prior observed regular close. Previous-day return is close-to-close. ATR is a simple mean of 14 historical true ranges; historical volatility is the sample standard deviation of 20 historical daily close returns. Both are shifted by one session. These are ML features only, never trading filters. Incomplete prior regular days leave NaNs and interrupt rolling windows. History flows from IS into OOS as already-known past data, never backward. Evening observations are not part of the regular daily history. No imputation, normalization, scaler fitting, or OOS aggregate features. Early warm-up rows retain missing history.',
              '`target_reached_1_5r` is 1 only for target exits; conservative stop-first collisions are 0. `target_net_return_r` is realized adverse-fill/cost-adjusted points divided by actual entry-stop risk. Skipped fills retain feature rows with null targets; exclude these from future supervised training rather than assigning artificial failures. Labels are known only after exit. Future validation must split chronologically by date, purge any overlapping outcome horizons and fit all preprocessing on training folds only.', '',
              '## Reproduction', '',
              'Run `python -m pip install -r requirements.txt`, `python -m unittest discover -s tests -v`, then `python orb_pipeline.py`. Configurable flags: `--capital`, `--notional-fraction`, `--cost-bps`, `--slippage-bps`, `--source`, `--output`. Rules remain fixed; no parameter search. Source hash is checked before/after processing. Environment versions and detailed session exclusions are in the audit JSON.',
              'This is a descriptive index-price baseline, not evidence of executable futures profitability. OOS was evaluated once under the same fixed rules.']
    (root / 'orb_research_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='NIFTY_50_minute.csv')
    parser.add_argument('--output', default='.')
    parser.add_argument('--capital', type=float, default=1_000_000)
    parser.add_argument('--cost-bps', type=float, default=5)
    parser.add_argument('--slippage-bps', type=float, default=1)
    parser.add_argument('--notional-fraction', type=float, default=0.1)
    args = parser.parse_args()
    config = Config(capital=args.capital, cost_bps=args.cost_bps, slippage_bps=args.slippage_bps, notional_fraction=args.notional_fraction)
    if not all(np.isfinite(v) for v in asdict(config).values()) or config.capital <= 0 or config.notional_fraction <= 0 or min(config.cost_bps, config.slippage_bps) < 0:
        raise ValueError('Capital must be positive and costs/slippage finite and nonnegative')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    sessions, daily, audit = load_and_audit(args.source)
    history = historical_features(daily)
    results = {}
    audit['execution_skips'] = {}
    for split, start, end in [('is', '2015-01-01', '2021-12-31'), ('oos', '2022-01-01', '9999-12-31')]:
        trades, events, dates, skips = run_split(sessions, history, start, end, config)
        metrics, account = account_metrics(trades, dates, config)
        metrics['split'] = {'name': split, 'start': start, 'last_observed_date': dates[-1] if dates else None}
        trades.to_csv(root / f'orb_{split}_trades.csv', index=False)
        events.to_csv(root / f'orb_ml_{split}.csv', index=False)
        account.to_csv(root / f'orb_{split}_daily_account.csv')
        for period in ['monthly','yearly']:
            pd.DataFrame(metrics[period]).to_csv(root / f'orb_{split}_{period}.csv', index=False)
        (root / f'orb_{split}_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        results[split] = metrics
        audit['execution_skips'][split] = skips
        print(split.upper(), json.dumps({k: metrics[k] for k in ['number_of_trades','net_pnl','expectancy_net_r','sharpe_daily_252_rf0']}))
    audit['source_unchanged'] = sha256(args.source) == audit['source_sha256']
    if not audit['source_unchanged']: raise RuntimeError('Source changed during execution')
    audit['environment'] = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__}
    audit['config'] = asdict(config)
    audit['pipeline_sha256'] = sha256(Path(__file__))
    (root / 'orb_data_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    write_report(root, results, audit, config)


if __name__ == '__main__':
    main()
