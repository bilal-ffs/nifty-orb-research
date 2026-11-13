"""NIFTY ORB + Vervoort ATR: 5-minute signals, before-cost one-unit results.

Run: python orb_backtest.py
Requires pandas and numpy. Input minute timestamps are bar OPEN times in IST.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def prepare(path):
    raw = pd.read_csv(path)
    input_row_count = len(raw)
    raw['time'] = pd.to_datetime(raw['date'], format='%d-%m-%Y %H:%M', errors='raise')
    duplicate_mask = raw.time.duplicated(keep=False)
    duplicate_days = set(raw.loc[duplicate_mask,'time'].dt.normalize())
    duplicate_rows = raw.loc[duplicate_mask].to_dict(orient='records')
    # Retain first source observation for indicator continuity; do not trade
    # on any day with conflicting duplicate timestamps.
    raw = raw.drop_duplicates('time',keep='first')
    raw = raw.sort_values('time').set_index('time')
    cols = ['open', 'high', 'low', 'close']
    if not np.isfinite(raw[cols].to_numpy()).all():
        raise ValueError('Non-finite OHLC')
    if ((raw.high < raw[cols].max(axis=1)) | (raw.low > raw[cols].min(axis=1))).any():
        raise ValueError('Invalid OHLC bounds')
    clock = raw.index.hour * 60 + raw.index.minute
    regular = raw.loc[(clock >= 555) & (clock < 930)].copy()
    # Midnight-aligned five-minute bins align with 09:15, 09:20, etc.
    bars = regular.resample('5min', closed='left', label='left').agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
        close=('close', 'last'), minute_count=('close', 'count'))
    bars = bars.loc[bars.minute_count > 0].copy()
    incomplete = bars[bars.minute_count != 5]
    # A day must have a complete opening range and square-off bar. Exclude
    # any day with missing minutes through 15:00, rather than invent prices.
    counts = regular.groupby(regular.index.normalize()).size()
    usable = []
    for day, group in regular.groupby(regular.index.normalize()):
        expected = pd.date_range(day + pd.Timedelta(hours=9, minutes=15),
                                 day + pd.Timedelta(hours=14, minutes=59), freq='min')
        if expected.isin(group.index).all() and day not in duplicate_days:
            usable.append(day)
    audit = {
        'input': str(path), 'input_rows': input_row_count,
        'duplicate_timestamp_rows': [{k:str(v) for k,v in r.items()} for r in duplicate_rows],
        'duplicate_policy': 'First observation retained for ATR continuity; affected days excluded from trading',
        'first_timestamp': str(raw.index.min()), 'last_timestamp': str(raw.index.max()),
        'timestamps': 'Assumed IST minute bar open times',
        'outside_0915_1530_rows': len(raw) - len(regular),
        'regular_session_days': len(counts), 'eligible_days': len(usable),
        'excluded_days': [str(x.date()) for x in counts.index if x not in set(usable)],
        'incomplete_5min_bars': len(incomplete),
    }
    return regular, bars, set(usable), audit


def indicators(bars, period=5, multiplier=3.5):
    """Pine ta.atr: true range, SMA seed, then Wilder RMA(alpha=1/n)."""
    previous_close = None
    previous_stop = np.nan
    atr = np.nan
    seed = []
    atrs, stops = [], []
    for row in bars.itertuples():
        tr = row.high - row.low if previous_close is None else max(
            row.high-row.low, abs(row.high-previous_close), abs(row.low-previous_close))
        if np.isnan(atr):
            seed.append(tr)
            if len(seed) == period:
                atr = sum(seed) / period
        else:
            atr = (atr * (period-1) + tr) / period
        prev = 0.0 if np.isnan(previous_stop) else previous_stop
        loss = multiplier * atr
        if row.close > prev and previous_close is not None and previous_close > prev:
            stop = max(prev, row.close-loss) if np.isfinite(loss) else np.nan
        elif row.close < prev and previous_close is not None and previous_close < prev:
            stop = min(prev, row.close+loss) if np.isfinite(loss) else np.nan
        else:
            stop = row.close-loss if row.close > prev else row.close+loss
        atrs.append(atr); stops.append(stop)
        previous_stop = stop
        previous_close = row.close
    bars['atr_5'] = atrs
    bars['vervoort_stop'] = stops
    return bars


def fill(side, stop, target, o, h, l, c):
    """Active bracket fills: gap at open, then OHLC inferred price path.

    On both-hit bars, TradingView assumes the nearer extreme occurs first.
    Ties use low first. Minute mode applies this only within one minute.
    """
    if side == 'long':
        if o <= stop: return o, 'stop_gap', False
        if o >= target: return o, 'target_gap', False
        hit_stop, hit_target = l <= stop, h >= target
    else:
        if o >= stop: return o, 'stop_gap', False
        if o <= target: return o, 'target_gap', False
        hit_stop, hit_target = h >= stop, l <= target
    both = bool(hit_stop and hit_target)
    if both:
        high_first = abs(o-h) < abs(o-l)
        target_first = high_first if side == 'long' else not high_first
        return (target, 'target', True) if target_first else (stop, 'stop', True)
    if hit_stop: return stop, 'stop', False
    if hit_target: return target, 'target', False
    return None


TRADE_COLUMNS = ['trade_id','side','entry_bar_time','entry_time','exit_bar_time',
                 'exit_time','entry_price','exit_price','initial_stop','final_stop',
                 'target','initial_risk_points','orb_high','orb_low','orb_width',
                 'orb_width_pct','target_R','pnl_points','R','exit_reason',
                 'holding_minutes','ambiguous_fill_bar']


def backtest(minutes, bars, eligible, mode):
    trades = []
    position = None
    day = None
    traded = False
    hi = lo = np.nan
    # Minute arrays avoid repeatedly slicing a million-row DataFrame.
    minute_groups = {}
    if mode == 'minute':
        for timestamp, row in zip(minutes.index.floor('5min'),
                                  minutes[['open','high','low','close']].itertuples()):
            minute_groups.setdefault(timestamp,[]).append(row)

    def close_position(price, reason, stamp, actual, ambiguous=False):
        nonlocal position
        p = position
        pnl = (price-p['entry_price']) * (1 if p['side']=='long' else -1)
        p.update(exit_price=price, exit_reason=reason, exit_bar_time=stamp,
                 exit_time=actual, pnl_points=pnl, R=pnl/p['initial_risk_points'],
                 holding_minutes=(actual-p['entry_time']).total_seconds()/60,
                 ambiguous_fill_bar=ambiguous)
        trades.append(p)
        position = None

    for row in bars.itertuples():
        t = row.Index
        date = t.normalize()
        if date != day:
            if position is not None:
                raise AssertionError('Position survived square-off')
            day = date; traded = False; hi = lo = np.nan
        if date not in eligible:
            continue
        clock = t.hour*60+t.minute
        # Existing stops/targets operate during the bar, BEFORE its close
        # recalculates ATR and modifies the stop for subsequent price ticks.
        if position:
            execution_bars = minute_groups[t] if mode=='minute' else [row]
            for sub in execution_bars:
                result = fill(position['side'], position['final_stop'], position['target'],
                              sub.open, sub.high, sub.low, sub.close)
                if result:
                    price, reason, ambiguous = result
                    # Exact intra-bar hit time is unknown; save containing bar open.
                    close_position(price, reason, sub.Index, sub.Index, ambiguous)
                    break
        if 555 <= clock < 570:
            hi = row.high if np.isnan(hi) else max(hi,row.high)
            lo = row.low if np.isnan(lo) else min(lo,row.low)
        width = hi-lo
        close_time = t+pd.Timedelta(minutes=5)
        if (570 <= clock < 750 and not traded and position is None
                and np.isfinite(width) and width>0 and np.isfinite(row.vervoort_stop)):
            side = None
            if row.close>hi and row.vervoort_stop<row.close: side='long'
            elif row.close<lo and row.vervoort_stop>row.close: side='short'
            if side:
                sign = 1 if side=='long' else -1
                risk = abs(row.close-row.vervoort_stop)
                position = dict(trade_id=len(trades)+1, side=side, entry_bar_time=t,
                                entry_time=close_time, entry_price=row.close,
                                initial_stop=row.vervoort_stop, final_stop=row.vervoort_stop,
                                target=row.close+sign*1.5*width, initial_risk_points=risk,
                                orb_high=hi,orb_low=lo,orb_width=width,
                                orb_width_pct=100*width/row.close,target_R=1.5*width/risk)
                traded = True
        if position and t>position['entry_bar_time']:
            if position['side']=='long' and row.vervoort_stop<row.close:
                position['final_stop']=max(position['final_stop'],row.vervoort_stop)
            elif position['side']=='short' and row.vervoort_stop>row.close:
                position['final_stop']=min(position['final_stop'],row.vervoort_stop)
        if clock==895 and position:
            close_position(row.close,'15:00_square_off',t,close_time)
    assert position is None
    result = pd.DataFrame(trades,columns=TRADE_COLUMNS)
    assert not result.entry_time.dt.normalize().duplicated().any()
    assert (result.initial_risk_points>0).all()
    assert (result.exit_time>=result.entry_time).all()
    assert (result.exit_time.dt.normalize()==result.entry_time.dt.normalize()).all()
    assert np.allclose(result.R, result.pnl_points/result.initial_risk_points)
    return result


def metrics(g):
    p=g.pnl_points; r=g.R
    positive=r.clip(lower=0).sum(); negative=-r.clip(upper=0).sum()
    gp=p.clip(lower=0).sum(); gl=-p.clip(upper=0).sum()
    equity=r.cumsum(); dd=equity-equity.cummax().clip(lower=0)
    return dict(trades=len(g),wins=int((p>0).sum()),losses=int((p<0).sum()),
                win_rate_pct=100*(p>0).mean(),net_points=p.sum(),avg_points=p.mean(),
                net_R=r.sum(),avg_R=r.mean(),profit_factor_points=gp/gl if gl else np.nan,
                profit_factor_R=positive/negative if negative else np.nan,
                max_drawdown_R=-dd.min(), avg_initial_risk_points=g.initial_risk_points.mean(),
                ambiguous_fill_bars=int(g.ambiguous_fill_bar.sum()))


def exports(trades, mode):
    prefix=ROOT/f'PY_ORB_{mode}'
    trades.to_csv(str(prefix)+'_ALL.csv',index=False,float_format='%.8f')
    summary=[]
    for side in ['long','short']:
        g=trades[trades.side==side]
        g.to_csv(str(prefix)+f'_{side.upper()}.csv',index=False,float_format='%.8f')
        summary.append(dict(side=side,**metrics(g)))
    pd.DataFrame(summary).to_csv(str(prefix)+'_SUMMARY.csv',index=False,float_format='%.8f')
    t=trades.copy()
    t['year']=t.entry_time.dt.year
    # Entry bar labels match TradingView export timestamps. Actual entries
    # occur five minutes later, so both timestamps are retained per trade.
    m=t.entry_bar_time.dt.hour*60+t.entry_bar_time.dt.minute
    t['entry_window']=(m//30*30).map(lambda x:f'{x//60:02d}:{x%60:02d}-{(x+30)//60:02d}:{(x+30)%60:02d}')
    for name,keys in [('YEAR',['side','year']),('ENTRY_TIME',['side','entry_window'])]:
        data=[dict(zip(keys,k if isinstance(k,tuple) else (k,)),**metrics(g))
              for k,g in t.groupby(keys,sort=True)]
        pd.DataFrame(data).to_csv(str(prefix)+f'_{name}.csv',index=False,float_format='%.8f')
    # Descriptive ORB-width quartiles, separate by side, calculated on all
    # trades. These are exploratory in-sample bins, not optimized filters.
    t['orb_width_quartile']=t.groupby('side').orb_width.transform(
        lambda x:pd.qcut(x,4,labels=['Q1','Q2','Q3','Q4'],duplicates='raise')).astype(str)
    data=[]
    for (side,q),g in t.groupby(['side','orb_width_quartile']):
        data.append(dict(side=side,quartile=q,min_width=g.orb_width.min(),
                         max_width=g.orb_width.max(),**metrics(g)))
    pd.DataFrame(data).to_csv(str(prefix)+'_ORB_WIDTH.csv',index=False,float_format='%.8f')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'NIFTY_50_minute.csv')
    parser.add_argument('--mode',choices=['both','pine5m','minute'],default='both')
    args=parser.parse_args()
    minutes,bars,eligible,audit=prepare(args.input)
    bars=indicators(bars)
    bars.to_csv(ROOT/'PY_ORB_5MIN_INDICATORS.csv',float_format='%.8f')
    summaries={}
    for mode in (['pine5m','minute'] if args.mode=='both' else [args.mode]):
        trades=backtest(minutes,bars,eligible,mode)
        summaries[mode]=exports(trades,mode)
        print(mode,json.dumps(summaries[mode],indent=2),flush=True)
    audit['parameters']={'orb':'09:15-09:29 inclusive','signal_timeframe_minutes':5,
                         'entry_bar_session':'09:30 <= bar open < 12:30',
                         'atr_period':5,'atr_multiplier':3.5,'target_orb_width_multiplier':1.5,
                         'quantity':1,'costs':0,'square_off':'15:00 IST'}
    audit['results']=summaries
    (ROOT/'PY_ORB_RUN_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    main()
