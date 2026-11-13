"""Causal five-minute ORB + supplied Vervoort recurrence; IS only."""
from pathlib import Path
import csv
import hashlib
import numpy as np
import pandas as pd

MULTIPLIERS=np.arange(.5,5.01,.5).tolist()
FEATURES=['feature_or_width_points','feature_or_width_pct','feature_or_direction',
          'feature_or_return','feature_breakout_direction','feature_breakout_minutes_since_0915',
          'feature_breakout_distance_points','feature_atr_points','feature_atr_pct',
          'feature_trail_distance_atr','feature_signal_risk_pct','feature_atr_trend_state']
CAPITAL=1_000_000.
ALLOCATION=.1
FEE=.0005
SLIP=.0001


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def load_is(path):
    rows=[]
    with open(path,newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=['date','open','high','low','close','volume']: raise ValueError('Schema mismatch')
        for row in reader:
            # Source is chronologically sorted. Stop at first 2022 timestamp;
            # its OHLC values are not interpreted or processed.
            year=int(row['date'][6:10])
            if year>=2022: break
            if year>=2015: rows.append(row)
    frame=pd.DataFrame(rows)
    frame['timestamp']=pd.to_datetime(frame.date,format='%d-%m-%Y %H:%M').dt.tz_localize('Asia/Kolkata')
    if not frame.timestamp.is_monotonic_increasing: raise ValueError('Unsorted IS source')
    for c in ['open','high','low','close','volume']: frame[c]=pd.to_numeric(frame[c],errors='coerce')
    return aggregate(frame)


def aggregate(frame):
    minute=frame.timestamp.dt.hour*60+frame.timestamp.dt.minute
    outside=int((~minute.between(555,929)).sum())
    g=frame[minute.between(555,929)].copy()
    g['date']=g.timestamp.dt.strftime('%Y-%m-%d')
    g['invalid']=(~np.isfinite(g[['open','high','low','close','volume']]).all(axis=1) |
                  (g[['open','high','low','close']]<=0).any(axis=1)|(g.volume<0) |
                  (g.high<g[['open','low','close']].max(axis=1)) |
                  (g.low>g[['open','high','close']].min(axis=1)))
    sessions=[]; eligible=[]
    for date,s in g.groupby('date',sort=True):
        expected=pd.date_range(date+' 09:15',periods=375,freq='min',tz='Asia/Kolkata')
        missing=expected.difference(s.timestamp)
        dup=int(s.timestamp.duplicated(keep=False).sum())
        valid=len(missing)==0 and dup==0 and not s.invalid.any() and len(s)==375
        sessions.append({'date':date,'eligible':valid,'missing_minutes':missing.strftime('%H:%M').tolist(),
                         'duplicate_rows':dup,'invalid_rows':int(s.invalid.sum())})
        if valid: eligible.append(date)
    # A candle's 5 source timestamps must all exist uniquely and be valid.
    g['bar_start']=g.timestamp.dt.floor('5min')  # 09:15 is aligned to IST 5-minute grid.
    bars=g.groupby(['date','bar_start'],sort=True).agg(open=('open','first'),high=('high','max'),
           low=('low','min'),close=('close','last'),n=('timestamp','size'),unique=('timestamp','nunique'),invalid=('invalid','max')).reset_index()
    complete=(bars.n==5)&(bars.unique==5)&(~bars.invalid)
    removed=int((~complete).sum())
    bars=bars[complete].rename(columns={'bar_start':'timestamp'}).reset_index(drop=True)
    bars['eligible']=bars.date.isin(eligible)
    audit={'is_minute_rows':len(frame),'outside_regular_rows':outside,'zero_volume_unavailable':int((frame.volume==0).sum()),
           'sessions':sessions,'eligible_sessions':len(eligible),'incomplete_5m_bars_removed':removed,
           'indicator_policy':'All complete regular 5m bars, including complete bars of excluded sessions, update continuous ATR/trail. Incomplete sessions never trade. No overnight reset of indicator state.',
           'whole_day_calendar_limit':'Absent whole sessions cannot be identified without an exchange calendar.'}
    return bars,eligible,audit


def wilder_atr(bars,period=5):
    h=bars.high.to_numpy(); l=bars.low.to_numpy(); c=bars.close.to_numpy()
    tr=h-l
    if len(c)>1: tr[1:]=np.maximum.reduce([h[1:]-l[1:],abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])])
    atr=np.full(len(c),np.nan)
    if len(c)>=period:
        atr[period-1]=tr[:period].mean()
        for i in range(period,len(c)): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr


def vervoort(close,atr,multiplier):
    trail=np.full(len(close),np.nan); trend=np.zeros(len(close),dtype=int)
    for i,current in enumerate(close):
        previous=close[i-1] if i else np.nan
        prev_stop=trail[i-1] if i and np.isfinite(trail[i-1]) else 0.
        loss=multiplier*atr[i]
        if np.isfinite(loss):
            if current>prev_stop and previous>prev_stop: trail[i]=max(prev_stop,current-loss)
            elif current<prev_stop and previous<prev_stop: trail[i]=min(prev_stop,current+loss)
            else: trail[i]=current-loss if current>prev_stop else current+loss
        if previous<prev_stop and current>prev_stop: trend[i]=1
        elif previous>prev_stop and current<prev_stop: trend[i]=-1
        else: trend[i]=trend[i-1] if i else 0
    return trail,trend


def features(row,opening,hi,lo,direction):
    width=hi-lo; op=float(opening.iloc[0].open); last=float(opening.iloc[-1].close)
    minute=row.timestamp.hour*60+row.timestamp.minute+5-555
    return {'feature_or_width_points':width,'feature_or_width_pct':width/op,
            'feature_or_direction':int(np.sign(last-op)),'feature_or_return':last/op-1,
            'feature_breakout_direction':direction,'feature_breakout_minutes_since_0915':minute,
            'feature_breakout_distance_points':row.close-hi if direction==1 else lo-row.close,
            'feature_atr_points':row.atr,'feature_atr_pct':row.atr/row.close,
            'feature_trail_distance_atr':round(abs(row.close-row.trail)/row.atr,12),
            'feature_signal_risk_pct':abs(row.close-row.trail)/row.close,
            'feature_atr_trend_state':int(row.trend)}


def simulate(g,signal,direction,hi,lo,mode,multiplier):
    row=g.iloc[signal]
    confirmation=row.timestamp+pd.Timedelta(minutes=5)
    fill_index=signal+1
    reference_entry=float(row.close) if mode=='pine_close' else float(g.iloc[fill_index].open)
    entry=reference_entry*(1+direction*SLIP)
    initial_stop=float(row.trail); risk=direction*(entry-initial_stop)
    # Stop must be valid versus unslipped next open as well as actual fill.
    if risk<=0 or direction*(reference_entry-initial_stop)<=0:
        return None,'invalid_initial_stop_at_fill'
    target=float(row.close+direction*1.5*(hi-lo))
    active_stop=initial_stop; updates=0
    for j in range(fill_index,len(g)):
        bar=g.iloc[j]
        minute=bar.timestamp.hour*60+bar.timestamp.minute
        if minute>895: break
        if (direction==1 and bar.open<=active_stop) or (direction==-1 and bar.open>=active_stop):
            reference_exit=float(bar.open); reason='stop_gap'; precision='exact_open'; exit_time=bar.timestamp
        elif j==fill_index and mode=='next_open' and direction*(reference_entry-target)>=0:
            reference_exit=reference_entry; reason='target_gap_at_entry'; precision='exact_open'; exit_time=bar.timestamp
        elif (bar.low<=active_stop if direction==1 else bar.high>=active_stop):
            reference_exit=active_stop; reason='stop'; precision='within_candle'; exit_time=bar.timestamp
        elif (bar.high>=target if direction==1 else bar.low<=target):
            reference_exit=target; reason='target'; precision='within_candle'; exit_time=bar.timestamp
        elif minute==895:  # 14:55 candle closes at 15:00.
            reference_exit=float(bar.close); reason='square_off'; precision='exact_close'; exit_time=bar.timestamp+pd.Timedelta(minutes=5)
        else:
            # Only AFTER processing intrabar exits using the previously active stop.
            if np.isfinite(bar.trail) and direction*(bar.close-bar.trail)>0:
                changed=max(active_stop,float(bar.trail)) if direction==1 else min(active_stop,float(bar.trail))
                updates+=int(changed!=active_stop); active_stop=changed
            continue
        exit_price=reference_exit*(1-direction*SLIP)
        gross=direction*(reference_exit-reference_entry)
        slipped=direction*(exit_price-entry)
        fees=(entry+exit_price)*FEE
        units=CAPITAL*ALLOCATION/entry
        label_end=exit_time if precision in ['exact_close','exact_open'] else bar.timestamp+pd.Timedelta(minutes=5)
        return {'date':row.date,'multiplier':multiplier,'mode':mode,'direction':'long' if direction==1 else 'short',
                'signal_candle_start':row.timestamp.isoformat(),'prediction_timestamp':confirmation.isoformat(),
                'entry_timestamp':confirmation.isoformat(),'entry_reference_price':reference_entry,'entry_price':entry,
                'initial_stop':initial_stop,'risk_points':risk,'target_price':target,
                'target_distance_actual_r':direction*(target-entry)/risk,
                'exit_timestamp':exit_time.isoformat(),'label_end_timestamp':label_end.isoformat(),
                'exit_precision':precision,'exit_reference_price':reference_exit,'exit_price':exit_price,
                'final_active_stop':active_stop,'trail_updates':updates,'exit_reason':reason,
                'target_hit':int(reason in ['target','target_gap_at_entry']),
                'gross_points':gross,'net_points':slipped-fees,'fees_points':fees,'slippage_points':gross-slipped,
                'gross_r':gross/risk,'net_r':(slipped-fees)/risk,'fees_r':fees/risk,'slippage_r':(gross-slipped)/risk,
                'units':units,'gross_pnl':gross*units,'net_pnl':(slipped-fees)*units,
                'fees_pnl':fees*units,'slippage_pnl':(gross-slipped)*units,
                **features(row,g.iloc[:3],hi,lo,direction)},None
    raise ValueError('Missing mandatory square-off candle')


def backtest(bars,multiplier,mode):
    g=bars.copy()
    g['trail'],g['trend']=vervoort(g.close.to_numpy(),g.atr.to_numpy(),multiplier)
    trades=[]; rejected=[]
    for date,day in g[g.eligible].groupby('date',sort=True):
        day=day.reset_index(drop=True)
        opening=day.iloc[:3]; hi=float(opening.high.max()); lo=float(opening.low.min())
        if len(day)!=75 or hi<=lo: continue
        for i,row in enumerate(day.itertuples(index=False)):
            minute=row.timestamp.hour*60+row.timestamp.minute
            if not 570<=minute<895: continue
            direction=1 if row.close>hi else -1 if row.close<lo else 0
            if not direction or not np.isfinite(row.trail) or direction*(row.close-row.trail)<=0: continue
            trade,reason=simulate(day,i,direction,hi,lo,mode,multiplier)
            if trade: trades.append(trade)
            else: rejected.append({'date':date,'mode':mode,'multiplier':multiplier,'reason':reason})
            break  # First allowed order attempt consumes the daily allowance.
    return pd.DataFrame(trades),rejected


def metrics(t,dates):
    def pf(x):
        loss=-x[x<0].sum()
        return float(x[x>0].sum()/loss) if loss>0 else None
    n=len(t)
    d=pd.DataFrame(index=pd.Index(dates,name='date'))
    d['net_pnl']=t.groupby('date').net_pnl.sum().reindex(dates,fill_value=0)
    d['equity']=CAPITAL+d.net_pnl.cumsum()
    previous=d.equity.shift(1,fill_value=CAPITAL)
    d['account_return']=d.net_pnl/previous
    peak=d.equity.cummax().clip(lower=CAPITAL)
    d['drawdown']=1-d.equity/peak
    sd=d.account_return.std(ddof=1)
    return {'trades':n,'long_trades':int((t.direction=='long').sum()),'short_trades':int((t.direction=='short').sum()),
            'gross_expectancy_points':float(t.gross_points.mean()),'net_expectancy_points':float(t.net_points.mean()),
            'gross_expectancy_r':float(t.gross_r.mean()),'net_expectancy_r':float(t.net_r.mean()),
            'gross_profit_factor':pf(t.gross_pnl),'net_profit_factor':pf(t.net_pnl),
            'gross_win_rate':float((t.gross_pnl>0).mean()),'net_win_rate':float((t.net_pnl>0).mean()),
            'target_rate':float(t.target_hit.mean()),'stop_rate':float(t.exit_reason.isin(['stop','stop_gap']).mean()),
            'square_off_rate':float((t.exit_reason=='square_off').mean()),'total_net_pnl':float(t.net_pnl.sum()),
            'maximum_drawdown_pct':float(d.drawdown.max()),'maximum_drawdown_inr':float((peak-d.equity).max()),
            'daily_sharpe':float(np.sqrt(252)*d.account_return.mean()/sd) if sd>0 and (d.equity>0).all() else None,
            'average_initial_risk_points':float(t.risk_points.mean()),
            'average_cost_to_risk':float((t.fees_r+t.slippage_r).mean())},d
