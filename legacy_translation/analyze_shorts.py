"""Exploratory short-side diagnostics. No costs or strategy-rule changes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Short side tests'
OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261109)


def metrics(g):
    r=g.R; loss=-r.clip(upper=0).sum()
    eq=r.cumsum(); dd=eq-eq.cummax().clip(lower=0)
    return dict(trades=len(g),net_R=r.sum(),avg_R=r.mean(),win_rate_pct=100*(r>0).mean(),
                profit_factor_R=r.clip(lower=0).sum()/loss if loss>0 else np.nan,
                max_drawdown_R=-dd.min(),net_points=g.pnl_points.sum(),
                target_hit_pct=100*g.exit_reason.str.startswith('target').mean())


def table(t,key,name):
    rows=[]
    for k,g in t.groupby(key,observed=True,sort=True):
        rows.append(dict(bucket=str(k),period='ALL',**metrics(g)))
        for period,x in g.groupby('period'):
            rows.append(dict(bucket=str(k),period=period,**metrics(x)))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/name,index=False,float_format='%.8f')
    return out


def md_table(df,cols):
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in df[cols].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return lines


def month_ci(g,start,end):
    # Resample whole calendar months, including months without selected trades.
    months=pd.period_range(start,end,freq='M')
    agg=g.groupby(g.entry_time.dt.to_period('M')).R.agg(['sum','count']).reindex(months,fill_value=0)
    picks=RNG.integers(0,len(agg),size=(10000,len(agg)))
    sums=agg['sum'].to_numpy()[picks].sum(axis=1)
    counts=agg['count'].to_numpy()[picks].sum(axis=1)
    vals=sums[counts>0]/counts[counts>0]
    return np.quantile(vals,[.025,.975])


def main():
    t=pd.read_csv(ROOT/'PY_ORB_minute_SHORT.csv',parse_dates=['entry_time','entry_bar_time'])
    t['year']=t.entry_time.dt.year
    t['period']=np.where(t.year<=2020,'2015-2020','2021-2025')
    t['weekday']=t.entry_time.dt.dayofweek.map(dict(enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])))
    clock=t.entry_bar_time.dt.hour*60+t.entry_bar_time.dt.minute
    t['entry_window']=(clock//30*30).map(lambda x:f'{x//60:02d}:{x%60:02d}-{(x+30)//60:02d}:{(x+30)%60:02d}')
    t['width_pct']=100*t.orb_width/((t.orb_high+t.orb_low)/2)
    t['target_R_bucket']=pd.cut(t.target_R,[0,.5,1,1.5,2,np.inf],right=False,
                               labels=['<0.5R','0.5-1R','1-1.5R','1.5-2R','>=2R'])
    t['extension_ORB']=(t.orb_low-t.entry_price)/t.orb_width
    t['extension_bucket']=pd.cut(t.extension_ORB,[0,.1,.25,.5,np.inf],right=False,
                                labels=['<0.10 ORB','0.10-0.25 ORB','0.25-0.50 ORB','>=0.50 ORB'])
    # Previous observed regular-session day's closing bar, requiring a full
    # 15:25 bar. Do not silently skip missing prior closes to an older day.
    b=pd.read_csv(ROOT/'PY_ORB_5MIN_INDICATORS.csv',parse_dates=['time'])
    b['date']=b.time.dt.normalize()
    daily=b.groupby('date',sort=True).agg(day_open=('open','first'),last_time=('time','last'),
                                        day_close=('close','last'),last_count=('minute_count','last'))
    daily.loc[(daily.last_time.dt.strftime('%H:%M')!='15:25')|(daily.last_count!=5),'day_close']=np.nan
    daily['previous_close']=daily.day_close.shift(1)
    daily['gap_pct']=100*(daily.day_open/daily.previous_close-1)
    t['date']=t.entry_time.dt.normalize()
    t=t.merge(daily[['gap_pct']],left_on='date',right_index=True,how='left')
    t['gap_bucket']=pd.cut(t.gap_pct,[-np.inf,-.25,.25,np.inf],right=False,
                           labels=['Gap down <-0.25%','Flat -0.25 to +0.25%','Gap up >=+0.25%'])
    outputs={
        'Entry time':table(t,'entry_window','SHORT_ENTRY_TIME.csv'),
        'Weekday':table(t,'weekday','SHORT_WEEKDAY.csv'),
        'Initial target distance':table(t,'target_R_bucket','SHORT_TARGET_R.csv'),
        'Breakdown extension':table(t,'extension_bucket','SHORT_EXTENSION.csv'),
        'Opening gap':table(t,'gap_bucket','SHORT_GAP.csv'),
        'Year':table(t,'year','SHORT_YEAR.csv'),
    }
    # Joint view is reported rather than optimized. Flag small cells.
    t['width_half']=np.where(t.width_pct<=t.width_pct.median(),'smaller half','larger half')
    joint=[]
    for (window,width),g in t.groupby(['entry_window','width_half']):
        joint.append(dict(entry_window=window,width_half=width,small_sample=len(g)<50,**metrics(g)))
    pd.DataFrame(joint).to_csv(OUT/'SHORT_TIME_X_WIDTH.csv',index=False,float_format='%.8f')
    concentration=[]
    for period,g in [('ALL',t),*list(t.groupby('period'))]:
        for count in [0,1,5,10]:
            remove=g.R.nlargest(count).index
            kept=g.drop(index=remove)
            concentration.append(dict(period=period,largest_winners_removed=count,
                                      removed_R=g.loc[remove,'R'].sum(),**metrics(kept)))
    concentration=pd.DataFrame(concentration)
    concentration.to_csv(OUT/'SHORT_WINNER_CONCENTRATION.csv',index=False,float_format='%.8f')
    # Evaluate a few broad predefined timing windows; do not select the best
    # arbitrary combination of the many exploratory cells above.
    masks={'All shorts':np.ones(len(t),dtype=bool),
           '09:30-10:30':(clock.to_numpy()>=570)&(clock.to_numpy()<630),
           '10:30-11:30':(clock.to_numpy()>=630)&(clock.to_numpy()<690),
           '11:30-12:30':(clock.to_numpy()>=690)&(clock.to_numpy()<750)}
    robustness=[]
    for name,mask in masks.items():
        g=t.loc[mask]
        for period,x in [('ALL',g),*list(g.groupby('period'))]:
            universe=t if period=='ALL' else t[t.period==period]
            lo,hi=month_ci(x,universe.entry_time.min().to_period('M'),universe.entry_time.max().to_period('M'))
            robustness.append(dict(window=name,period=period,avg_R_ci_low=lo,avg_R_ci_high=hi,**metrics(x)))
    robust=pd.DataFrame(robustness)
    robust.to_csv(OUT/'SHORT_TIMING_ROBUSTNESS.csv',index=False,float_format='%.8f')
    t.to_csv(OUT/'SHORT_ENRICHED_TRADES.csv',index=False,float_format='%.8f')
    assert len(t)==1061 and t.trade_id.nunique()==1061
    for key,df in outputs.items():
        if key!='Opening gap':assert df[df.period=='ALL'].trades.sum()==len(t)
    assert np.isclose(t.R.sum(),53.26192290319,atol=1e-6)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key in zip(axes,['Entry time','Weekday','Initial target distance']):
        df=outputs[key]
        buckets=list(df[df.period=='ALL'].bucket)
        x=np.arange(len(buckets))
        for offset,period,color in [(-.18,'2015-2020','#4c72b0'),(.18,'2021-2025','#dd8452')]:
            vals=df[df.period==period].set_index('bucket').reindex(buckets)
            ax.bar(x+offset,vals.avg_R,width=.34,color=color,label=period)
        ax.axhline(0,color='gray',linewidth=1);ax.set_title(key)
        ax.set_xticks(x,buckets,rotation=45,ha='right');ax.set_ylabel('Average R before costs')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('NIFTY ORB shorts: stability across periods')
    fig.tight_layout();fig.savefig(OUT/'SHORT_TESTS.png',dpi=160);plt.close(fig)
    report=['# Additional NIFTY ORB short-side tests','',
            '1,061 trades, 2015-01-09 through 2025-07-25. Before costs. These are subsets of the original one-trade-per-day strategy; a long trade can prevent a later short. No rules were changed or independently rebacktested.','',
            '## Findings','',
            '- The 10:30–11:30 signal-bar window is the strongest broad timing candidate: 188 trades, +0.1928R average, +36.2536R total. Earlier-period average is +0.1076R (122 trades), later-period average +0.3505R (66 trades). These are observed patterns, not validated filters.',
            '- The 11:30–12:00 signal bucket is negative in both periods, with 75 trades total. Adjacent 12:00–12:30 results are positive but have only 55 trades.',
            '- Thursday is positive in both periods: +0.1004R and +0.1642R average. Monday is also positive in both periods but is weaker later. Tuesday is negative overall, yet turns positive later; the weekday pattern is not uniformly stable.',
            '- Gap-down days below -0.25% have positive averages in both periods: +0.0776R and +0.1120R. Gap-up shorts switch from negative to positive, illustrating regime sensitivity.',
            '- Initial targets >=2R look stronger overall, but most of their profit is later: +0.0159R early versus +0.3608R later. A high initial target R is not a stable stand-alone filter in this analysis.',
            '- Removing the ten best trades reduces net R from +53.2619R to +25.4165R. Those trades supply 52.3% of net profit. Earlier-period performance turns negative after removing its five best trades.',
            '- The overall month-bootstrap 95% interval for average R is approximately -0.0023R to +0.1030R, spanning zero. Uncertainty remains material even before costs. The pooled 10:30–11:30 interval is +0.0413R to +0.3523R, but it does not adjust for prior inspection or multiple comparisons.','',
            'Entry windows use five-minute candle opening labels. Actual entries occur five minutes later. For example, the 10:30–11:30 window means fills at 10:35 through 11:30.','']
    for key,df in outputs.items():
        if key=='Year':continue
        report+=['## '+key,'']+md_table(df,['bucket','period','trades','avg_R','net_R','win_rate_pct'])+['']
    report+=['## Timing stability and uncertainty','',
             'Approximate 95% intervals resample whole calendar months, including months with no selected trades. They retain dependence within each month but not across months; this is not a correction for testing many filters.','']
    report+=md_table(robust,['window','period','trades','avg_R','avg_R_ci_low','avg_R_ci_high'])+['',
             '## Dependence on large winners','',
             'Removing the best trades is a stress test of profit concentration, not an achievable trading rule.','']
    report+=md_table(concentration,['period','largest_winners_removed','removed_R','trades','net_R','avg_R'])+['',
             '## Definitions and limitations','',
             '- Target R is the fixed initial target distance divided by initial risk, before trailing-stop changes.',
             '- Breakdown extension is (ORB low − entry price) / ORB width, known at entry.',
             '- Opening gap is 09:15 open versus the previous observed regular-session day close. Prior close must have a full 15:25 candle; unavailable observations are omitted from gap buckets.',
             f'- Gap analysis excludes {int(t.gap_pct.isna().sum())} trades with unavailable prior close.',
             '- Width/time interactions are descriptive; cells below 50 trades are flagged.',
             '- The 2015–2020/2021–2025 comparison is a stability check on already-inspected data, not a pristine out-of-sample test.',
             '- Many slices were inspected. Strong-looking buckets can occur by chance. No best-filter search or combined filter is implemented.',
             '- All original data exclusions, minute-OHLC execution assumptions, and costs omitted from the source backtest still apply.','']
    (OUT/'SHORT_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    for key,df in outputs.items():
        if key!='Year':print('\n'+key+'\n'+df[['bucket','period','trades','avg_R','net_R']].to_string(index=False))
    print('\nROBUSTNESS\n',robust[['window','period','trades','avg_R','avg_R_ci_low','avg_R_ci_high']].to_string(index=False))
    print('\nCONCENTRATION\n',concentration[['period','largest_winners_removed','removed_R','net_R']].to_string(index=False))
    print('\nPASS: row counts, group totals, unique trade IDs and baseline net R. Saved to',OUT)


if __name__=='__main__':main()
