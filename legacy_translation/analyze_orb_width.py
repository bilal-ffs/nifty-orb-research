"""Descriptive ORB-width study on the existing minute-execution trade sample."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ORB width analysis'
OUT.mkdir(exist_ok=True)
LABELS=['Q1 smallest','Q2','Q3','Q4 largest']
RNG=np.random.default_rng(20261109)


def wilson(k,n):
    z=1.95996398454
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(center-half),100*(center+half)


def metrics(g):
    r=g.R
    negative_R=-r.clip(upper=0).sum()
    loss=g.pnl_points<0
    lo,hi=wilson(int(loss.sum()),len(g))
    return dict(trades=len(g),min_width_points=g.orb_width.min(),
                max_width_points=g.orb_width.max(),min_width_pct=g.width_pct.min(),
                max_width_pct=g.width_pct.max(),loss_rate_pct=100*loss.mean(),
                loss_rate_ci_low=lo,loss_rate_ci_high=hi,
                target_hit_pct=100*g.exit_reason.str.startswith('target').mean(),
                stop_exit_pct=100*g.exit_reason.str.startswith('stop').mean(),
                profitable_stop_exit_pct=100*((g.R>0)&g.exit_reason.str.startswith('stop')).mean(),
                back_inside_close_pct=100*g.back_inside_close.mean(),
                opposite_boundary_close_pct=100*g.opposite_boundary_close.mean(),
                back_inside_30min_pct=100*g.back_inside_30min.mean(),
                net_R=r.sum(),avg_R=r.mean(),
                profit_factor_R=r.clip(lower=0).sum()/negative_R if negative_R>0 else np.nan,
                median_target_R=g.target_R.median())


def summarize(t,col,name,extra=None):
    keys=['side']+([] if extra is None else extra)+[col]
    rows=[]
    for k,g in t.groupby(keys,observed=True,sort=True):
        rows.append(dict(zip(keys,k),**metrics(g)))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/name,index=False,float_format='%.8f')
    return out


def diff_ci(a,b):
    """IID empirical bootstrap, widest minus smallest, percentage points."""
    x=RNG.binomial(len(a),a.mean(),10000)/len(a)
    y=RNG.binomial(len(b),b.mean(),10000)/len(b)
    return np.quantile(100*(x-y),[.025,.975])


def main():
    t=pd.read_csv(ROOT/'PY_ORB_minute_ALL.csv',parse_dates=['entry_time','entry_bar_time'])
    b=pd.read_csv(ROOT/'PY_ORB_5MIN_INDICATORS.csv',parse_dates=['time'])
    # Normalize by ORB midpoint, known before the signal and independent of
    # trade direction/entry distance. This differs slightly from the older
    # export's entry-price denominator.
    t['width_pct']=100*t.orb_width/((t.orb_high+t.orb_low)/2)
    t['year']=t.entry_time.dt.year
    day_bars={d:g for d,g in b.groupby(b.time.dt.normalize())}
    flags=[]
    for row in t.itertuples():
        g=day_bars[row.entry_time.normalize()]
        # Strictly AFTER entry close; observe through 15:00, including after
        # the strategy has exited, to study breakout behavior independently.
        ct=g.time+pd.Timedelta(minutes=5)
        g=g[(ct>row.entry_time)&(ct<=row.entry_time.normalize()+pd.Timedelta(hours=15))]
        if row.side=='long':
            inside=g.close<row.orb_high; opposite=g.close<row.orb_low
        else:
            inside=g.close>row.orb_low; opposite=g.close>row.orb_high
        early=g.time+pd.Timedelta(minutes=5)<=row.entry_time+pd.Timedelta(minutes=30)
        flags.append((bool(inside.any()),bool(opposite.any()),bool((inside&early).any())))
    t[['back_inside_close','opposite_boundary_close','back_inside_30min']]=flags
    assert (t.opposite_boundary_close<=t.back_inside_close).all()
    assert (t.back_inside_30min<=t.back_inside_close).all()
    assert len(t)==2166 and not t.trade_id.duplicated().any()
    for side in ['long','short']:
        idx=t.side==side
        t.loc[idx,'point_quartile']=pd.qcut(t.loc[idx,'orb_width'],4,labels=LABELS).astype(str)
        t.loc[idx,'pct_quartile']=pd.qcut(t.loc[idx,'width_pct'],4,labels=LABELS).astype(str)
    for (side,year),g in t.groupby(['side','year']):
        t.loc[g.index,'annual_pct_quartile']=pd.qcut(g.width_pct.rank(method='first'),4,labels=LABELS).astype(str)
    t['period']=np.where(t.year<=2020,'2015-2020','2021-2025')
    points=summarize(t,'point_quartile','WIDTH_POINTS_QUARTILES.csv')
    pct=summarize(t,'pct_quartile','WIDTH_PERCENT_QUARTILES.csv')
    annual=summarize(t,'annual_pct_quartile','WIDTH_WITHIN_YEAR_QUARTILES.csv')
    assert int(pct.trades.sum())==int(points.trades.sum())==int(annual.trades.sum())==len(t)
    summarize(t,'pct_quartile','WIDTH_PERCENT_BY_YEAR.csv',['year'])
    summarize(t,'pct_quartile','WIDTH_PERCENT_BY_PERIOD.csv',['period'])
    # Freeze quartile thresholds on 2015-2020; apply unchanged to later data.
    cutoffs={}
    for side in ['long','short']:
        idx=t.side==side
        cuts=t.loc[idx&(t.year<=2020),'width_pct'].quantile([.25,.5,.75]).to_numpy()
        cutoffs[side]=list(cuts)
        t.loc[idx,'frozen_pct_quartile']=pd.cut(t.loc[idx,'width_pct'],[-np.inf,*cuts,np.inf],labels=LABELS).astype(str)
    frozen=summarize(t,'frozen_pct_quartile','WIDTH_FROZEN_THRESHOLDS_BY_PERIOD.csv',['period'])
    t.to_csv(OUT/'WIDTH_ENRICHED_TRADES.csv',index=False,float_format='%.8f')
    (OUT/'WIDTH_THRESHOLDS.json').write_text(json.dumps(cutoffs,indent=2),encoding='utf-8')
    # Difference confidence intervals use empirical binomial bootstrap.
    comparisons=[]
    for scheme,col in [('Points','point_quartile'),('Percent','pct_quartile'),('Within-year percent','annual_pct_quartile')]:
        for side in ['long','short']:
            a=t[(t.side==side)&(t[col]=='Q4 largest')]
            z=t[(t.side==side)&(t[col]=='Q1 smallest')]
            for metric in ['loss','back_inside_close','opposite_boundary_close']:
                av=(a.R<0) if metric=='loss' else a[metric]
                zv=(z.R<0) if metric=='loss' else z[metric]
                lo,hi=diff_ci(av,zv)
                comparisons.append(dict(scheme=scheme,side=side,metric=metric,
                                        widest_minus_smallest_pp=100*(av.mean()-zv.mean()),
                                        ci_low_pp=lo,ci_high_pp=hi))
    comp=pd.DataFrame(comparisons)
    comp.to_csv(OUT/'WIDTH_LARGEST_VS_SMALLEST.csv',index=False,float_format='%.8f')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for j,side in enumerate(['long','short']):
        g=pct[pct.side==side]
        x=np.arange(4)
        axes[0,j].errorbar(x,g.loss_rate_pct,
                          yerr=[g.loss_rate_pct-g.loss_rate_ci_low,g.loss_rate_ci_high-g.loss_rate_pct],
                          fmt='o-',capsize=4,color='#c44e52')
        axes[0,j].set_title(f'{side.title()}: losing trades (95% intervals)')
        axes[0,j].set_ylabel('Trades (%)');axes[0,j].set_ylim(0,100)
        for k,row in enumerate(g.itertuples()):
            axes[0,j].annotate(f'n={row.trades}',(k,row.loss_rate_ci_high+3),ha='center',fontsize=9)
        axes[1,j].plot(x,g.avg_R,'o-',color='#4c72b0')
        axes[1,j].axhline(0,color='gray',linewidth=1)
        axes[1,j].set_title(f'{side.title()}: average R per trade')
        axes[1,j].set_ylabel('R before costs')
        axes[1,j].set_xticks(x,['Smallest','Q2','Q3','Largest'])
        axes[1,j].set_xlabel('Opening-range width as % of ORB midpoint')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('NIFTY ORB width study | 2015-01-09 to 2025-07-25 | before costs',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'WIDTH_ANALYSIS.png',dpi=170);plt.close(fig)
    report=['# Does a bigger opening range make the breakout more likely to fail?','',
            'Sample: 2,166 executed trades from the minute-execution backtest, 2015-01-09 through 2025-07-25. All figures before costs.','',
            'The results do not establish that a bigger ORB causes more losing trades. Loss rates are not monotonic across width buckets. The largest-minus-smallest loss-rate intervals span zero for both sides using raw points, normalized percentage width, and within-year percentage quartiles.','',
            'Large short-side ranges do show more subsequent closes back above the ORB low: 79.2% versus 68.8% for the smallest ranges. The association persists using within-year width quartiles. This is a price-action observation through 15:00, including after the actual trade exits; it is not equivalent to a losing trade.','',
            'Larger ranges have lower target-hit rates, but the target is itself 1.5 times ORB width and therefore farther away. Target misses alone do not establish a weaker trading edge. The largest normalized-width quartile has positive mean R on both sides, but those means are descriptive and sensitive to sample/regime.','',
            '## Failure definitions','',
            '- Primary: the strategy closes the trade with negative R.',
            '- Back inside: a subsequent 5-minute close is below the ORB high for a long, or above the ORB low for a short. This includes moves through the entire range.',
            '- Opposite boundary: a subsequent 5-minute close breaks the far side of the opening range.',
            '- Price-action flags observe from after entry through 15:00, even if the strategy exited earlier. They do not mean the realized trade lost. A trailing-stop exit can also be profitable.',
            '- A target miss is not counted as a loss automatically. Larger ranges mechanically place the fixed target farther away.','',
            '## Width normalized for the index level','',
            'Width % = 100 × (ORB high − ORB low) / ORB midpoint. Quartiles are calculated separately by trade direction.','',
            '| Side | Width quartile | Width % range | Trades | Losing % | Target hit % | Back inside % | Opposite break % | Average R |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in pct.itertuples():
        report.append(f'| {r.side} | {r.pct_quartile} | {r.min_width_pct:.3f}–{r.max_width_pct:.3f} | {r.trades} | {r.loss_rate_pct:.1f} | {r.target_hit_pct:.1f} | {r.back_inside_close_pct:.1f} | {r.opposite_boundary_close_pct:.1f} | {r.avg_R:+.4f} |')
    report+=['','## Largest minus smallest: loss-rate difference','',
             '| Width measure | Side | Difference, percentage points | Approx. 95% interval |',
             '|---|---|---:|---:|']
    for r in comp[comp.metric=='loss'].itertuples():
        report.append(f'| {r.scheme} | {r.side} | {r.widest_minus_smallest_pp:+.1f} | {r.ci_low_pp:+.1f} to {r.ci_high_pp:+.1f} |')
    report+=['','Positive differences mean more losses with the largest ranges. Intervals spanning zero do not establish a difference. These intervals assume independent trades; serial dependence and multiple comparisons are not corrected.','',
             '## Later period using earlier fixed thresholds','',
             'Cutoffs are set on 2015–2020 only and applied unchanged to 2021–2025. This is a chronological stability check, not an untouched validation sample; the strategy and dataset were already inspected.','',
             '| Side | Frozen width bucket | Later-period trades | Losing % | Average R |',
             '|---|---|---:|---:|---:|']
    for r in frozen[frozen.period=='2021-2025'].itertuples():
        report.append(f'| {r.side} | {r.frozen_pct_quartile} | {r.trades} | {r.loss_rate_pct:.1f} | {r.avg_R:+.4f} |')
    report+=['','## Interpretation limits','',
             '- Results are conditional on trades this exact strategy took: direction confirmation, entry session, one trade/day, and trailing stop. They do not measure every ORB breakout or days without a trade.',
             '- Full-sample quartiles are descriptive. Within-year quartiles reduce index-level/year-regime confounding but do not isolate causal effects. Volatility and entry time can still affect results.',
             '- Data exclusions and execution assumptions from PY_ORB_METHOD.md still apply. Price-action flags use available five-minute closes, not tick paths or wick-only touches.',
             '- Net R and target-hit probability answer different questions. A low target-hit rate may coexist with profitable trailing-stop exits.',
             '- No threshold has been optimized or added to the strategy. Costs remain excluded.','',
             'See the CSVs for point-width buckets, normalized buckets, within-year buckets, annual and period breakdowns, frozen-threshold checks, and enriched trade-level flags.','']
    (OUT/'WIDTH_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print('NORMALIZED WIDTH');print(pct[['side','pct_quartile','min_width_pct','max_width_pct','trades','loss_rate_pct','target_hit_pct','back_inside_close_pct','opposite_boundary_close_pct','avg_R']].to_string(index=False))
    print('DIFFERENCES');print(comp.to_string(index=False))
    print('LATER PERIOD');print(frozen[frozen.period=='2021-2025'][['side','frozen_pct_quartile','trades','loss_rate_pct','avg_R']].to_string(index=False))
    print('Saved to',OUT)


if __name__=='__main__':main()
