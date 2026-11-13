"""IS-only retrospective diagnostics. Never runs/changes the ORB pipeline."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261014
REPLICATES = 5000
FEATURES = {
    'feature_or_width_points': 'Opening range width (points)',
    'feature_or_width_pct': 'Opening range width / price',
    'feature_or_width_prev_atr14': 'Opening range width / previous ATR14',
    'feature_historical_vol20': 'Historical volatility (20 daily returns)',
    'feature_prev_day_return': 'Previous-day close return',
    'feature_gap_pct': 'Opening gap / previous close',
    'feature_prev_day_range_points': 'Previous-day range (points)',
}
EXIT_CATEGORIES = ['target', 'stop', 'stop_gap', 'session_exit']
PROTECTED = ['orb_pipeline.py', 'orb_is_trades.csv', 'orb_ml_is.csv',
             'orb_is_metrics.json', 'orb_data_audit.json', 'NIFTY_50_minute.csv']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_is(root):
    t = pd.read_csv(root / 'orb_is_trades.csv')
    f = pd.read_csv(root / 'orb_ml_is.csv')
    metrics = json.loads((root / 'orb_is_metrics.json').read_text())
    for dates in [t.date, f.event_id]:
        if not dates.between('2015-01-01', '2021-12-31').all():
            raise ValueError('Input includes observations outside IS')
        if not dates.is_unique:
            raise ValueError('Expected one IS trade/event per day')
    if len(t) != 1624 or len(f) != 1624:
        raise ValueError('Expected 1,624 original IS trades and events')
    if set(t.date) != set(f.event_id):
        raise ValueError('Trade/event dates do not match')
    # Join only pre-execution features; outcome labels are not bin inputs.
    wanted = ['event_id', *[c for c in f if c.startswith('feature_')]]
    joined = t.merge(f[wanted], left_on='date', right_on='event_id', validate='one_to_one')
    if not np.allclose(t.net_r, f.set_index('event_id').loc[t.date, 'target_net_return_r']):
        raise ValueError('IS target returns disagree with ledger')
    direction = joined.direction.map({'long': 1, 'short': -1})
    if not (direction == joined.feature_breakout_direction).all():
        raise ValueError('Feature direction disagrees with ledger')
    return joined, metrics


def reconstruct(t, config):
    d = t.direction.map({'long': 1, 'short': -1}).astype(float)
    entry_slip = d * (t.entry_price - t.entry_reference_open)
    exit_slip = d * (t.exit_reference_price - t.exit_price)
    entry_fee = t.entry_price * config['cost_bps'] / 10000
    exit_fee = t.exit_price * config['cost_bps'] / 10000
    gross = d * (t.exit_reference_price - t.entry_reference_open)
    net = gross - entry_slip - exit_slip - entry_fee - exit_fee
    c = pd.DataFrame({'date': t.date, 'direction': t.direction,
                      'entry_reference_price': t.entry_reference_open, 'exit_reference_price': t.exit_reference_price,
                      'entry_adverse_fill': t.entry_price, 'exit_adverse_fill': t.exit_price,
                      'original_units': t.units, 'original_risk_points': t.risk_points,
                      'gross_points': gross, 'net_points': net,
                      'gross_r': gross / t.risk_points, 'net_r': net / t.risk_points,
                      'entry_fee_points': entry_fee, 'exit_fee_points': exit_fee,
                      'entry_slippage_points': entry_slip, 'exit_slippage_points': exit_slip})
    for stem, points in [('gross', gross), ('net', net), ('entry_fee', entry_fee), ('exit_fee', exit_fee),
                         ('entry_slippage', entry_slip), ('exit_slippage', exit_slip)]:
        c[stem + '_pnl'] = points * t.units
        if stem not in ['gross','net']: c[stem + '_r'] = points / t.risk_points
    c['fees_pnl'] = c.entry_fee_pnl + c.exit_fee_pnl
    c['slippage_pnl'] = c.entry_slippage_pnl + c.exit_slippage_pnl
    c['total_cost_pnl'] = c.fees_pnl + c.slippage_pnl
    c['fees_r'] = c.entry_fee_r + c.exit_fee_r
    c['slippage_r'] = c.entry_slippage_r + c.exit_slippage_r
    for reconstructed, original in [('gross_points','gross_points'),('net_points','net_points'),
                                    ('gross_r','gross_r'),('net_r','net_r'),('gross_pnl','gross_pnl'),
                                    ('net_pnl','net_pnl'),('fees_pnl','transaction_cost_pnl'),('slippage_pnl','slippage_pnl')]:
        if not np.allclose(c[reconstructed], t[original], rtol=1e-10, atol=1e-8):
            raise ValueError(f'Reconstruction mismatch: {reconstructed}')
    if not np.allclose(t.entry_price, t.entry_reference_open * (1+d*config['slippage_bps']/10000)):
        raise ValueError('Entry slippage assumption mismatch')
    if not np.allclose(t.exit_price, t.exit_reference_price * (1-d*config['slippage_bps']/10000)):
        raise ValueError('Exit slippage assumption mismatch')
    return c


def cluster_stats(frame, value, key, replicates=REPLICATES, seed=SEED):
    """Percentile bootstrap of whole trading-day clusters, not individual rows."""
    x = frame[value].to_numpy(dtype=float)
    if not len(x):
        return dict(mean=np.nan, std=np.nan, standard_error=np.nan, ci_low=np.nan, ci_high=np.nan, days=0)
    clusters = frame.groupby('date')[value].agg(['sum','count'])
    sums, counts = clusters['sum'].to_numpy(), clusters['count'].to_numpy()
    days = len(clusters)
    mean = float(x.mean())
    se = float(np.sqrt(days / (days - 1) * np.sum((sums - mean*counts)**2)) / len(x)) if days > 1 else np.nan
    # Stable per-analysis streams; changing plotting/report order cannot change CIs.
    code = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, code]))
    boot = []
    for start in range(0, replicates, 250):
        idx = rng.integers(0, days, size=(min(250, replicates-start), days))
        boot.extend(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    lo, hi = np.quantile(boot, [.025,.975])
    return {'mean': mean, 'std': float(x.std(ddof=1)) if len(x)>1 else np.nan,
            'standard_error': se, 'ci_low': float(lo), 'ci_high': float(hi), 'days': days}


def profit_factor(values):
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def summary(g, key):
    n = len(g)
    s = cluster_stats(g, 'net_r', key)
    return {'number_of_trades': n, 'trading_days': s['days'],
            'gross_expectancy_r': float(g.gross_r.mean()) if n else np.nan,
            'net_expectancy_r': s['mean'], 'net_r_ci_low': s['ci_low'], 'net_r_ci_high': s['ci_high'],
            'net_profit_factor': profit_factor(g.net_pnl),
            'net_win_rate_pct': 100*float((g.net_pnl>0).mean()) if n else np.nan,
            'target_pct': 100*float((g.exit_reason=='target').mean()) if n else np.nan,
            'stop_loss_pct': 100*float(g.exit_reason.isin(['stop','stop_gap']).mean()) if n else np.nan,
            'session_exit_pct': 100*float((g.exit_reason=='session_exit').mean()) if n else np.nan,
            'other_exit_pct': 100*float((~g.exit_reason.isin(EXIT_CATEGORIES)).mean()) if n else np.nan,
            'total_net_pnl': float(g.net_pnl.sum()), 'small_sample_n_lt_50': n<50,
            'bootstrap_replicates': REPLICATES, 'bootstrap_seed': SEED}


def quantile_groups(series):
    good = series.dropna()
    labels, edges = pd.qcut(good, q=5, retbins=True, duplicates='drop', labels=False)
    result = pd.Series('Missing', index=series.index, dtype=object)
    for k in range(len(edges)-1):
        result.loc[labels.index[labels==k]] = f'Q{k+1}'
    return result, edges


def conditioned(t):
    rows = []
    for feature in FEATURES:
        labels, edges = quantile_groups(t[feature])
        for k in range(len(edges)-1):
            group = f'Q{k+1}'
            rows.append({'feature':feature, 'group':group, 'lower_bound':float(edges[k]),
                         'upper_bound':float(edges[k+1]), 'boundary_rule':'[lower, upper]' if k==0 else '(lower, upper]',
                         **summary(t[labels==group],feature+group)})
        if (labels=='Missing').any():
            rows.append({'feature':feature,'group':'Missing','lower_bound':np.nan,'upper_bound':np.nan,
                         'boundary_rule':'Unavailable prior history; retained separately',
                         **summary(t[labels=='Missing'],feature+'Missing')})
    # Candle-start clock bins, not the prediction/next-open clock.
    minute = pd.to_datetime(t.signal_candle_start).dt.hour*60+pd.to_datetime(t.signal_candle_start).dt.minute
    for lo, hi, name in [(570,600,'09:30–10:00'),(600,630,'10:00–10:30'),(630,660,'10:30–11:00'),(660,720,'11:00–12:00')]:
        rows.append({'feature':'breakout_timing','group':name,'lower_bound':lo,'upper_bound':hi,
                     'boundary_rule':'[lower, upper) IST signal-candle start minute-of-day',
                     **summary(t[minute.between(lo,hi-1)],'timing'+name)})
    for direction in ['long','short']:
        rows.append({'feature':'breakout_direction','group':direction,'lower_bound':np.nan,'upper_bound':np.nan,
                     'boundary_rule':'Signal direction', **summary(t[t.direction==direction],'direction'+direction)})
    result = pd.DataFrame(rows)
    if not (result.groupby('feature').number_of_trades.sum()==len(t)).all():
        raise ValueError('Feature partition totals fail reconciliation')
    if not np.allclose(result.groupby('feature').total_net_pnl.sum(),t.net_pnl.sum()):
        raise ValueError('Feature partition P&L mismatch')
    return result


def exit_distribution(t):
    rows = []
    categories = list(dict.fromkeys([*EXIT_CATEGORIES,*sorted(t.exit_reason.unique())]))
    for reason in categories:
        g = t[t.exit_reason==reason]
        rows.append({'exit_category':reason,'trade_count':len(g),'percentage_of_trades':100*len(g)/len(t),
                     'average_gross_r':g.gross_r.mean(),'average_net_r':g.net_r.mean(),
                     'total_net_pnl':g.net_pnl.sum(), 'net_winning_trades':int((g.net_pnl>0).sum()),
                     'negative_net_pnl_sum':g.loc[g.net_pnl<0,'net_pnl'].sum()})
    return pd.DataFrame(rows)


def plot_charts(t,c,groups,exits,root):
    folder = root/'orb_diagnostic_charts'
    folder.mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax = plt.subplots(figsize=(9,5))
    vals = [c.gross_r.mean(),-c.fees_r.mean(),-c.slippage_r.mean(),c.net_r.mean()]
    ax.bar(['Gross R','Fees drag','Slippage drag','Net R'],vals,color=['#3478a4','#df913c','#d65d58','#75589c'])
    ax.axhline(0,color='black',lw=.8)
    for i,v in enumerate(vals): ax.annotate(f'{v:.4f}R',(i,v),xytext=(0,6 if v>=0 else -16),textcoords='offset points',ha='center')
    ax.set(ylabel='Mean R per trade (original risk denominator)',title='IS: same realized paths, cost decomposition')
    fig.tight_layout(); fig.savefig(folder/'cost_expectancy.png',dpi=180); plt.close(fig)
    features = [*FEATURES,'breakout_timing','breakout_direction']
    fig,axes = plt.subplots(3,3,figsize=(17,14))
    for ax,feature in zip(axes.flat,features):
        g = groups[groups.feature==feature].reset_index(drop=True)
        x = np.arange(len(g))
        ax.plot(x,g.gross_expectancy_r,'o--',label='Gross mean R',color='#3478a4')
        ax.scatter(x,g.net_expectancy_r,label='Net mean R',color='#c65353')
        ax.vlines(x,g.net_r_ci_low,g.net_r_ci_high,color='#c65353',lw=2,label='Net 95% day-bootstrap CI')
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(x,[f'{row.group}\nn={row.number_of_trades}' for row in g.itertuples()],rotation=25 if feature=='breakout_timing' else 0)
        ax.set_title(FEATURES.get(feature,feature.replace('_',' ').title()),fontsize=10)
        ax.set_ylabel('Mean R')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('IS exploratory subgroups: unadjusted pointwise CIs; no filter selection',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(folder/'feature_expectancy.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    active = exits[exits.trade_count>0]
    axes[0].bar(active.exit_category,active.percentage_of_trades,color='#3478a4')
    axes[0].set(title='IS exit frequencies',ylabel='% of trades')
    axes[1].bar(active.exit_category,active.total_net_pnl,color=['#3478a4' if v>=0 else '#c65353' for v in active.total_net_pnl])
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(title='IS net P&L by exit',ylabel='INR proxy P&L')
    fig.tight_layout(); fig.savefig(folder/'exit_distribution.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,feature in zip(axes,['breakout_direction','breakout_timing']):
        g=groups[groups.feature==feature]
        bottom=np.zeros(len(g))
        for col,label,color in [('target_pct','Target','#3478a4'),('stop_loss_pct','Stop incl. gaps','#c65353'),('session_exit_pct','Session exit','#d6a13b')]:
            ax.bar(g.group,g[col],bottom=bottom,label=label,color=color)
            bottom+=g[col].to_numpy()
        ax.set(ylabel='% of group trades',title=feature.replace('_',' ').title())
        ax.tick_params(axis='x',rotation=20)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'exit_conditions.png',dpi=180); plt.close(fig)


def markdown_table(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)): return 'N/A' if pd.isna(v) else f'{v:.6f}'
        return str(v)
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def report(t,c,g,e,metrics,hashes,root):
    gross_stats=cluster_stats(t,'gross_r','overall_gross')
    net_stats=cluster_stats(t,'net_r','overall_net')
    profits=c.loc[c.gross_pnl>0,'gross_pnl'].sum()
    consumed=100*c.total_cost_pnl.sum()/profits
    net_surplus_consumed=100*c.total_cost_pnl.sum()/c.gross_pnl.sum()
    rows=[]
    for name,values in [('Gross',c.gross_points),('Net',c.net_points)]:
        isgross=name=='Gross'
        pnl=c.gross_pnl if isgross else c.net_pnl
        stats=gross_stats if isgross else net_stats
        rows.append({'basis':name,'mean_points':values.mean(),'mean_R':stats['mean'],'std_R':stats['std'],
                     'SE_R':stats['standard_error'],'CI95_low_R':stats['ci_low'],'CI95_high_R':stats['ci_high'],
                     'profit_factor_currency':profit_factor(pnl),'profit_factor_points':profit_factor(values),
                     'win_rate_pct':100*(values>0).mean(),'total_PnL_INR':pnl.sum()})
    positive=g[(g.number_of_trades>0)&(g.net_expectancy_r>0)]
    session=t[t.exit_reason=='session_exit']
    all_losses=-t.loc[t.net_pnl<0,'net_pnl'].sum()
    session_losses=-session.loc[session.net_pnl<0,'net_pnl'].sum()
    stop=t[t.exit_reason.isin(['stop','stop_gap'])]
    directions=g[g.feature=='breakout_direction'].set_index('group')
    timing=g[g.feature=='breakout_timing'].reset_index(drop=True)
    width=g[(g.feature=='feature_or_width_pct')&(g.group!='Missing')].reset_index(drop=True)
    cost_rows=[]
    for stem in ['entry_fee','exit_fee','entry_slippage','exit_slippage']:
        cost_rows.append({'component':stem,'mean_points':c[stem+'_points'].mean(),'mean_R':c[stem+'_r'].mean(),
                          'mean_INR':c[stem+'_pnl'].mean(),'total_INR':c[stem+'_pnl'].sum()})
    text=['# ORB V1: IS-only diagnostic analysis','',
          'Scope: 1,624 original executed IS trades (2015–2021). No OOS outcomes/files were inspected. No rules, data exclusions, parameters, fills, or existing result files were changed; no model was fitted and no filters were selected.', '',
          '## Reconstruction and reconciliation','',
          'Gross points = direction × (exit reference price − entry reference open). Entry slippage = direction × (entry adverse fill − entry reference open); exit slippage = direction × (exit reference price − exit adverse fill). Fees are calculated separately on each original adverse fill at the recorded per-side bps. Net points = gross points − both slippage components − both fees. Multiply by the original units for INR proxy P&L.',
          'Original position sizing, adverse-fill entry-to-stop risk denominator, target level, realized entry/exit bars and exit categories are held fixed. Gross R uses that same original risk denominator. Therefore gross target R can exceed 1.5R after entry slippage is removed. This removes friction consistently from realized prices; it is NOT a rerun of a hypothetical frictionless strategy with changed target/stop paths or changed units. Gross profit factor is primarily reported in currency under original sizing, with points profit factor also provided.',
          f"Recorded config: `{json.dumps(metrics['config'],sort_keys=True)}`. All reconstructed per-trade points, R, currency P&L, fees, and slippage match the IS ledger (rtol 1e-10; atol 1e-8). Total net P&L reconciles to INR {c.net_pnl.sum():,.8f}, versus existing metric INR {metrics['net_pnl']:,.8f}.", '',
          markdown_table(pd.DataFrame(rows)), '',
          markdown_table(pd.DataFrame(cost_rows)), '',
          f'Total transaction fees: INR {c.fees_pnl.sum():,.6f}; total adverse slippage: INR {c.slippage_pnl.sum():,.6f}; combined: INR {c.total_cost_pnl.sum():,.6f}. Average combined entry cost (fee + slippage): {float((c.entry_fee_points+c.entry_slippage_points).mean()):.6f} points / INR {float((c.entry_fee_pnl+c.entry_slippage_pnl).mean()):.6f}; average combined exit cost: {float((c.exit_fee_points+c.exit_slippage_points).mean()):.6f} points / INR {float((c.exit_fee_pnl+c.exit_slippage_pnl).mean()):.6f}.',
          f'Costs equal {consumed:.4f}% of gross trading profits, defined as the sum of positive gross INR trades (INR {profits:,.6f}). As a separate denominator, costs equal {net_surplus_consumed:.4f}% of the aggregate gross net-of-losses surplus (INR {c.gross_pnl.sum():,.6f}); do not confuse these two percentages.', '',
          '![Cost decomposition](orb_diagnostic_charts/cost_expectancy.png)', '',
          '## Statistical uncertainty','',
          f'Trading-day cluster percentile bootstrap: {REPLICATES:,} replicates; fixed seed {SEED}; independent deterministic streams keyed by analysis label. Each draw samples whole traded-day clusters with replacement, retaining every trade from each sampled day; the statistic is total sampled R divided by sampled trade count. Overall and each subgroup are resampled within their own traded-day populations. There is one trade per traded day in V1, so the cluster bootstrap numerically reduces to a trade bootstrap here. Non-traded days are excluded from per-trade expectancy, not assigned artificial zero-R trades.',
          'Standard deviation is sample trade-R standard deviation. Standard error uses the day-cluster sandwich formula sqrt[D/(D−1) × Σ(day_sum_R − mean_R × day_trade_count)²] / total_trade_count, which equals sample SD / sqrt(n) for this ledger. Intervals assume resampled trading days are exchangeable: they do not account for serial dependence, changing regimes, prior baseline selection, or retrospective full-session data-quality selection. Subgroup CIs are unadjusted pointwise 95% intervals, not simultaneous intervals or formal tests of differences. Many overlapping feature partitions are inspected; multiple comparisons and selection risk prevent treating the best subgroup as an established edge. CI overlap/non-overlap does not prove equality/difference.', '',
          '## Feature-conditioned performance','',
          'All seven continuous features use five IS-only value quantiles (pandas qcut), with right-closed boundaries except the first includes its minimum. Ties stay together; counts may differ slightly. If ties collapse edges, fewer bins are flagged rather than inventing rank-based value cuts. Missing historical features are reported as a separate Missing group and included in reconciliation. Percentages/returns/volatility are decimal fractions in boundaries, not percentages. Boundaries in the CSV retain full floating-point precision; report values are rounded.',
          'Breakout timing uses signal-candle START in IST: [09:30,10:00), [10:00,10:30), [10:30,11:00), [11:00,12:00). Prediction occurs one minute later, so a 09:59 candle belongs to the first group although its close/entry clock is 10:00. Long/short is signal direction. The supplied features were available at prediction time; no fill, target, future bar or outcome field defines a group. ATR/volatility and prior-day features retain the existing lagged-history construction; no imputation or feature transformation is fitted.',
          'Each feature partition separately totals 1,624 trades and the same total net P&L; partitions overlap and must not be summed across features. Win rate is net-positive trades. Stop frequency includes normal stops and stop gaps. Small samples are flagged at n < 50 (a reporting flag, not a filter).', '',
          '![Exploratory feature means and CIs](orb_diagnostic_charts/feature_expectancy.png)', '']
    for feature in [*FEATURES,'breakout_timing','breakout_direction']:
        part=g[g.feature==feature]
        text += [f'### {FEATURES.get(feature,feature.replace("_"," ").title())}', '',
                 markdown_table(part[['group','lower_bound','upper_bound','number_of_trades','gross_expectancy_r','net_expectancy_r','net_r_ci_low','net_r_ci_high','net_profit_factor','net_win_rate_pct','target_pct','stop_loss_pct','session_exit_pct','small_sample_n_lt_50']]), '']
    text += ['## Exit distributions','',markdown_table(e),'',
             '![Exit distribution](orb_diagnostic_charts/exit_distribution.png)', '',
             '![Exit frequencies by direction and timing](orb_diagnostic_charts/exit_conditions.png)', '',
             f'Session-end exits: {len(session)} trades ({100*len(session)/len(t):.4f}%); net mean {session.net_r.mean():.6f}R; total INR {session.net_pnl.sum():,.6f}. They account for {100*session_losses/all_losses:.4f}% of all negative-trade INR loss magnitude. Their share of loss magnitude can be compared with their trade share, but this is descriptive, not a causal attribution.',
             f"Session exits therefore {'do not' if session_losses/all_losses<len(session)/len(t) else 'do'} disproportionately account for realized negative-trade loss magnitude on this comparison; their aggregate contribution is {'positive' if session.net_pnl.sum()>0 else 'negative'}. Averaging R equally across trades and summing sized currency P&L are different weightings; both are retained.",
             f'Stop/stop-gap exits: {len(stop)} trades ({100*len(stop)/len(t):.4f}%); total net INR {stop.net_pnl.sum():,.6f}. Target exits: {int((t.exit_reason=="target").sum())} ({100*(t.exit_reason=="target").mean():.4f}%). Thus {100*(t.exit_reason!="target").mean():.4f}% did not reach the target while their positions remained open. Session exits quantify positions still open without hitting target/stop by the forced 15:29 open. No post-stop or post-session counterfactual path was examined; failure to exit at target is not proof that a stopped position would never have touched the target later that day.',
             f"Longs reach targets on {directions.loc['long','target_pct']:.4f}% of trades and stops on {directions.loc['long','stop_loss_pct']:.4f}%; shorts reach targets on {directions.loc['short','target_pct']:.4f}% and stops on {directions.loc['short','stop_loss_pct']:.4f}%. Long net expectancy is {directions.loc['long','net_expectancy_r']:.6f}R versus short {directions.loc['short','net_expectancy_r']:.6f}R. These are descriptive differences, not tests of a direction effect.",
             f"Target frequency declines across timing groups from {timing.iloc[0].target_pct:.4f}% in the first bin to {timing.iloc[-1].target_pct:.4f}% in the last; session-exit frequency rises from {timing.iloc[0].session_exit_pct:.4f}% to {timing.iloc[-1].session_exit_pct:.4f}%. Later groups have fewer trades and wider uncertainty. These associations do not establish causality or justify timing filters.",
             'Direction and timing tables and charts use future outcomes only as retrospective labels, not to construct predictors.', '',
             '## Explicit diagnostic conclusions','',
             f'1. **Gross expectancy:** {gross_stats["mean"]:.6f}R / {c.gross_points.mean():.6f} points per trade, descriptively {"positive" if gross_stats["mean"]>0 else "negative"}. Its day-bootstrap 95% CI is [{gross_stats["ci_low"]:.6f}, {gross_stats["ci_high"]:.6f}]R; {"it includes zero, so these assumptions do not establish positive population expectancy" if gross_stats["ci_low"]<=0<=gross_stats["ci_high"] else "it excludes zero under the stated exchangeable-day bootstrap assumptions"}.',
             f'2. **Friction drag:** fees remove {c.fees_r.mean():.6f}R and slippage removes {c.slippage_r.mean():.6f}R per trade, total {c.fees_r.mean()+c.slippage_r.mean():.6f}R. Net expectancy is {net_stats["mean"]:.6f}R, with 95% CI [{net_stats["ci_low"]:.6f}, {net_stats["ci_high"]:.6f}]R. This is an exact ledger decomposition, not an estimate of real futures trading costs.',
             f'3. **Concentration:** {int(((g.number_of_trades>0)&(g.net_expectancy_r<0)).sum())} of {int((g.number_of_trades>0).sum())} reported nonempty groups have negative mean net R, including both directions and all timing bins. Losses are broadly distributed across the observed partitions, though their severity varies. In relative OR-width bins the narrowest mean net R is {width.iloc[0].net_expectancy_r:.6f} versus {width.iloc[-1].net_expectancy_r:.6f} for the widest. These are overlapping partitions; differences do not establish independent causal effects or validate choosing filters.',
             f'4. **Positive IS subgroups:** {len(positive)} reported groups have positive mean net R. '+('The table below lists them; assess sample sizes and uncertainty, not just signs.' if len(positive) else 'None of the reported groups has positive mean net R.'),
             f'5. **Exit driver:** stops including gaps contribute INR {stop.net_pnl.sum():,.6f}; session-end exits contribute INR {session.net_pnl.sum():,.6f}. The distribution table identifies the dominant negative contribution; frequency and P&L contribution are separate concepts.',
             '6. **ML classification:** yes, 1,624 labeled IS events and descriptive variation in target/stop frequencies are enough to justify a limited exploratory feasibility investigation; they are not evidence that classification will recover a cost-adjusted edge. A useful classifier must improve decisions after substantial friction and be validated in chronological, nested IS folds with train-only preprocessing and controls for selection. Classification accuracy alone is insufficient. The present partitions neither prove nor disprove nonlinear predictability; no models or filters were tested. There is no demonstrated cost-adjusted edge to justify claiming ML profitability.', '']
    if len(positive): text += [markdown_table(positive[['feature','group','number_of_trades','net_expectancy_r','net_r_ci_low','net_r_ci_high']]),'']
    text += ['## Reproduction and limitations','',
             '`python orb_diagnostics.py` then `python -m unittest orb_diagnostic_tests -v`. Required packages are the existing pandas/numpy environment plus matplotlib (used version '+matplotlib.__version__+'). The script opens only IS trades, IS features and IS metrics. Pipeline, audit and cleaned source are checksummed as protected files without processing their contents; no OOS file is opened. The source is not needed for reconstruction because reference prices, risk and units are present. No required field was unreconstructable.',
             'All existing protected files are SHA-256 checked before and after diagnostics. Existing baseline costs, slippage, exclusion/signal/exit rules and result files are untouched. The retrospective eligibility screen and exchangeable-day bootstrap assumptions limit generalization. Index-price P&L remains the original futures-proxy accounting illustration, not executable futures evidence.', '',
             markdown_table(pd.DataFrame([{'protected_input':name,'sha256':h} for name,h in hashes.items()])), '']
    (root/'orb_diagnostics_is.md').write_text('\n'.join(text),encoding='utf-8')
    return {'gross':gross_stats,'net':net_stats,'positive_groups':len(positive),'cost_consumption_pct':float(consumed)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='.')
    args=parser.parse_args()
    root=Path(args.project)
    hashes={name:digest(root/name) for name in PROTECTED}
    t,metrics=load_is(root)
    c=reconstruct(t,metrics['config'])
    if not np.isclose(c.net_pnl.sum(),metrics['net_pnl'],rtol=0,atol=1e-6):
        raise ValueError('IS total net P&L does not reconcile')
    g=conditioned(t)
    e=exit_distribution(t)
    if e.trade_count.sum()!=1624 or not np.isclose(e.total_net_pnl.sum(),metrics['net_pnl']):
        raise ValueError('Exit totals do not reconcile')
    c.to_csv(root/'orb_cost_decomposition.csv',index=False)
    g.to_csv(root/'orb_feature_conditioned_is.csv',index=False)
    e.to_csv(root/'orb_exit_distribution_is.csv',index=False)
    plot_charts(t,c,g,e,root)
    result=report(t,c,g,e,metrics,hashes,root)
    for name,h in hashes.items():
        if digest(root/name)!=h: raise RuntimeError('Protected input changed: '+name)
    print(json.dumps(result,indent=2))
    print('IS-only diagnostics complete; all protected inputs unchanged.')


if __name__=='__main__': main()
