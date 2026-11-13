"""Run the fixed ten-multiplier IS experiment and chronological classifiers."""
from pathlib import Path
import importlib.util
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit
import strategy as s

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
spec=importlib.util.spec_from_file_location('v1_readonly',ROOT/'orb_logistic_v1/train.py')
v1=importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)


def calendar_folds(dates):
    folds=[]
    for i,(train,test) in enumerate(TimeSeriesSplit(n_splits=5).split(dates),1):
        folds.append({'fold':i,'train_end':dates[train[-1]],'test_start':dates[test[0]],'test_end':dates[test[-1]]})
    return folds


def ml(t,folds,tag):
    X=t[s.FEATURES]; y=t.target_hit
    if not np.isfinite(X.to_numpy()).all(): raise ValueError('Missing signal-time feature')
    predictions=[]; definitions=[]
    for fold in folds:
        boundary=pd.Timestamp(fold['test_start']+' 00:00',tz='Asia/Kolkata')
        train=(t.date<=fold['train_end']) & (pd.to_datetime(t.label_end_timestamp)<boundary)
        test=t.date.between(fold['test_start'],fold['test_end'])
        counts={'train_n':int(train.sum()),'test_n':int(test.sum()),'train_positive':int(y[train].sum()),'test_positive':int(y[test].sum())}
        if y[train].nunique()!=2 or not test.any():
            definitions.append({**fold,**counts,'status':'insufficient_classes_or_events'}); continue
        model=v1.make_model()
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(X[train],y[train])
        p=model.predict_proba(X[test])[:,1]; baseline=float(y[train].mean())
        g=t.loc[test,['date','prediction_timestamp','gross_r','net_r','fees_r','slippage_r','target_hit']].copy()
        g['probability']=p; g['baseline_probability']=baseline; g['fold']=fold['fold']
        predictions.append(g)
        definitions.append({**fold,**counts,'status':'scored','training_prevalence':baseline,
                            'latest_training_label_end':t.loc[train,'label_end_timestamp'].max(),
                            'model':v1.scores(y[test],p),'baseline':v1.scores(y[test],np.full(test.sum(),baseline))})
        joblib.dump(model,OUT/f'{tag}_fold_{fold["fold"]}.joblib')
    oof=pd.concat(predictions,ignore_index=True)
    oof.to_csv(OUT/f'{tag}_oof.csv',index=False)
    result={'folds':definitions,'model':v1.scores(oof.target_hit,oof.probability),
            'baseline':v1.scores(oof.target_hit,oof.baseline_probability)}
    tables=[]
    for q,name in [(5,'quintiles'),(10,'calibration')]:
        labels,edges=pd.qcut(oof.probability,q,labels=False,retbins=True,duplicates='drop')
        rows=[]
        for k in range(len(edges)-1):
            g=oof[labels==k]
            if g.empty: continue
            rows.append({'multiplier':t.multiplier.iloc[0],'group':k+1,'lower_probability':edges[k],
                         'upper_probability':edges[k+1],'n':len(g),'mean_probability':g.probability.mean(),
                         'target_rate':g.target_hit.mean(),'mean_gross_r':g.gross_r.mean(),
                         'mean_net_r':g.net_r.mean(),'mean_cost_r':(g.fees_r+g.slippage_r).mean()})
        pd.DataFrame(rows).to_csv(OUT/f'{tag}_{name}.csv',index=False)
        tables.append(pd.DataFrame(rows))
    final=v1.make_model()
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning); final.fit(X,y)
    joblib.dump(final,OUT/f'{tag}_final.joblib')
    return result,tables[0]


def report(comparison,breakdown,mlmetrics,quintiles,audit,folds):
    primary=comparison[comparison['mode']=='next_open']
    lines=['# NIFTY ORB + supplied Vervoort ATR: IS research','',
           'IS only, 2015–2021. Ten explicitly requested multipliers; ATR period 5 and OR target multiplier 1.5 fixed. No best configuration selected, no OOS outcomes evaluated, no probability trading filters. Existing files remain unchanged.', '',
           '## Data and Pine recurrence','',
           f"Processed {audit['is_minute_rows']:,} IS minute rows; {audit['eligible_sessions']} complete regular sessions. Evening rows excluded in memory: {audit['outside_regular_rows']}. Zero volume unavailable, never used. Sessions with missing/duplicate/invalid minutes are rejected. Complete five-minute bars on rejected sessions still update indicators; incomplete bars do not. Missing whole days cannot be checked without a calendar.",
           'Start-labeled five-minute candles aggregate exactly five unique IST minute starts: 09:15–09:19 → 09:15, 09:20–09:24 → 09:20, etc. OR uses 09:15,09:20,09:25 starts. First signal candle starts 09:30 and confirms at 09:35. Final eligible signal starts 14:50 and confirms at 14:55; the 14:55 candle is not an entry signal. Square-off occurs at the 14:55 candle close, 15:00.',
           'ATR is Wilder RMA of true range, seeded with the first five completed-bar TR values. First TR uses high−low; later TR includes previous close, including overnight gaps. ATR/trail/trend continue across days and through regular candles after 15:00. No evening candles contribute. Before ATR initialization trail is NaN; each previous missing trail is nz(...,0). Trend starts at 0 and changes only under the supplied strict crossing conditions.',
           'For closes above previous trail on consecutive bars, trail=max(previous trail,close−mult×ATR). For consecutive closes below, trail=min(previous trail,close+mult×ATR). Otherwise trail=close−loss if close>previous trail, else close+loss. Trend uses previous close and current close against the same previous stop, strictly; equality does not cross.', '',
           '## Execution and parity limits','',
           'pine_close mode fills at signal close; next_open mode (primary) fills at the next five-minute open. Both include 5 bps fees per side and 1 bp adverse slippage per fill. Targets remain signal close ±1.5×OR width in both modes; actual entry-to-initial-stop R is recomputed. Target distance in R varies with the ATR stop and entry gap; this is not fixed 1.5R.',
           'No entry-bar high/low from the completed signal candle can trigger a newly created close-fill bracket. Subsequent candles test the previously active stop before considering the current finalized ATR trail. Stops precede targets when both touched; stop gaps fill at open. Target fills receive no favorable gap improvement. If the next-open entry already exceeds its fixed target, a target-gap round trip exits at that open reference price and pays both slippage/fees. Stops invalid versus next reference open or adverse fill are rejected, logged, and consume the daily order attempt.',
           'A surviving position updates its stop at each completed candle, only when trail remains on the proper side of close and only favorably; the update becomes active for the next candle. Existing stop/target orders remain active through 14:55–15:00; if neither triggers, square-off uses the 15:00 close. Exit timestamps for intrabar fills identify their containing candle, with conservative label-end at candle completion.',
           'Pine parity here means supplied indicator/signal recurrence and signal-close entry timing. It does not claim exact TradingView broker-emulator transaction parity: requested conservative stop-first OHLC handling differs from TradingView historical price-path assumptions, and costs, next-open rejection and full-session exclusions are explicitly imposed. No TradingView export was provided for numerical cross-platform reconciliation. Official semantics: [TradingView strategies](https://www.tradingview.com/pine-script-docs/concepts/strategies/) and [Pine reference](https://www.tradingview.com/pine-script-reference/v6/#fun_ta.atr).', '',
           '## Capital and metrics','',
           'Each configuration/mode starts INR 1,000,000; fractional proxy units = INR 100,000 / adverse entry fill (fixed 10% initial-capital allocation). No compounding sizing. Daily equity=capital+cumulative net P&L. Daily return=net P&L/prior observed regular-session equity; rejected/no-trade IS days are zero. Sharpe uses sample daily SD, sqrt(252), risk-free 0. Drawdown uses daily closing equity with initial capital included. Long/short metrics are independent hypothetical subledgers using the same dates/capital, with opposite-direction days flat. Index-price fills are not executable futures evidence.',
           'Gross is reference-price P&L on the same realized path, units and actual risk denominator; adverse slippage and fees reconcile to net. Gross/net PF use currency; win rates and exit types differ. Average cost-to-risk averages per-trade (fees+slippage)/actual initial risk. Small initial risk can produce very large R/cost-to-risk; no outlier clipping or risk filter is introduced.', '',
           '## Primary next-open comparison','',v1.md_table(primary),'',
           'Pine-close mode metrics are separately in atr_multiplier_comparison.csv and distinct ledgers. Long/short metrics for both modes are in long_short_breakdown.csv. Daily accounts and ledgers are retained for every configuration.', '',
           '## ML protocol','',
           'One classifier per multiplier, labeled from its primary next-open backtest, not original ORB/V1 targets. Original seven current-session prediction-time feature definitions are adapted to five-minute confirmation (including clock +5 minutes); historical missing features remain excluded. Added signal-time ATR points/price, trail distance/ATR, signal-close initial risk/price, and stored ATR trend state. All are computed at confirmation; no actual entry fill/risk, future trail, realized R, target_hit or exit enters X.',
           'StandardScaler + LogisticRegression(C=1,class_weight=None), default regularization/solver, fixed max_iter=1000 and seed. Scaling and estimator fit only each training fold. Trail-distance/ATR is deterministically rounded to 12 decimal places to prevent scaling floating-point cancellation noise in mathematically constant features; no outcome-based feature selection. Five shared chronological calendar-date test windows, constructed once from observed IS regular dates, ensure common evaluation periods despite different eligible event sets. Latest label-end must precede test-window midnight. Nonoverlapping intraday labels need no additional embargo. Class counts and skipped-fold status are saved; all scored outcomes stay IS-only.', '',
           'Invalid initial-stop rejection uses the eventual entry fill and restricts the supervised dataset to executed trades. This is an execution-dependent sample selection limitation, not a prediction-time feature. Initial indicator state starts at the available IS history (January 2015), with no unavailable prehistory inferred.', '',
           v1.md_table(pd.DataFrame(folds)),'',
           v1.md_table(pd.DataFrame([{'multiplier':m,**{k:r['model'][k] for k in ['n','positive_prevalence','roc_auc','pr_auc_average_precision','log_loss','brier_score','balanced_accuracy_threshold_0_5']},
                                     'baseline_log_loss':r['baseline']['log_loss'],'baseline_brier':r['baseline']['brier_score']} for m,r in mlmetrics.items()])), '',
           'Primary PR-AUC is average precision; trapezoidal PR area is separately recorded. Each constant baseline uses training prevalence. Calibration deciles and probability quintiles are configuration-specific OOF bins, with boundaries/counts retained. Quintiles summarize gross/net R retrospectively; they do not select trading thresholds or claim executable profitability. Same calendar windows are not identical trade populations; configuration changes both qualifying trades and labels.', '',
           '## Multiple testing and conclusions','',
           'Ten requested configurations and many probability bins were examined. All are reported; no maximum selected as a validated edge. Reported performance and model comparisons are descriptive point estimates, without multiplicity-adjusted hypothesis testing. A future inferential claim would require predeclared tests, dependence-aware uncertainty and familywise/FDR control; that has not been performed. IS results cannot establish out-of-sample or futures profitability.',
           f"Primary net expectancy is positive for {int((primary.net_expectancy_r>0).sum())} of ten configurations; baseline log loss is beaten by {sum(r['model']['log_loss']<r['baseline']['log_loss'] for r in mlmetrics.values())} classifiers. These counts are descriptive, not evidence from independent tests.", '',
           '## Reproduction and checks','',
           '`python -m unittest discover -s orb_vervoort_atr_research -p "test_*.py" -v`, then `python orb_vervoort_atr_research/experiment.py`. Input checksum and reconciliation results are saved. Tests cover boundary alignment, Wilder initialization, Vervoort recurrence/trend/equality, next-open versus close fills, gap rejection, stop-first collisions, trailing-stop timing, square-off and future-data invariance. Saved results reconcile points, currency, R, direction counts, exit rates, daily account equity, feature/label alignment and folds. No external platform export means cross-platform numerical parity remains unverified.']
    (OUT/'research_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    source=ROOT/'NIFTY_50_minute.csv'; source_hash=s.digest(source)
    bars,eligible,audit=s.load_is(source)
    bars['atr']=s.wilder_atr(bars)
    dates=[r['date'] for r in audit['sessions']]
    folds=calendar_folds(dates)
    comparisons=[]; directions=[]; mlmetrics={}; quintiles=[]; rejected=[]; checks=[]
    for mult in s.MULTIPLIERS:
        tag=f'atr_{mult:.1f}'
        for mode in ['next_open','pine_close']:
            trades,skips=s.backtest(bars,mult,mode); rejected.extend(skips)
            trades.to_csv(OUT/f'{tag}_{mode}_trades.csv',index=False)
            m,daily=s.metrics(trades,dates)
            comparisons.append({'multiplier':mult,'mode':mode,**m})
            daily.to_csv(OUT/f'{tag}_{mode}_daily.csv')
            for direction in ['long','short']:
                dm,_=s.metrics(trades[trades.direction==direction],dates)
                directions.append({'multiplier':mult,'mode':mode,'direction':direction,**dm})
            if not trades.date.is_unique: raise ValueError('Multiple daily trades')
            np.testing.assert_allclose(trades.gross_pnl-trades.fees_pnl-trades.slippage_pnl,trades.net_pnl,atol=1e-7)
            np.testing.assert_allclose(trades.gross_r-trades.fees_r-trades.slippage_r,trades.net_r,atol=1e-8)
            if not trades.date.between('2015-01-01','2021-12-31').all(): raise ValueError('Non-IS trade')
            if not np.isclose(daily.equity.iloc[-1],s.CAPITAL+trades.net_pnl.sum()): raise ValueError('Account reconciliation')
            checks.append({'multiplier':mult,'mode':mode,'trades':len(trades),'net_pnl':float(trades.net_pnl.sum()),'passed':True})
            if mode=='next_open':
                mm,q=ml(trades,folds,tag); mlmetrics[str(mult)]=mm; quintiles.append(q)
        print(f'Completed ATR multiplier {mult:.1f}',flush=True)
    comparison=pd.DataFrame(comparisons); breakdown=pd.DataFrame(directions)
    comparison.to_csv(OUT/'atr_multiplier_comparison.csv',index=False)
    breakdown.to_csv(OUT/'long_short_breakdown.csv',index=False)
    pd.concat(quintiles,ignore_index=True).to_csv(OUT/'probability_quintiles.csv',index=False)
    if source_hash!=s.digest(source): raise ValueError('Source changed')
    audit['source_sha256']=source_hash; audit['source_unchanged']=True; audit['execution_rejections']=rejected
    for name,obj in [('data_audit.json',audit),('ml_metrics.json',mlmetrics),('fold_windows.json',folds),
                     ('reconciliation_checks.json',checks),('features.json',{'selected':s.FEATURES,'timing':'Completed signal only, before fill','seed':v1.SEED})]:
        (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    report(comparison,breakdown,mlmetrics,quintiles,audit,folds)
    print(comparison[comparison['mode']=='next_open'][['multiplier','trades','net_expectancy_r','total_net_pnl','daily_sharpe']].to_string(index=False))


if __name__=='__main__': main()
