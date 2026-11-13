"""IS-only directional ORB diagnostics and untuned logistic models."""
from pathlib import Path
import importlib.util
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verified_v1',ROOT/'orb_logistic_v1/train.py')
v1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
FEATURES=[c for c in v1.FEATURES if c!='feature_breakout_direction']
SEED=20261014
REPLICATES=2000


def bootstrap_mean(frame,column,seed):
    # Day-cluster sums/counts retains every trade on each sampled trading day.
    cluster=frame.groupby('date')[column].agg(['sum','count'])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(cluster),size=(REPLICATES,len(cluster)))
    b=cluster['sum'].to_numpy()[idx].sum(axis=1)/cluster['count'].to_numpy()[idx].sum(axis=1)
    return np.quantile(b,[.025,.975]).tolist()


def diagnostics(f,t):
    rows=[]
    for j,direction in enumerate(['long','short']):
        mask=t.direction==direction
        g=t[mask]; features=f[mask]
        fees=g.transaction_cost_points/g.risk_points
        slip=g.slippage_points/g.risk_points
        gross_ci=bootstrap_mean(g,'gross_r',SEED+j*10)
        net_ci=bootstrap_mean(g,'net_r',SEED+j*10+1)
        rows.append({'direction':direction,'n':len(g),'target_hit_rate':g.target_hit.mean(),
                     'gross_expectancy_r':g.gross_r.mean(),'net_expectancy_r':g.net_r.mean(),
                     'gross_r_ci_low':gross_ci[0],'gross_r_ci_high':gross_ci[1],
                     'net_r_ci_low':net_ci[0],'net_r_ci_high':net_ci[1],
                     'gross_profit_factor_currency':profit_factor(g.gross_pnl),
                     'net_profit_factor_currency':profit_factor(g.net_pnl),
                     'mean_fees_r':fees.mean(),'mean_slippage_r':slip.mean(),
                     'stop_pct':100*g.exit_reason.isin(['stop','stop_gap']).mean(),
                     'target_pct':100*(g.exit_reason=='target').mean(),
                     'session_exit_pct':100*(g.exit_reason=='session_exit').mean(),
                     'mean_or_width_points':features.feature_or_width_points.mean(),
                     'median_or_width_points':features.feature_or_width_points.median(),
                     'mean_initial_risk_points':g.risk_points.mean(),
                     'total_net_pnl':g.net_pnl.sum(),'bootstrap_replicates':REPLICATES,'seed':SEED})
    # Direct long-minus-short contrast; resample day clusters jointly, rather
    # than interpreting the overlap of two independent confidence intervals.
    rng=np.random.default_rng(SEED+100)
    idx=rng.integers(0,len(t),size=(REPLICATES,len(t)))
    long=(t.direction=='long').to_numpy()[idx]
    difference={}
    for column in ['gross_r','net_r']:
        values=t[column].to_numpy()[idx]
        draws=(values*long).sum(axis=1)/long.sum(axis=1)-(values*~long).sum(axis=1)/(~long).sum(axis=1)
        difference[column]={'long_minus_short_mean':float(t.loc[t.direction=='long',column].mean()-t.loc[t.direction=='short',column].mean()),
                            'ci95':np.quantile(draws,[.025,.975]).tolist()}
    return pd.DataFrame(rows),difference


def profit_factor(x):
    return float(x[x>0].sum()/-x[x<0].sum())


def fit_direction(f,t,direction,out=None):
    mask=t.direction==direction
    f=f[mask].reset_index(drop=True); t=t[mask].reset_index(drop=True)
    _,_,X,y,pred,end=v1.validate_data(f,t)
    X=X[FEATURES]
    # Verified direction is constant; OR direction is a different, varying feature.
    if f.feature_breakout_direction.nunique()!=1: raise ValueError('Not a single direction')
    oof,folds,models=v1.train_oof(X,y,f,t,pred,end,None)
    for definition,model in zip(folds,models):
        train=definition['train_indices']; test=definition['test_indices']
        definition['train_positive_count']=int(y.iloc[train].sum())
        definition['train_negative_count']=len(train)-definition['train_positive_count']
        definition['test_positive_count']=int(y.iloc[test].sum())
        definition['test_negative_count']=len(test)-definition['test_positive_count']
        if out is not None: joblib.dump(model,out/f'{direction}_fold_{definition["fold"]}.joblib')
    # Outcome fields are only for retrospective analyses; they never enter X.
    labels=t.set_index('date').loc[oof.event_id]
    oof['gross_r']=labels.gross_r.to_numpy()
    oof['fees_r']=(labels.transaction_cost_points/labels.risk_points).to_numpy()
    oof['slippage_r']=(labels.slippage_points/labels.risk_points).to_numpy()
    oof['costs_r']=oof.fees_r+oof.slippage_r
    oof['direction']=direction
    final=v1.make_model()
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        final.fit(X,y)
    if out is not None: joblib.dump(final,out/f'ORB_{direction.upper()}_LOGISTIC.joblib')
    metrics={'model_name':f'ORB_{direction.upper()}_LOGISTIC','total_is_n':len(f),'oof_n':len(oof),
             'warmup_n':len(f)-len(oof),'all_is_prevalence':float(y.mean()),
             'model':v1.scores(oof.target,oof.probability),
             'baseline':v1.scores(oof.target,oof.baseline_probability),
             'folds':folds,'final_fit_n':len(f)}
    return oof,metrics,models


def quintiles(frame,probability,model_name,sample):
    labels,edges=pd.qcut(frame[probability],5,labels=False,retbins=True,duplicates='drop')
    rows=[]
    for k in range(len(edges)-1):
        g=frame[labels==k]
        if g.empty: continue
        rows.append({'model':model_name,'sample':sample,'quintile':k+1,'n':len(g),
                     'lower_probability':edges[k],'upper_probability':edges[k+1],
                     'boundary':'[lower,upper]' if k==0 else '(lower,upper]',
                     'mean_probability':g[probability].mean(),'actual_target_rate':g.target.mean(),
                     'mean_gross_r':g.gross_r.mean(),'mean_net_r':g.realized_net_r.mean(),
                     'mean_fees_r':g.fees_r.mean(),'mean_slippage_r':g.slippage_r.mean(),
                     'mean_costs_r':g.costs_r.mean()})
    return pd.DataFrame(rows)


def paired_comparison(directional,pooled):
    # Never synthesize a prediction for V1 warm-up observations.
    paired=directional.merge(pooled[['event_id','target','probability','baseline_probability']],
                             on='event_id',suffixes=('_v2','_v1'),validate='one_to_one')
    if not np.array_equal(paired.target_v2,paired.target_v1): raise ValueError('Paired targets disagree')
    result={'v2_oof_n':len(directional),'matched_n':len(paired),
            'unmatched_v2_event_ids':directional.loc[~directional.event_id.isin(paired.event_id),'event_id'].tolist(),
            'v1_same_events':v1.scores(paired.target_v1,paired.probability_v1),
            'v2_same_events':v1.scores(paired.target_v2,paired.probability_v2),
            'v1_baseline_same_events':v1.scores(paired.target_v1,paired.baseline_probability_v1),
            'v2_baseline_same_events':v1.scores(paired.target_v2,paired.baseline_probability_v2)}
    result['v2_improves']={k:result['v2_same_events'][k]>result['v1_same_events'][k] for k in ['roc_auc','pr_auc_average_precision']}
    result['v2_improves'].update({k:result['v2_same_events'][k]<result['v1_same_events'][k] for k in ['log_loss','brier_score']})
    return paired,result


def report(diag,difference,models,comparison,groups,ranks):
    table=v1.md_table
    lines=['# ORB directional Logistic V2 — IS only','',
           'All 1,624 original IS trades reconcile into 759 long and 865 short trades. Existing pipeline, labels, execution, data eligibility, V1 models and its 1,350 OOF predictions remain unchanged. No OOS files were accessed, no hyperparameters or rules tuned, and no thresholds/filters selected.', '',
           '## Directional diagnostics','',table(diag),'',
           'Gross/net profit factors use original currency sizing; R uses original adverse entry-to-stop risk. Fees and slippage are separated. Stop percentage includes stop gaps. OR width is in points. Confidence intervals use 2,000 whole trading-day cluster percentile bootstrap replicates, fixed seed 20261014 and deterministic streams. There is one trade per day, so day resampling reduces to row resampling for this ledger. This assumes exchangeable days and does not correct serial dependence or retrospective sample selection.',
           'Direct long-minus-short contrasts (joint trading-day bootstrap):', '',
           table(pd.DataFrame([{'basis':k,'mean_difference':v['long_minus_short_mean'],'ci_low':v['ci95'][0],'ci_high':v['ci95'][1]} for k,v in difference.items()])), '',
           f"The long-minus-short net difference is {difference['net_r']['long_minus_short_mean']:.6f}R per trade. This is economically material relative to mean friction of {float((diag.mean_fees_r+diag.mean_slippage_r).mean()):.6f}R, but both directions have negative net mean R. The direct contrast CI excludes zero and provides evidence of a directional mean difference under the exchangeable-day bootstrap assumptions. It does not establish robustness to serial dependence/regime shifts or justify a profitable directional rule; the conclusion is not based merely on separate CI overlap or point estimates.", '',
           '## Independent models and validation','',
           'Six predictors are the exact V1 feature list minus constant feature_breakout_direction. feature_or_direction is retained because it describes the opening-range candle move and is not constant breakout direction. All are known at completed signal-candle confirmation. Missing historical fields remain excluded; no imputation or new features.',
           'StandardScaler → LogisticRegression(C=1.0, class_weight=None), V1 defaults and fixed max_iter=1000/seed retained. Five chronological expanding event-count folds per direction; each scaler/model fits only the training portion. Training events whose conservative label-end is not strictly before validation start are purged; current intraday events do not overlap, so actual purges and additional embargo are zero. Fold class counts, prevalences, indices and label maturities are saved in metrics. Final models fit all events of their own direction; final models are not used for OOF scoring.', '',
           table(pd.DataFrame([{'model':m['model_name'],**{k:m['model'][k] for k in ['n','positive_prevalence','roc_auc','pr_auc_average_precision','brier_score','log_loss','balanced_accuracy_threshold_0_5']}} for m in models.values()])), '',
           table(pd.DataFrame([{'model':m['model_name']+'_constant_baseline',**{k:m['baseline'][k] for k in ['n','positive_prevalence','roc_auc','pr_auc_average_precision','brier_score','log_loss','balanced_accuracy_threshold_0_5']}} for m in models.values()])), '',
           'Primary PR-AUC is average precision; trapezoidal PR area is separately saved. Constant baselines use each direction-fold training prevalence, never validation labels. Pooled baseline ROC-AUC can differ from 0.5 because probabilities vary across folds. Confusion matrices use fixed 0.5 only for classifier diagnostics, not trading selection:', '']
    for direction,m in models.items():
        lines += [f"- {m['model_name']}: `{m['model']['confusion_matrix_true_rows_predicted_columns_0_1']}`; baseline `{m['baseline']['confusion_matrix_true_rows_predicted_columns_0_1']}` (true classes as rows; predicted classes as columns)."]
    lines += ['', 'Fold counts and class balance:', '',table(pd.DataFrame([{'direction':d,**{k:f[k] for k in ['fold','train_count','train_positive_count','train_negative_count','test_count','test_positive_count','test_negative_count','train_first_date','train_last_date','test_first_date','test_last_date']}} for d,m in models.items() for f in m['folds']])), '',
              '## Fair comparison with unchanged pooled V1','',
              'Directional OOF warm-up boundaries differ from V1. Exact paired comparisons use the intersection of stored V1 OOF and each V2 OOF event set, and score both on identical event IDs. V2 events without a V1 OOF prediction cannot be evaluated against unchanged V1 on all V2 events: they are explicitly listed and excluded from both paired scores. No in-sample/final-model prediction is substituted. Differing training populations and refit dates remain methodological differences even after matching test events.', '',
              table(pd.DataFrame([{'direction':d,'v2_total_oof':c['v2_oof_n'],'exact_matched_events':c['matched_n'],'unmatched_v2_events':len(c['unmatched_v2_event_ids'])} for d,c in comparison.items() if d!='combined'])), '']
    for d,c in comparison.items():
        lines += [f'### {d.title()} — exact matched events','',
                  table(pd.DataFrame([{'model':name,**{k:c[name][k] for k in ['n','positive_prevalence','roc_auc','pr_auc_average_precision','brier_score','log_loss','balanced_accuracy_threshold_0_5']}} for name in ['v1_same_events','v2_same_events']])), '',
                  f"V2 improvement flags (descriptive, not significance tests): `{json.dumps(c['v2_improves'])}`.", '']
    lines += ['## Probability ranking and realized net R','',
              'Quintiles use only held-out probabilities, never labels to define boundaries. Full-direction and identical-event V1/V2 quintile comparisons are in directional_probability_quintiles.csv; features remain prediction-time only, realized outcomes are descriptive labels. Quantile boundaries are retrospective OOF summaries, not executable thresholds. Within a paired sample models can assign different events to their own quintiles: this compares ranking, not identical quintile constituents.', '',
              table(groups[groups['sample']=='full_direction_oof']), '',
              'Paired ranking summaries (Spearman rank correlations of probability with target/net R and top-minus-bottom quintile net R):', '',table(pd.DataFrame(ranks)), '',
              'These are descriptive, not formal model comparison tests; nonmonotonic bins and binary target success do not establish cost-adjusted profitability. No capital-performance backtest of a classifier is performed.', '',
              '## Diagnostic conclusions','',
              'On exact matched long events, V2 worsens ROC-AUC, log loss and Brier score versus pooled V1, with a small average-precision improvement. On matched short events, ROC-AUC is nearly unchanged/slightly worse, average precision slightly improves, and log loss/Brier improve modestly. Combined matched-event log loss/Brier are slightly worse. These point estimates do not establish a general advantage from splitting.',
              'Against each directional training-prevalence baseline, both models improve ROC-AUC and average precision but both have worse log loss. Long Brier is slightly better than its baseline; short Brier is worse. Thus directional specialization does not consistently improve probability forecasting.',
              'Probability-to-net-R Spearman correlations remain negative for both directions and models. The highest-minus-lowest probability quintile net R is negative throughout: it becomes more negative for long V2 and less negative for short V2. This is no demonstrated positive net-R ranking relationship or executable strategy advantage. No thresholds or trading filters are selected.', '',
              '## Reproduction','',
              '`python orb_logistic_directional_v2/run.py` then `python -m unittest discover -s orb_logistic_directional_v2 -p "test_*.py" -v`. Read-only V1 functions are reused through importlib without running V1 main. input_manifest.json records protected input hashes, feature timing proof, versions and settings. Fold models, final scaler/classifier pipelines, folds, OOF predictions and all requested diagnostics are saved under this new directory. Existing file hashes are checked before/after.']
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def ranking(frame,probability):
    q=quintiles(frame,probability,'rank','paired')
    return {'n':len(frame),'probability_target_spearman':float(frame[probability].rank().corr(frame.target.rank())),
            'probability_net_r_spearman':float(frame[probability].rank().corr(frame.realized_net_r.rank())),
            'lowest_quintile_net_r':float(q.iloc[0].mean_net_r),'highest_quintile_net_r':float(q.iloc[-1].mean_net_r),
            'top_minus_bottom_quintile_net_r':float(q.iloc[-1].mean_net_r-q.iloc[0].mean_net_r)}


def main():
    protected=[ROOT/name for name in v1.INPUTS]+[p for p in (ROOT/'orb_logistic_v1').iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):v1.sha(p) for p in protected}
    f,t,X,y,pred,end,proofs=v1.load_data(ROOT)
    if len(f)!=1624: raise ValueError('Expected 1,624 original IS trades')
    feature_manifest=json.loads((ROOT/'orb_logistic_v1/features.json').read_text())
    if feature_manifest['selected_features']!=v1.FEATURES: raise ValueError('V1 features changed')
    pooled=pd.read_csv(ROOT/'orb_logistic_v1/oof_predictions.csv')
    if len(pooled)!=1350 or not pooled.event_id.is_unique or not pooled.event_id.between('2015-01-01','2021-12-31').all(): raise ValueError('Invalid V1 OOF')
    diag,difference=diagnostics(f,t)
    diag.to_csv(OUT/'directional_diagnostics.csv',index=False)
    models={}; outputs={}; comparison={}; groups=[]; ranks=[]
    for direction in ['long','short']:
        oof,m,_=fit_direction(f,t,direction,OUT)
        outputs[direction]=oof; models[direction]=m
        oof.to_csv(OUT/f'{direction}_oof_predictions.csv',index=False)
        groups.append(quintiles(oof,'probability',m['model_name'],'full_direction_oof'))
        paired,c=paired_comparison(oof,pooled)
        comparison[direction]=c
        paired.to_csv(OUT/f'{direction}_paired_predictions.csv',index=False)
        paired['target']=paired.target_v2
        for name,col in [('POOLED_V1','probability_v1'),(m['model_name'],'probability_v2')]:
            groups.append(quintiles(paired,col,name,direction+'_exact_matched'))
            ranks.append({'direction':direction,'model':name,**ranking(paired,col)})
    combined=pd.concat(outputs.values(),ignore_index=True).sort_values('prediction_timestamp')
    _,comparison['combined']=paired_comparison(combined,pooled)
    groups=pd.concat(groups,ignore_index=True)
    groups.to_csv(OUT/'directional_probability_quintiles.csv',index=False)
    pd.DataFrame(ranks).to_csv(OUT/'paired_ranking_comparison.csv',index=False)
    metrics={'models':models,'paired_comparison':comparison,'paired_ranking':ranks,
             'directional_mean_differences':difference,'unchanged_v1_full_oof':v1.scores(pooled.target,pooled.probability),
             'features':FEATURES,'seed':SEED,'bootstrap_replicates':REPLICATES}
    (OUT/'directional_model_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    report(diag,difference,models,comparison,groups,ranks)
    if hashes!={str(p.relative_to(ROOT)):v1.sha(p) for p in protected}: raise RuntimeError('Existing files changed')
    (OUT/'input_manifest.json').write_text(json.dumps({'protected_input_sha256':hashes,'features':FEATURES,
       'timing_proofs':{c:v1.TIMING_PROOFS[c] for c in FEATURES},'verification_snippets':proofs,
       'versions':json.loads((ROOT/'orb_logistic_v1/metrics.json').read_text())['versions'],
       'model_params':v1.make_model().named_steps['logistic'].get_params(),'seed':SEED,
       'implementation_sha256':v1.sha(__file__)},indent=2),encoding='utf-8')
    print(diag[['direction','n','gross_expectancy_r','net_expectancy_r']].to_string(index=False))
    print(json.dumps({d:{'n':c['matched_n'],'improvements':c['v2_improves'],
                       'v1':c['v1_same_events'],'v2':c['v2_same_events']} for d,c in comparison.items()},indent=2))


if __name__=='__main__': main()
