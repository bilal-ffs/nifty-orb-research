"""Fixed, chronological IS-only Logistic Regression V1. No OOS access."""
from pathlib import Path
import ast
import hashlib
import json
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             auc, log_loss, brier_score_loss, accuracy_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261014
TARGET = 'target_reached_1_5r'
# Fixed structural selection, independent of labels or measured performance.
FEATURES = ['feature_or_width_points', 'feature_or_width_pct', 'feature_or_direction',
            'feature_or_return', 'feature_breakout_direction',
            'feature_breakout_minutes_since_0915', 'feature_breakout_distance_points']
TIMING_PROOFS = {
    'feature_or_width_points': '09:15–09:29 candle highs/lows, completed by 09:30.',
    'feature_or_width_pct': 'OR width / 09:15 open; known by 09:30.',
    'feature_or_direction': 'Sign of 09:29 close minus 09:15 open; known by 09:30.',
    'feature_or_return': '09:29 close / 09:15 open − 1; known by 09:30.',
    'feature_breakout_direction': 'Completed signal close strictly above/below OR boundary.',
    'feature_breakout_minutes_since_0915': 'Signal start minute + 1 − 09:15, derived clock feature at confirmation.',
    'feature_breakout_distance_points': 'Completed signal close beyond the relevant OR boundary.',
}
INPUTS = ['orb_ml_is.csv','orb_is_trades.csv','orb_pipeline.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pipeline(path):
    source=Path(path).read_text(encoding='utf-8')
    functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    required={'first_event': [*FEATURES, 'prediction = row.timestamp + pd.Timedelta(minutes=1)',
                              'between(555, 569)', '570 <= minute <= 719'],
              'simulate_trade': ["'target_hit': int(reason == 'target')", "'exit_candle_end'", '>= 929'],
              'run_split': ["'target_reached_1_5r': trade['target_hit']"]}
    for name,fragments in required.items():
        if name not in functions or any(fragment not in functions[name] for fragment in fragments):
            raise ValueError('Label/timing contract cannot be verified in pipeline: '+name)
    return {name:functions[name] for name in required}


def validate_data(f,t):
    if not f.event_id.is_unique or not t.date.is_unique:
        raise ValueError('Duplicate event dates')
    if not f.event_id.between('2015-01-01','2021-12-31').all() or not t.date.between('2015-01-01','2021-12-31').all():
        raise ValueError('Only IS 2015–2021 is permitted')
    if set(f.event_id)!=set(t.date): raise ValueError('ML/ledger events differ')
    f=f.sort_values('prediction_timestamp',kind='stable').reset_index(drop=True)
    t=t.set_index('date').loc[f.event_id].reset_index()
    y=pd.to_numeric(f[TARGET],errors='raise')
    if not y.isin([0,1]).all() or y.nunique()!=2: raise ValueError('Invalid binary target')
    if not np.array_equal(y.to_numpy(),(t.exit_reason=='target').astype(int).to_numpy()):
        raise ValueError('Label is not target-first exit indicator')
    if not np.array_equal(y.to_numpy(),t.target_hit.to_numpy()): raise ValueError('Ledger labels differ')
    if not (f.execution_status=='filled').all(): raise ValueError('Unfilled events require a separate contract')
    prediction=pd.to_datetime(f.prediction_timestamp)
    signal=pd.to_datetime(f.signal_candle_start)
    entry=pd.to_datetime(t.entry_timestamp)
    label_end=pd.to_datetime(t.exit_candle_end)
    if not ((prediction-signal)==pd.Timedelta(minutes=1)).all(): raise ValueError('Confirmation alignment failed')
    if not (prediction==pd.to_datetime(t.prediction_timestamp)).all(): raise ValueError('Prediction/ledger alignment failed')
    if not (prediction.dt.strftime('%Y-%m-%d')==f.event_id).all(): raise ValueError('Event-date misalignment')
    if not prediction.is_monotonic_increasing or not prediction.is_unique: raise ValueError('Invalid chronology')
    if not (entry>=prediction).all() or not (label_end>prediction).all(): raise ValueError('Invalid information window')
    if not (label_end.dt.date==prediction.dt.date).all(): raise ValueError('Unexpected overnight outcome')
    direction=t.direction.map({'long':1,'short':-1})
    if not np.array_equal(f.feature_breakout_direction.to_numpy(),direction.to_numpy()): raise ValueError('Direction misaligned')
    clock=prediction.dt.hour*60+prediction.dt.minute-555
    if not np.array_equal(f.feature_breakout_minutes_since_0915.to_numpy(),clock.to_numpy()): raise ValueError('Breakout clock mismatch')
    if not np.allclose(t.target_price,t.entry_price+direction*1.5*t.risk_points): raise ValueError('Target is not original 1.5R')
    if not np.allclose(f.target_net_return_r,t.net_r): raise ValueError('Return labels misaligned')
    X=f[FEATURES].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(X.to_numpy()).all(): raise ValueError('Missing/nonfinite selected features: stop, do not impute')
    return f,t,X,y.astype(int),prediction,label_end


def load_data(root):
    proofs=verify_pipeline(root/'orb_pipeline.py')
    f=pd.read_csv(root/'orb_ml_is.csv')
    t=pd.read_csv(root/'orb_is_trades.csv')
    return (*validate_data(f,t),proofs)


def make_model():
    # Defaults for C, regularization, class_weight, solver and tolerance.
    # A higher fixed iteration cap is a convergence safeguard, not tuning.
    return Pipeline([('scaler',StandardScaler()),
                     ('logistic',LogisticRegression(C=1.0,class_weight=None,
                                                   max_iter=1000,random_state=SEED))])


def make_folds(prediction,label_end):
    folds=[]
    for number,(candidate,test) in enumerate(TimeSeriesSplit(n_splits=5).split(prediction),1):
        boundary=prediction.iloc[test[0]]
        train=candidate[(label_end.iloc[candidate]<boundary).to_numpy()]
        if not len(train): raise ValueError('No mature training labels')
        if prediction.iloc[train].max()>=boundary or label_end.iloc[train].max()>=boundary:
            raise ValueError('Overlapping information windows')
        folds.append((number,train,test,candidate[~np.isin(candidate,train)]))
    return folds


def scores(y,p):
    classification=(np.asarray(p)>=.5).astype(int)
    precision,recall,_=precision_recall_curve(y,p)
    return {'n':len(y),'positive_prevalence':float(np.mean(y)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'pr_auc_average_precision':float(average_precision_score(y,p)),
            'pr_auc_trapezoidal':float(auc(recall,precision)),
            'log_loss':float(log_loss(y,p,labels=[0,1])),
            'brier_score':float(brier_score_loss(y,p)),
            'accuracy_threshold_0_5':float(accuracy_score(y,classification)),
            'balanced_accuracy_threshold_0_5':float(balanced_accuracy_score(y,classification)),
            'confusion_matrix_true_rows_predicted_columns_0_1':confusion_matrix(y,classification,labels=[0,1]).tolist()}


def probability_groups(oof,column,q):
    labels,edges=pd.qcut(oof[column],q=q,labels=False,retbins=True,duplicates='drop')
    rows=[]
    for k in range(len(edges)-1):
        g=oof.loc[labels==k]
        if g.empty: continue  # Tied fold-constant probabilities can leave empty intervals.
        rows.append({'group':k+1,'probability_lower':float(edges[k]),'probability_upper':float(edges[k+1]),
                     'boundary':'[lower,upper]' if k==0 else '(lower,upper]',
                     'n':len(g),'mean_probability':float(g[column].mean()),
                     'observed_target_rate':float(g.target.mean()),
                     'mean_realized_net_r':float(g.realized_net_r.mean())})
    return pd.DataFrame(rows)


def train_oof(X,y,f,t,prediction,label_end,out=None):
    rows=[]; definitions=[]; models=[]
    for number,train,test,purged in make_folds(prediction,label_end):
        model=make_model()
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(X.iloc[train],y.iloc[train])
        p=model.predict_proba(X.iloc[test])[:,1]
        baseline=float(y.iloc[train].mean())
        rows.append(pd.DataFrame({'row_index':test,'event_id':f.event_id.iloc[test].to_numpy(),
                                 'prediction_timestamp':f.prediction_timestamp.iloc[test].to_numpy(),
                                 'fold':number,'target':y.iloc[test].to_numpy(),
                                 'probability':p,'baseline_probability':baseline,
                                 'realized_net_r':t.net_r.iloc[test].to_numpy()}))
        definitions.append({'fold':number,'train_indices':train.tolist(),'test_indices':test.tolist(),
                            'purged_indices':purged.tolist(),'train_count':len(train),'test_count':len(test),
                            'train_first_date':f.event_id.iloc[train[0]],'train_last_date':f.event_id.iloc[train[-1]],
                            'test_first_date':f.event_id.iloc[test[0]],'test_last_date':f.event_id.iloc[test[-1]],
                            'latest_training_label_end':label_end.iloc[train].max().isoformat(),
                            'first_test_prediction':prediction.iloc[test].min().isoformat(),
                            'training_prevalence':baseline,
                            'model':scores(y.iloc[test],p),'baseline':scores(y.iloc[test],np.full(len(test),baseline))})
        if out is not None: joblib.dump(model,out/f'fold_{number}.joblib')
        models.append(model)
    return pd.concat(rows,ignore_index=True),definitions,models


def feature_manifest(f):
    excluded={}
    for c in f:
        if c in FEATURES: continue
        if c.startswith('feature_') and c!='feature_history_available':
            excluded[c]={'reason':'Historical feature has missing values; excluded as a whole column, no imputation.',
                         'missing_count':int(f[c].isna().sum())}
        else:
            excluded[c]={'reason':'Identifier/raw timestamp, history-availability flag, execution-dependent field or outcome; excluded from predictors.'}
    return {'selected_features':FEATURES,'target':TARGET,'timing_proofs':TIMING_PROOFS,
            'excluded_columns':excluded,'schema_dtypes':f.dtypes.astype(str).to_dict(),
            'selection_policy':'Seven current-session features fixed structurally from pipeline definitions; historical features excluded for unavailable records, not performance. No row deletion or fitted imputation.'}


def md_table(frame):
    def fmt(v): return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(out,metrics,folds,quintiles,manifest):
    model,baseline=metrics['oof_model'],metrics['oof_baseline']
    comparisons=[]
    for key in ['positive_prevalence','roc_auc','pr_auc_average_precision','pr_auc_trapezoidal','log_loss','brier_score','accuracy_threshold_0_5','balanced_accuracy_threshold_0_5']:
        comparisons.append({'metric':key,'logistic':model[key],'constant_baseline':baseline[key]})
    lines=['# NIFTY 50 ORB Logistic Regression V1 — IS only','',
           f"{metrics['total_is_events']} events; {metrics['oof_events']} chronological OOF predictions; {metrics['warmup_events']} initial events used only as training warm-up. Final model fits all IS events. No OOS file opened or predicted; existing project files remain unchanged.",'',
           'Target: 1 for the original 1.5R target exit before stop/session exit, otherwise 0. Verified against pipeline target construction, IS ledger exit category, target_hit and entry + direction × 1.5 × actual risk. Net returns are used only for retrospective probability-group descriptions.',
           'Prediction is at completed breakout-candle confirmation, one minute after its start, before next-open execution under the original sequencing convention. Original full-session quality screening is retained and introduces retrospective sample-selection limitations.', '',
           '## Features and model','',
           '\n'.join(f'- `{name}`: {manifest["timing_proofs"][name]}' for name in FEATURES),'',
           'Five historical fields with missing records are excluded as entire columns: prior ATR-relative OR width, gap, previous-day return/range, and historical volatility. History-availability flag is excluded. All 1,624 rows remain; no imputation. Selection is structural and is not based on labels or subgroup profitability. Raw dates/timestamps, fill-dependent fields, target labels, realized R and exits never enter X.',
           'Pipeline: StandardScaler → LogisticRegression. C=1.0, class_weight=None, default solver/regularization/tolerance; max_iter=1000 is a fixed convergence cap; random_state=20261014. No hyperparameter, threshold, feature-performance, calibration or profitability optimization. Each fold scaler and estimator fit only that fold training data. Final scaler/model fit all IS separately; the final model produces no evaluation predictions.', '',
           '## Chronology, information windows and baseline','',
           'Five expanding TimeSeriesSplit folds use contiguous, equal-sized event-count test blocks; dates are irregular trading events, so folds do not have equal calendar duration. Training precedes validation. Training labels whose conservative exit-candle-end timestamp is not strictly before first validation confirmation are purged. Original events end intraday and there is at most one event per day: zero actual overlaps/purges, hence zero additional embargo days. Shared backward-looking historical observations alone are not future-label overlap. No random split.',
           md_table(pd.DataFrame([{k:fold[k] for k in ['fold','train_count','test_count','train_first_date','train_last_date','test_first_date','test_last_date','training_prevalence']} for fold in folds])),'',
           'The comparison baseline predicts only each fold training-label prevalence, fixed throughout that fold test block. Its probabilities vary between folds, so its pooled ROC-AUC need not equal 0.5. Per-fold baseline ROC-AUC is 0.5; PR average precision equals that fold test prevalence. Fold-specific and pooled metrics are saved. Primary PR-AUC is average precision (step-weighted precision–recall area); trapezoidal PR area is separately labeled and can differ, especially for tied constant predictions.', '',
           '## OOF results','',md_table(pd.DataFrame(comparisons)),'',
           f'Model confusion matrix (rows=true 0,1; columns=predicted 0,1): `{model["confusion_matrix_true_rows_predicted_columns_0_1"]}`. Baseline: `{baseline["confusion_matrix_true_rows_predicted_columns_0_1"]}`. Threshold is fixed at 0.5; high accuracy with the majority negative class is not evidence of useful target detection.',
           f'Pooled OOF model beats baseline on: ROC-AUC={metrics["beats_baseline"]["roc_auc"]}, PR average precision={metrics["beats_baseline"]["pr_auc_average_precision"]}, log loss={metrics["beats_baseline"]["log_loss"]}, Brier={metrics["beats_baseline"]["brier_score"]}. These are descriptive point estimates, not significance claims or external generalization evidence.', '',
           ('The model improves pooled discrimination but does not beat the training-prevalence baseline on either log loss or Brier score. There is no demonstrated overall probability-forecast advantage.' if metrics['beats_baseline']['roc_auc'] and not metrics['beats_baseline']['log_loss'] and not metrics['beats_baseline']['brier_score'] else 'Metric-specific comparisons above define the observed baseline comparison; no single metric establishes forecasting or trading utility.'), '',
           'Calibration by OOF probability decile is in calibration_deciles.csv for the model and baseline_calibration.csv for the baseline. Ties remain together and may reduce the number of bins; empty intervals are omitted. Boundaries, counts, mean predicted probability and observed target rate are saved. No calibrator was trained.', '',
           '## Descriptive net R ranking','',md_table(quintiles),'',
           'These quintiles use only held-out probabilities and retrospective realized net R. Boundaries are pooled IS OOF quantiles, not live entry thresholds. Target likelihood and net R are different objectives: the binary target excludes profitable session exits and cost/R varies. These groups neither constitute a backtest of a traded classifier nor select filters, sizing or thresholds; no executable performance claim is made.', '',
           f"All {len(quintiles)} probability groups have {'negative' if (quintiles.mean_realized_net_r<0).all() else 'mixed'} descriptive mean net R. Lowest-probability quintile: {quintiles.iloc[0].mean_realized_net_r:.6f}R; highest-probability quintile: {quintiles.iloc[-1].mean_realized_net_r:.6f}R. Observed target frequencies and net R need not be monotonic in estimated target probability.", '',
           '## Reproduction','',
           'From the project directory: `python orb_logistic_v1/train.py`, then `python -m unittest discover -s orb_logistic_v1 -p "test_*.py" -v`. Version pins are in requirements.txt. final_model.joblib includes scaler and classifier; fold models permit checking held-out prediction/scaler provenance. Load with the saved feature order; future missing values require a separately reviewed data contract. No future evaluation is performed here.',
           'Feature list, exclusions/timing evidence, verification snippets, input hashes, environment, warm-up IDs, fold indices, training prevalences, metrics and OOF outputs are saved. Existing input hashes are checked before/after execution.', '',
           'Official references: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    out=Path(__file__).resolve().parent
    root=out.parent
    hashes={name:sha(root/name) for name in INPUTS}
    f,t,X,y,prediction,label_end,proofs=load_data(root)
    oof,folds,_=train_oof(X,y,f,t,prediction,label_end,out)
    manifest=feature_manifest(f)
    quintiles=probability_groups(oof,'probability',5)
    model_metrics=scores(oof.target,oof.probability)
    baseline_metrics=scores(oof.target,oof.baseline_probability)
    warmup=np.setdiff1d(np.arange(len(f)),oof.row_index)
    metrics={'total_is_events':len(f),'oof_events':len(oof),'warmup_events':len(warmup),
             'warmup_event_ids':f.event_id.iloc[warmup].tolist(),
             'oof_model':model_metrics,'oof_baseline':baseline_metrics,
             'beats_baseline':{k:model_metrics[k]>baseline_metrics[k] for k in ['roc_auc','pr_auc_average_precision']},
             'threshold':.5,'final_fit_count':len(f),'seed':SEED,
             'model_params':make_model().named_steps['logistic'].get_params(),
             'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,
                         'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__},
             'input_sha256':hashes,'implementation_sha256':sha(__file__)}
    metrics['beats_baseline'].update({k:model_metrics[k]<baseline_metrics[k] for k in ['log_loss','brier_score']})
    final=make_model()
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        final.fit(X,y)
    joblib.dump(final,out/'final_model.joblib')
    oof.to_csv(out/'oof_predictions.csv',index=False)
    quintiles.to_csv(out/'probability_quintiles.csv',index=False)
    probability_groups(oof,'probability',10).to_csv(out/'calibration_deciles.csv',index=False)
    probability_groups(oof,'baseline_probability',10).to_csv(out/'baseline_calibration.csv',index=False)
    for name,value in [('features.json',manifest),('folds.json',folds),('metrics.json',metrics),('verification_snippets.json',proofs)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame({'feature':FEATURES,'coefficient_standardized':final.named_steps['logistic'].coef_[0]}).to_csv(out/'final_coefficients.csv',index=False)
    write_report(out,metrics,folds,quintiles,manifest)
    if hashes!={name:sha(root/name) for name in INPUTS}: raise RuntimeError('Existing input changed')
    print(json.dumps({'model':model_metrics,'baseline':baseline_metrics,'beats_baseline':metrics['beats_baseline']},indent=2))
    print('Saved IS-only models/results. Existing input hashes unchanged.')


if __name__=='__main__': main()
