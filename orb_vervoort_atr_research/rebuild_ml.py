"""Rebuild ML from saved ledgers; preserves all strategy outcomes."""
import json
import pandas as pd
import experiment as e
import strategy as s


def main():
    audit=json.loads((e.OUT/'data_audit.json').read_text())
    folds=json.loads((e.OUT/'fold_windows.json').read_text())
    metrics={}; quintiles=[]
    for multiplier in s.MULTIPLIERS:
        tag=f'atr_{multiplier:.1f}'
        for mode in ['next_open','pine_close']:
            path=e.OUT/f'{tag}_{mode}_trades.csv'
            t=pd.read_csv(path,float_precision='round_trip')
            # Stabilize a ratio that can be mathematically constant within a
            # configuration; 1e-12 resolution prevents scaling cancellation noise.
            t['feature_trail_distance_atr']=t.feature_trail_distance_atr.round(12)
            t.to_csv(path,index=False)
            if mode=='next_open':
                metrics[str(multiplier)],q=e.ml(t,folds,tag)
                quintiles.append(q)
    (e.OUT/'ml_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    q=pd.concat(quintiles,ignore_index=True)
    q.to_csv(e.OUT/'probability_quintiles.csv',index=False)
    comparison=pd.read_csv(e.OUT/'atr_multiplier_comparison.csv')
    breakdown=pd.read_csv(e.OUT/'long_short_breakdown.csv')
    e.report(comparison,breakdown,metrics,quintiles,audit,folds)
    print('All ten classifiers rebuilt; trade outcomes unchanged.')


if __name__=='__main__': main()
