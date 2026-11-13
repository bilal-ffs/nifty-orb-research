"""Focused checks for ATR initialization and order execution semantics."""
import numpy as np
import pandas as pd
from orb_backtest import fill, indicators, backtest


def main():
    assert fill('long',95,110,93,100,90,97)==(93,'stop_gap',False)
    assert fill('short',105,90,108,110,100,102)==(108,'stop_gap',False)
    assert fill('long',95,110,111,113,109,112)==(111,'target_gap',False)
    assert fill('long',95,110,100,112,94,103)==(95,'stop',True)
    assert fill('short',110,95,100,112,94,103)==(95,'target',True)
    assert fill('long',95,110,100,109,96,103) is None
    x=pd.DataFrame({'open':[100.]*7,'high':[102.]*7,'low':[98.]*7,'close':[100.]*7})
    x=indicators(x)
    assert x.atr_5.iloc[:4].isna().all()
    assert np.allclose(x.atr_5.iloc[4:],4.)
    assert np.allclose(x.vervoort_stop.iloc[4:],86.)
    # Entry bar's earlier low must not stop out a close-of-bar entry.
    # A tighter ATR stop calculated at the NEXT bar close must not be
    # retroactively applied to that bar's earlier low.
    idx=pd.to_datetime(['2025-01-02 09:15','2025-01-02 09:20',
                        '2025-01-02 09:25','2025-01-02 09:30',
                        '2025-01-02 09:35','2025-01-02 14:55'])
    bars=pd.DataFrame({'open':[99,99,99,100,103,103],
                       'high':[100,100,100,104,105,105],
                       'low':[98,98,98,90,96,103],
                       'close':[99,99,99,103,104,104],
                       'vervoort_stop':[90,90,90,95,102,102]},index=idx)
    # The first next-bar high hits target 106? No: target is 103+1.5*2=106.
    t=backtest(pd.DataFrame(),bars,{idx[0].normalize()},'pine5m')
    assert len(t)==1
    assert t.iloc[0].exit_reason=='15:00_square_off'
    assert t.iloc[0].entry_time==pd.Timestamp('2025-01-02 09:35')
    assert t.iloc[0].exit_time==pd.Timestamp('2025-01-02 15:00')
    assert t.iloc[0].initial_risk_points==8
    assert t.iloc[0].R==0.125
    assert t.iloc[0].final_stop==102
    print('PASS: ATR seed, gap fills, intrabar ordering, no entry-bar lookahead,')
    print('no retroactive trailing stop, close-of-bar entry, square-off, and R.')


if __name__=='__main__':
    main()
