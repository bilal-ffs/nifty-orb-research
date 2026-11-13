import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from orb_pipeline import (Config, first_event, simulate_trade, historical_features,
                          load_and_audit, run_split, account_metrics)


def session():
    ts = pd.date_range('2021-12-30 09:15', periods=375, freq='min', tz='Asia/Kolkata')
    return pd.DataFrame({'timestamp': ts, 'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': np.nan})


def history():
    return pd.Series({'prev_close': 100., 'prev_day_return': .01, 'prev_day_range': 3.,
                      'prev_atr14': 2., 'historical_vol20': .02})


class OrbTests(unittest.TestCase):
    def test_timestamp_alignment_and_next_open(self):
        g = session()
        g.loc[15, ['high','close']] = [102., 102.]
        g.loc[16, ['open','high','low','close']] = [103.,104.,102.,103.]
        event = first_event(g, history())
        self.assertEqual(event[4]['prediction_timestamp'], '2021-12-30T09:31:00+05:30')
        trade, _ = simulate_trade(g, *event[:4], Config(slippage_bps=0))
        self.assertEqual(trade['entry_reference_open'], 103.)
        self.assertEqual(trade['entry_timestamp'], event[4]['prediction_timestamp'])
        self.assertNotEqual(trade['entry_timestamp'], trade['signal_candle_start'])

    def test_missing_opening_candle(self):
        g = session().drop(index=3).reset_index(drop=True)
        g.loc[14, ['high','close']] = [103.,102.]
        self.assertIsNone(first_event(g, history()))

    def test_next_available_open_and_cutoff(self):
        g = session()
        g.loc[15, ['high','close']] = [103.,102.]
        g = g.drop(index=16).reset_index(drop=True)
        event = first_event(g, history())
        trade, _ = simulate_trade(g, *event[:4], Config(slippage_bps=0))
        self.assertIn('09:32:00', trade['entry_timestamp'])
        g = session()
        g.loc[164, ['high','close']] = [103.,102.]  # 11:59 candle
        event = first_event(g, history())
        trade, _ = simulate_trade(g, *event[:4], Config(slippage_bps=0))
        self.assertIn('12:00:00', trade['entry_timestamp'])
        g = g.drop(index=165).reset_index(drop=True)
        event = first_event(g, history())
        trade, reason = simulate_trade(g, *event[:4], Config())
        self.assertIsNone(trade)
        self.assertEqual(reason, 'entry_after_1200')

    def test_stop_first_collision_both_directions(self):
        for direction in [1,-1]:
            g = session()
            g.loc[16, ['high','low']] = [105.,95.]
            trade, _ = simulate_trade(g, 15, direction, 101.,99., Config(slippage_bps=0))
            self.assertEqual(trade['exit_reason'], 'stop')
            self.assertEqual(trade['gross_r'], -1.)
            self.assertEqual(trade['target_hit'], 0)

    def test_conservative_stop_gaps(self):
        for direction, gap in [(1,97.),(-1,103.)]:
            g = session()
            g.loc[16:, ['high','low']] = [100.5,99.5]
            g.loc[17, 'open'] = gap
            trade, _ = simulate_trade(g, 15, direction,101.,99.,Config(slippage_bps=0))
            self.assertEqual(trade['exit_reason'], 'stop_gap')
            self.assertEqual(trade['exit_reference_price'], gap)
            self.assertEqual(trade['gross_r'], -3.)

    def test_target_and_session_exit(self):
        g = session()
        g.loc[16:, ['high','low']] = [100.5,99.5]
        g.loc[17, 'high'] = 102.
        trade, _ = simulate_trade(g,15,1,101.,99.,Config(slippage_bps=0))
        self.assertEqual(trade['exit_reason'],'target')
        self.assertEqual(trade['gross_r'],1.5)
        g.loc[17,'high'] = 100.5
        g.loc[374,['open','high','low','close']] = [100.2,110.,90.,105.]
        trade, _ = simulate_trade(g,15,1,101.,99.,Config(slippage_bps=0))
        self.assertEqual(trade['exit_reason'],'session_exit')
        self.assertIn('15:29:00',trade['exit_timestamp'])
        self.assertEqual(trade['exit_reference_price'],100.2)

    def test_slippage_cost_reconciliation(self):
        g = session()
        trade, _ = simulate_trade(g,15,1,101.,99.,Config())
        self.assertGreater(trade['entry_price'],trade['entry_reference_open'])
        self.assertLess(trade['exit_price'],trade['exit_reference_price'])
        self.assertAlmostEqual(trade['net_pnl'],trade['gross_pnl']-trade['slippage_pnl']-trade['transaction_cost_pnl'])
        self.assertGreater(trade['transaction_cost_pnl'],0)

    def test_future_candles_cannot_change_predictors(self):
        g = session()
        g.loc[15,['high','close']] = [103.,102.]
        before = first_event(g,history())[4]
        g.loc[16:,['open','high','low','close']] = [500.,600.,400.,550.]
        after = first_event(g,history())[4]
        self.assertEqual(before,after)

    def test_history_shift_and_future_invariance(self):
        index = pd.date_range('2021-11-01',periods=50).strftime('%Y-%m-%d')
        close = np.arange(100.,150.)
        d = pd.DataFrame({'open':close,'high':close+2,'low':close-2,'close':close},index=index)
        a = historical_features(d)
        d.iloc[30:] *= 10
        b = historical_features(d)
        pd.testing.assert_frame_equal(a.iloc[:31],b.iloc[:31])
        self.assertEqual(a.iloc[30].prev_close,129.)
        self.assertAlmostEqual(a.iloc[30].prev_day_return,129/128-1)

    def test_audit_missing_duplicate_invalid_evening_and_immutability(self):
        g = session()
        g = g.drop(index=30)
        g.loc[31,'high'] = 90.
        g = pd.concat([g,g.iloc[[1]]],ignore_index=True)
        evening = g.iloc[[0]].copy()
        evening['timestamp'] = pd.Timestamp('2021-12-30 18:00',tz='Asia/Kolkata')
        g = pd.concat([g,evening],ignore_index=True)
        g['date'] = g.timestamp.dt.strftime('%d-%m-%Y %H:%M')
        g['volume'] = 0
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input.csv'
            g[['date','open','high','low','close','volume']].to_csv(p,index=False)
            original = p.read_bytes()
            sessions,daily,audit = load_and_audit(p)
            self.assertEqual(p.read_bytes(),original)
        self.assertIsNone(sessions['2021-12-30'])
        self.assertEqual(audit['outside_regular_rows_excluded'],1)
        self.assertIn('09:45',audit['sessions'][0]['missing_minutes'])
        self.assertIn('invalid_ohlcv',audit['sessions'][0]['exclusion_reasons'])
        self.assertIn('duplicate_timestamp',audit['sessions'][0]['exclusion_reasons'])

    def test_split_and_account_series(self):
        g = session()
        g.loc[15,['high','close']] = [103.,102.]
        h = pd.DataFrame([history(),history()],index=['2021-12-30','2022-01-03'])
        sessions = {'2021-12-30':g,'2022-01-03':None}
        trades,events,dates,skips = run_split(sessions,h,'2015-01-01','2021-12-31',Config())
        self.assertEqual(len(trades),1)
        self.assertEqual(len(events),1)
        self.assertEqual(dates,['2021-12-30'])
        metrics,daily = account_metrics(trades,['2021-12-30','2021-12-31'],Config())
        self.assertEqual(daily.iloc[1].account_return,0)
        self.assertAlmostEqual(daily.iloc[0].account_return,trades.iloc[0].net_pnl/1_000_000)
        self.assertAlmostEqual(daily.iloc[-1].equity,1_000_000+trades.net_pnl.sum())

    def test_sizing_does_not_change_trade_path_or_r(self):
        g = session()
        a, _ = simulate_trade(g,15,1,101.,99.,Config(notional_fraction=.1))
        b, _ = simulate_trade(g,15,1,101.,99.,Config(notional_fraction=1))
        for key in ['entry_price','exit_price','exit_reason','risk_points','net_r','target_hit']:
            self.assertEqual(a[key],b[key])
        self.assertAlmostEqual(b['net_pnl'],10*a['net_pnl'])

    def test_insolvent_account_sharpe_is_undefined(self):
        g = session()
        trade, _ = simulate_trade(g,15,1,101.,99.,Config())
        trade['net_pnl'] = -1_100_000.
        metrics,daily = account_metrics(pd.DataFrame([trade]),['2021-12-30','2021-12-31'],Config())
        self.assertFalse(metrics['account_equity_positive_throughout'])
        self.assertIsNone(metrics['sharpe_daily_252_rf0'])
        self.assertIsNone(metrics['monthly'][0]['account_return'])


if __name__ == '__main__':
    unittest.main()
