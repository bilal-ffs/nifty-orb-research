import unittest
import numpy as np
import pandas as pd
import strategy as s


def day():
    timestamp=pd.date_range('2020-01-02 09:15',periods=75,freq='5min',tz='Asia/Kolkata')
    g=pd.DataFrame({'timestamp':timestamp,'date':'2020-01-02','open':100.,'high':101.,'low':99.,'close':100.,
                    'atr':1.,'trail':98.,'trend':0,'eligible':True})
    g.loc[3,['high','close']]=[103.,102.]
    return g


class StrategyTests(unittest.TestCase):
    def test_five_minute_boundary_alignment_and_missing_session(self):
        ts=pd.date_range('2020-01-02 09:15',periods=375,freq='min',tz='Asia/Kolkata')
        g=pd.DataFrame({'timestamp':ts,'date':ts.strftime('%d-%m-%Y %H:%M'),
                         'open':np.arange(375)+100.,'high':np.arange(375)+102.,
                         'low':np.arange(375)+99.,'close':np.arange(375)+101.,'volume':0.})
        bars,eligible,audit=s.aggregate(g)
        self.assertEqual(len(bars),75)
        self.assertEqual(bars.timestamp.iloc[0].strftime('%H:%M'),'09:15')
        self.assertEqual(bars.close.iloc[0],105.)
        self.assertEqual(bars.open.iloc[1],105.)
        self.assertEqual(bars.timestamp.iloc[-1].strftime('%H:%M'),'15:25')
        bars,eligible,audit=s.aggregate(g.drop(index=2))
        self.assertEqual(eligible,[])
        self.assertEqual(len(bars),74)
        self.assertFalse(bars.eligible.any())

    def test_wilder_seed_and_overnight_true_range(self):
        g=pd.DataFrame({'high':[102.,103.,104.,105.,106.,120.],
                        'low':[99.,100.,101.,102.,103.,117.],
                        'close':[101.,102.,103.,104.,105.,119.]})
        atr=s.wilder_atr(g)
        self.assertTrue(np.isnan(atr[:4]).all())
        self.assertEqual(atr[4],3.)
        self.assertEqual(atr[5],(3*4+15)/5)

    def test_vervoort_exact_recurrence_and_trend(self):
        close=np.array([100.,102.,104.,101.,99.,100.,105.])
        trail,trend=s.vervoort(close,np.full(7,2.),1.)
        np.testing.assert_allclose(trail,[98.,100.,102.,103.,101.,101.,103.])
        np.testing.assert_array_equal(trend,[0,0,0,-1,-1,-1,1])

    def test_missing_atr_initialization_and_equality(self):
        trail,trend=s.vervoort(np.array([100.,101.,102.,103.,104.]),np.array([np.nan]*4+[2.]),1.)
        self.assertTrue(np.isnan(trail[:4]).all())
        self.assertEqual(trail[4],102.)
        trail,trend=s.vervoort(np.array([100.,98.]),np.array([2.,2.]),1.)
        np.testing.assert_allclose(trail,[98.,100.])
        np.testing.assert_array_equal(trend,[0,0])

    def test_signal_close_vs_next_open_and_actual_risk(self):
        g=day(); g.loc[4,'open']=103.
        parity,_=s.simulate(g,3,1,101.,99.,'pine_close',1.)
        primary,_=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertEqual(parity['entry_reference_price'],102.)
        self.assertEqual(primary['entry_reference_price'],103.)
        self.assertEqual(parity['target_price'],105.)
        self.assertEqual(primary['target_price'],105.)
        self.assertNotAlmostEqual(primary['target_distance_actual_r'],1.5)
        self.assertIn('09:35:00',primary['entry_timestamp'])

    def test_no_retroactive_signal_bar_exit(self):
        g=day(); g.loc[3,['high','low']]=[200.,10.]
        trade,_=s.simulate(g,3,1,101.,99.,'pine_close',1.)
        self.assertEqual(trade['exit_reason'],'square_off')

    def test_stop_first_collision_and_gap(self):
        g=day(); g.loc[4,['high','low']]=[110.,97.]
        trade,_=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertEqual(trade['exit_reason'],'stop')
        self.assertEqual(trade['exit_reference_price'],98.)
        g=day(); g.loc[5,'open']=95.
        trade,_=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertEqual(trade['exit_reason'],'stop_gap')
        self.assertEqual(trade['exit_reference_price'],95.)

    def test_invalid_initial_gap_and_market_target_at_entry(self):
        g=day(); g.loc[4,'open']=97.
        trade,reason=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertIsNone(trade)
        self.assertEqual(reason,'invalid_initial_stop_at_fill')
        g=day(); g.loc[4,'open']=106.
        trade,_=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertEqual(trade['exit_reason'],'target_gap_at_entry')
        self.assertEqual(trade['gross_points'],0.)

    def test_current_finalized_trail_never_hits_retroactively(self):
        g=day()
        g.loc[4,['open','high','low','close','trail']]=[103.,104.,99.,103.,102.]
        g.loc[5,['open','high','low','close']]=[103.,104.,101.,103.]
        trade,_=s.simulate(g,3,1,101.,99.,'next_open',1.)
        self.assertIn('09:40:00',trade['exit_timestamp'])
        self.assertEqual(trade['exit_reference_price'],102.)
        self.assertGreater(trade['trail_updates'],0)

    def test_short_trail_never_loosens(self):
        g=day(); g.loc[3,['close','trail']]=[98.,102.]
        g.loc[4,['open','high','low','close','trail']]=[98.,101.,96.,98.,100.]
        g.loc[5,['open','high','low','close','trail']]=[98.,99.,97.,98.,101.]
        trade,_=s.simulate(g,3,-1,101.,99.,'next_open',1.)
        self.assertLessEqual(trade['final_active_stop'],100.)

    def test_square_off_at_1500_and_cost_reconciliation(self):
        trade,_=s.simulate(day(),3,1,101.,99.,'next_open',1.)
        self.assertEqual(trade['exit_reason'],'square_off')
        self.assertIn('15:00:00',trade['exit_timestamp'])
        self.assertAlmostEqual(trade['net_pnl'],trade['gross_pnl']-trade['fees_pnl']-trade['slippage_pnl'])

    def test_future_changes_cannot_change_indicator_or_signal_features(self):
        g=day(); a=s.wilder_atr(g); trail,trend=s.vervoort(g.close.to_numpy(),a,1.)
        changed=g.copy(); changed.loc[20:,['high','low','close']]*=10
        b=s.wilder_atr(changed); other,_=s.vervoort(changed.close.to_numpy(),b,1.)
        np.testing.assert_allclose(a[:20],b[:20],equal_nan=True)
        np.testing.assert_allclose(trail[:20],other[:20],equal_nan=True)
        feature=s.features(g.iloc[3],g.iloc[:3],101.,99.,1)
        changed.loc[4:,'trail']=500.
        self.assertEqual(feature,s.features(changed.iloc[3],changed.iloc[:3],101.,99.,1))


if __name__=='__main__': unittest.main(verbosity=2)
