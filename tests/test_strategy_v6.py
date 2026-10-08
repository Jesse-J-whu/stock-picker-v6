import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('strategy_v6_tests', ROOT / 'strategy.py')
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


def frame(length, volume=100.0):
    close = np.linspace(10, 20, length)
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=length, freq='D'),
        'open': close, 'close': close, 'high': close + 1, 'low': close - 1,
        'vol': np.full(length, volume, dtype=float),
    })


class V6StrategyTests(unittest.TestCase):
    def test_week_boll_accepts_duck_or_all_up(self):
        index = pd.RangeIndex(3)
        with patch.object(strategy, 'calc_boll_directions', return_value={
                'upper_up': pd.Series([False, True, True], index=index),
                'mid_up': pd.Series([False, True, True], index=index),
                'lower_up': pd.Series([False, False, True], index=index),
                'lower_down': pd.Series([False, True, False], index=index)}):
            self.assertEqual(strategy.calc_boll_week(pd.DataFrame(index=index)).tolist(),
                             [False, True, True])

    def test_week_close_stays_above_mid_after_latest_duck(self):
        data = frame(110)
        data['close'] = 11.0
        ducks = pd.Series(False, index=data.index)
        ducks.iloc[100] = True
        mid = pd.Series(10.0, index=data.index)
        with patch.object(strategy, 'calc_boll', return_value=ducks), \
                patch.object(strategy, 'ma', return_value=mid):
            self.assertTrue(strategy.calc_week_close_above_mid_after_latest_duck(data))
            data.loc[105, 'close'] = 10.0
            self.assertFalse(strategy.calc_week_close_above_mid_after_latest_duck(data))

    def test_week_hold_requires_actual_recent_duck(self):
        data = frame(110)
        ducks = pd.Series(False, index=data.index)
        with patch.object(strategy, 'calc_boll', return_value=ducks):
            self.assertFalse(strategy.calc_week_close_above_mid_after_latest_duck(data))

    def test_volume_thresholds_are_inclusive_and_both_required(self):
        month, week = frame(60), frame(209)
        month.loc[59, 'vol'] = month.loc[58, 'vol'] * 3
        week.loc[208, 'vol'] = week.loc[207, 'vol'] * 2
        self.assertTrue(strategy.calc_amo(month, week))
        week.loc[208, 'vol'] = week.loc[207, 'vol'] * 1.99
        self.assertFalse(strategy.calc_amo(month, week))

    def test_volume_event_outside_window_does_not_count(self):
        month, week = frame(60), frame(220)
        # Events inside the old multi-year windows but outside the new one-year windows.
        month.loc[45, 'vol'] = month.loc[44, 'vol'] * 3
        week.loc[160, 'vol'] = week.loc[159, 'vol'] * 2
        self.assertFalse(strategy.calc_amo(month, week))

    def test_macd_requires_recent_real_cross_and_hold(self):
        data = frame(30)
        dif = pd.Series([-1.0] * 10 + [1.0] * 20)
        dea = pd.Series([0.0] * 30)
        zero = pd.Series(0.0, index=dif.index)
        with patch.object(strategy, 'ema', side_effect=[dif, zero, dea]):
            self.assertTrue(strategy.macd_recent_cross_hold(data, 24, False))
        dif.iloc[-1] = -1
        with patch.object(strategy, 'ema', side_effect=[dif, zero, dea]):
            self.assertFalse(strategy.macd_recent_cross_hold(data, 24, False))

    def test_week_macd_cross_must_be_above_zero(self):
        data = frame(30)
        dif = pd.Series([-2.0] * 10 + [0.0] * 20)
        dea = pd.Series([-1.0] * 30)
        with patch.object(strategy, 'ema', side_effect=[dif, pd.Series(0.0, index=dif.index), dea]):
            self.assertFalse(strategy.macd_recent_cross_hold(data, 24, True))

    def test_kdj_allows_crosses_on_successive_months(self):
        data = frame(4)
        k = pd.Series([50.0, 50.0, 55.0, 55.0])
        d = pd.Series([55.0, 55.0, 50.0, 50.0])
        j = pd.Series([45.0, 60.0, 60.0, 60.0])
        with patch.object(strategy, 'calc_kdj_values', return_value=(k, d, j)):
            self.assertTrue(strategy.calc_kdj_sequential(data, 4))

    def test_kdj_static_order_without_cross_does_not_count(self):
        data = frame(4)
        k = pd.Series([60.0] * 4)
        d = pd.Series([50.0] * 4)
        j = pd.Series([70.0] * 4)
        with patch.object(strategy, 'calc_kdj_values', return_value=(k, d, j)):
            self.assertFalse(strategy.calc_kdj_sequential(data, 4))

    def test_market_cap_boundaries_are_inclusive(self):
        self.assertTrue(strategy.calc_circulating_market_cap(200_000))
        self.assertTrue(strategy.calc_circulating_market_cap(2_000_000))
        self.assertFalse(strategy.calc_circulating_market_cap(199_999.99))
        self.assertFalse(strategy.calc_circulating_market_cap(2_000_000.01))

    def test_gap_is_at_least_two_cents_inside_window(self):
        data = frame(26)
        data['high'] = 10.00
        data['low'] = 9.00
        data.loc[25, 'low'] = 10.02
        self.assertTrue(strategy.calc_gap_up(data, 24))
        data.loc[25, 'low'] = 10.019
        self.assertFalse(strategy.calc_gap_up(data, 24))

    def test_gap_outside_window_does_not_count(self):
        data = frame(30)
        data['high'] = 10.00
        data['low'] = 9.00
        data.loc[5, 'low'] = 10.02
        self.assertFalse(strategy.calc_gap_up(data, 24))

    def test_all_ten_groups_are_required(self):
        conditions = {'BOLL': True, 'MACD': True, 'OBV': True, 'DMA': True,
                      'AMO': True, 'KDJ': True, 'MA5': True, 'CAP': True,
                      'GAP': True, 'BOLL_HOLD': True, '_parts': {}}
        with patch.object(strategy, 'evaluate_conditions', return_value=conditions):
            self.assertTrue(strategy.apply_strategy(frame(60), frame(209), 200_000))
        conditions['BOLL_HOLD'] = False
        with patch.object(strategy, 'evaluate_conditions', return_value=conditions):
            self.assertFalse(strategy.apply_strategy(frame(60), frame(209), 200_000))


if __name__ == '__main__':
    unittest.main()
