"""
长周期鸭口选股策略 V6
======================
所有价格指标使用前复权行情，成交量不复权。十组条件全部满足：
月/周 BOLL 历史形态、月或周 MACD 金叉保持、月周 OBV、周 DMA、
一年内月/周放量、月 KDJ 顺序金叉，以及月收盘价曾高于 MA5。
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import re
import requests
from datetime import datetime, timedelta
from jinja2 import Template
import time
from market_data import MarketDataError
from qfq_data import AkshareMarketData, SOURCE, beijing_now

# ============================================================
# HTTP 基础设施
# ============================================================

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# 腾讯接口仍会为部分退市代码返回多年前的最后一根 K 线。不能把这种
# 历史快照当作当天行情，否则技术指标会产生看似有效的假信号。
MAX_KLINE_STALENESS_DAYS = 14
MIN_DATA_COVERAGE_RATIO = 0.80


def has_recent_kline(df, max_staleness_days=MAX_KLINE_STALENESS_DAYS):
    """确认 K 线末日期足够新，排除退市或长期停止更新的代码。"""
    if df.empty or 'date' not in df:
        return False
    try:
        last_date = pd.to_datetime(df['date'].iloc[-1]).date()
    except (TypeError, ValueError):
        return False
    return last_date >= (datetime.now().date() - timedelta(days=max_staleness_days))


# ============================================================
# 数据获取层
# ============================================================

def get_all_a_stocks():
    """通过腾讯实时行情批量探测有效A股"""
    print("[1/4] 获取A股股票列表...")

    code_ranges = []
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(1, 1000)]
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(2001, 3000)]
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(300001, 302000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(600000, 602000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(603000, 604000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(605000, 606000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(688001, 690000)]

    all_stocks = []
    batch_size = 80

    for i in range(0, len(code_ranges), batch_size):
        batch = code_ranges[i:i + batch_size]
        query = ','.join(batch)
        url = f"https://qt.gtimg.cn/q={query}"
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code != 200:
                continue
            text = resp.text
            for entry in text.split(';'):
                entry = entry.strip()
                if not entry:
                    continue
                match = re.search(r'v_(\w+)="(\d+)~(.+?)~(\d+)~([^~]*)~', entry)
                if not match:
                    continue
                name = match.group(3).strip()
                code = match.group(4)
                price_str = match.group(5)
                if not name or not code or len(code) != 6:
                    continue
                if 'ST' in name or '退' in name or 'PT' in name:
                    continue
                try:
                    price = float(price_str)
                    if price <= 0:
                        continue
                except (ValueError, TypeError):
                    continue
                all_stocks.append({'代码': code, '名称': name})
        except Exception:
            continue

        if (i // batch_size) % 20 == 0 and i > 0:
            print(f"    已探测 {i}/{len(code_ranges)}，有效 {len(all_stocks)} 只...")
        time.sleep(0.05)

    df = pd.DataFrame(all_stocks)
    if df.empty:
        print("  股票列表获取失败!")
        return df
    df = df.drop_duplicates(subset='代码').reset_index(drop=True)
    print(f"  共 {len(df)} 只股票待筛选")
    return df


def _fetch_kline(symbol, period, count):
    """
    通用K线获取（腾讯财经前复权接口）
    period: 'week' / 'month' / 'day'
    返回 DataFrame(date, open, close, high, low, vol) 或空 DataFrame
    """
    period_map = {'week': 'qfqweek', 'month': 'qfqmonth', 'day': 'qfqday'}
    qfq_key = period_map.get(period, f'qfq{period}')

    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
        f"_var=kline_{period}qfq&param={symbol},{period},,,{count},qfq"
    )
    try:
        resp = SESSION.get(url, timeout=20)
        if resp.status_code != 200:
            return pd.DataFrame()
        text = resp.text.strip()
        if '=' in text:
            text = text.split('=', 1)[1]
        data = json.loads(text)
        if data.get('code') != 0:
            return pd.DataFrame()
        stock_data = data.get('data', {})
        if not stock_data:
            return pd.DataFrame()
        first_key = list(stock_data.keys())[0]
        klines = stock_data[first_key].get(qfq_key, [])
        if not klines:
            return pd.DataFrame()
        rows = []
        for k in klines:
            if len(k) >= 6:
                try:
                    rows.append({
                        'date':  k[0],
                        'open':  float(k[1]),
                        'close': float(k[2]),
                        'high':  float(k[3]),
                        'low':   float(k[4]),
                        'vol':   float(k[5]),
                    })
                except (ValueError, IndexError):
                    continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values('date').reset_index(drop=True)
        if not has_recent_kline(df):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def get_kline(stock_code, period, count):
    """统一接口：按股票代码 + 周期获取K线"""
    if stock_code.startswith(('60', '68')):
        symbol = f"sh{stock_code}"
    else:
        symbol = f"sz{stock_code}"
    return _fetch_kline(symbol, period, count)


def get_daily_display(stock_code):
    """获取最新实时行情（用于展示）"""
    if stock_code.startswith(('60', '68')):
        symbol = f"sh{stock_code}"
    else:
        symbol = f"sz{stock_code}"
    url = f"https://qt.gtimg.cn/q={symbol}"
    try:
        resp = SESSION.get(url, timeout=15)
        text = resp.text.strip()
        match = re.search(r'"(.+)"', text)
        if not match:
            return {}
        parts = match.group(1).split('~')
        if len(parts) < 40:
            return {}
        price = float(parts[3])
        prev_close = float(parts[4])
        change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        return {
            'price':      price,
            'change_pct': round(change_pct, 2),
            'volume':     float(parts[36]) if parts[36] else 0,
            'high':       float(parts[33]) if parts[33] else price,
            'low':        float(parts[34]) if parts[34] else price,
            'open':       float(parts[5])  if parts[5]  else price,
        }
    except Exception:
        return {}


# ============================================================
# 数据提供者：Tushare 全市场日线（本地聚合日/周/月线）
# ============================================================

MARKET_DATA = None


def prepare_market_data():
    global MARKET_DATA
    print("[1/4] 同步并校验 AKShare 前复权行情...")
    MARKET_DATA = AkshareMarketData().load()


def get_all_a_stocks():
    if MARKET_DATA is None:
        raise MarketDataError("行情数据尚未初始化")
    stocks = MARKET_DATA.active_stocks()
    print(f"  共 {len(stocks)} 只上市股票待筛选")
    return stocks


def get_kline(stock_code, period, count):
    return MARKET_DATA.bars(stock_code, period, count)


def get_daily_display(stock_code):
    return MARKET_DATA.display_quote(stock_code)


# ============================================================
# 技术指标计算工具
# ============================================================

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def ma(series, n):
    return series.rolling(window=n, min_periods=n).mean()

def std_dev(series, n):
    return series.rolling(window=n, min_periods=n).std(ddof=0)

def ref(series, n):
    return series.shift(n)

def exist(cond_series, n):
    """最近 n 个周期内是否出现过 True"""
    return cond_series.rolling(window=n, min_periods=1).max().astype(bool)

def cross_up(s1, s2):
    """s1 上穿 s2（金叉）"""
    return (s1 > s2) & (ref(s1, 1) <= ref(s2, 1))


# ============================================================
# 各指标计算函数
# ============================================================

def calc_boll_directions(df, period=20):
    """同花顺默认 BOLL(20, 2) 的上/中/下轨方向。"""
    close = df['close']
    mid   = ma(close, period)
    upper = mid + 2 * std_dev(close, period)
    lower = mid - 2 * std_dev(close, period)
    return {
        'upper_up': upper > ref(upper, 1),
        'mid_up': mid > ref(mid, 1),
        'lower_up': lower > ref(lower, 1),
        'lower_down': lower < ref(lower, 1),
    }


def calc_boll(df, period=20):
    """UB↑、BOLL中轨↑、LB↓。"""
    direction = calc_boll_directions(df, period)
    return direction['upper_up'] & direction['mid_up'] & direction['lower_down']


def calc_boll_week(df, period=20):
    """周线允许鸭口，或者 UB/BOLL/LB 三轨同时向上。"""
    direction = calc_boll_directions(df, period)
    duck = direction['upper_up'] & direction['mid_up'] & direction['lower_down']
    all_up = direction['upper_up'] & direction['mid_up'] & direction['lower_up']
    return duck | all_up


def calc_week_close_above_mid_after_latest_duck(df, lookback=52, period=20):
    """最近一年内最近一次周鸭口起，周收盘价始终严格高于BOLL中轨。"""
    duck = calc_boll(df, period)
    start = max(0, len(df) - lookback)
    positions = np.flatnonzero(duck.to_numpy())
    positions = positions[positions >= start]
    if len(positions) == 0:
        return False
    latest = int(positions[-1])
    mid = ma(df['close'], period)
    return bool((df['close'].iloc[latest:] > mid.iloc[latest:]).all())


def calc_macd(df, with_zero_filter=False):
    """
    返回：macd_cross_hold（Series[bool]）
    金叉后 DIF 始终 >= DEA
    with_zero_filter=True 时，要求金叉发生时 DEA > 0（零轴上方）
    """
    close = df['close']
    dif   = ema(close, 12) - ema(close, 26)
    dea   = ema(dif, 9)

    jc = cross_up(dif, dea)
    if with_zero_filter:
        jc = jc & (dea > 0)

    dif_above = (dif >= dea).astype(float)

    result = pd.Series(False, index=df.index)
    jc_idx = df.index[jc]
    for idx in jc_idx:
        subsequent = dif_above.loc[idx:]
        if subsequent.min() >= 1.0:
            result.loc[idx:] = True
    return result


def macd_recent_cross_hold(df, lookback, with_zero_filter=False):
    """窗口内确实发生金叉，且从该金叉到最新一期 DIF 始终在 DEA 之上。"""
    close = df['close']
    dif = ema(close, 12) - ema(close, 26)
    dea = ema(dif, 9)
    crossed = cross_up(dif, dea)
    if with_zero_filter:
        crossed = crossed & (dea > 0)
    window_start = max(0, len(df) - lookback)
    for position in np.flatnonzero(crossed.to_numpy()):
        if position >= window_start and bool((dif.iloc[position:] >= dea.iloc[position:]).all()):
            return True
    return False


def calc_obv(df, ma_period=20):
    """OBV > MA(OBV, ma_period)"""
    close = df['close']
    vol   = df['vol']
    direction = np.sign(close.diff().fillna(0))
    obv   = (direction * vol).cumsum()
    maobv = ma(obv, ma_period)
    return obv > maobv


def calc_dma(df):
    """
    DMA 指标：DIF_DMA > DIFMA
    DIF_DMA = MA(close,10) - MA(close,50)
    DIFMA   = MA(DIF_DMA, 10)
    """
    close   = df['close']
    dif_dma = ma(close, 10) - ma(close, 50)
    difma   = ma(dif_dma, 10)
    return dif_dma > difma


def calc_amo(df_month, df_week):
    """一年内月量至少一次 >=3倍前月，且周量至少一次 >=2倍前周。"""
    monthly = df_month['vol'] / ref(df_month['vol'], 1)
    weekly = df_week['vol'] / ref(df_week['vol'], 1)
    month_ok = bool((monthly.tail(12) >= 3.0).any())
    week_ok = bool((weekly.tail(52) >= 2.0).any())
    return month_ok and week_ok


def calc_kdj_values(df, n=9, m1=3, m2=3):
    """返回同花顺常用参数 KDJ(9,3,3) 的 K、D、J。"""
    high  = df['high']
    low   = df['low']
    close = df['close']

    low_n  = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()

    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    rsv = rsv.clip(0, 100)

    k = pd.Series(50.0, index=df.index)
    d = pd.Series(50.0, index=df.index)
    for i in range(1, len(df)):
        k.iloc[i] = k.iloc[i-1] * (1 - 1/m1) + rsv.iloc[i] * (1/m1)
        d.iloc[i] = d.iloc[i-1] * (1 - 1/m2) + k.iloc[i] * (1/m2)

    j = 3 * k - 2 * d

    return k, d, j


def calc_kdj_sequential(df, lookback=12):
    """允许 J上穿K、K上穿D先后发生，之后实际形成 J>K>D。"""
    k, d, j = calc_kdj_values(df)
    jk_cross = cross_up(j, k)
    kd_cross = cross_up(k, d)
    start = max(1, len(df) - lookback)
    seen_jk = False
    seen_kd = False
    for position in range(start, len(df)):
        seen_jk = seen_jk or bool(jk_cross.iloc[position])
        seen_kd = seen_kd or bool(kd_cross.iloc[position])
        if seen_jk and seen_kd and j.iloc[position] > k.iloc[position] > d.iloc[position]:
            return True
    return False


def calc_ma5_history(df_month, lookback=12):
    """近一年内至少一期月收盘价严格高于月 MA5。"""
    ma5 = ma(df_month['close'], 5)
    return bool((df_month['close'].tail(lookback) > ma5.tail(lookback)).any())


def calc_gap_up(df, lookback):
    """窗口内至少一次向上跳空：本期最低价比前一期最高价至少高0.02元。"""
    gap = df['low'] - ref(df['high'], 1)
    return bool((gap.tail(lookback) >= 0.02 - 1e-9).any())


def calc_circulating_market_cap(circ_mv_wan):
    """流通市值统一换算为万元；20亿至200亿元均含边界。"""
    value = float(circ_mv_wan)
    return 200_000.0 <= value <= 2_000_000.0


# ============================================================
# 主策略：十组长周期月/周线条件
# ============================================================

def evaluate_conditions(df_month, df_week, circ_mv_wan):
    """返回十组经用户确认的条件；策略只在所有条件为真时命中。"""
    boll_m = bool(exist(calc_boll(df_month), 12).iloc[-1])
    boll_w = bool(exist(calc_boll_week(df_week), 52).iloc[-1])
    macd_m = macd_recent_cross_hold(df_month, 12, with_zero_filter=False)
    macd_w = macd_recent_cross_hold(df_week, 52, with_zero_filter=True)
    obv_m = bool(calc_obv(df_month).iloc[-1])
    obv_w = bool(calc_obv(df_week).iloc[-1])
    dma_w = bool(calc_dma(df_week).iloc[-1])
    amo = calc_amo(df_month, df_week)
    kdj_m = calc_kdj_sequential(df_month, 12)
    ma5_m = calc_ma5_history(df_month, 12)
    cap = calc_circulating_market_cap(circ_mv_wan)
    gap_m = calc_gap_up(df_month, 12)
    gap_w = calc_gap_up(df_week, 52)
    boll_hold_w = calc_week_close_above_mid_after_latest_duck(df_week, 52)
    return {
        'BOLL': boll_m and boll_w,
        'MACD': macd_m or macd_w,
        'OBV': obv_m and obv_w,
        'DMA': dma_w,
        'AMO': amo,
        'KDJ': kdj_m,
        'MA5': ma5_m,
        'CAP': cap,
        'GAP': gap_m or gap_w,
        'BOLL_HOLD': boll_hold_w,
        '_parts': {
            'boll_month': boll_m, 'boll_week': boll_w,
            'macd_month': macd_m, 'macd_week': macd_w,
            'obv_month': obv_m, 'obv_week': obv_w,
            'gap_month': gap_m, 'gap_week': gap_w,
        },
    }


def apply_strategy(df_month, df_week, circ_mv_wan, df_day=None):
    conditions = evaluate_conditions(df_month, df_week, circ_mv_wan)
    return all(conditions[key] for key in
               ('BOLL', 'MACD', 'OBV', 'DMA', 'AMO', 'KDJ', 'MA5', 'CAP', 'GAP',
                'BOLL_HOLD'))


def apply_strategy_detail(df_month, df_week, circ_mv_wan, df_day=None):
    conditions = evaluate_conditions(df_month, df_week, circ_mv_wan)
    parts = conditions['_parts']
    mark = lambda value: '✓' if value else '✗'
    return {
        'BOLL': f"月{mark(parts['boll_month'])} 周{mark(parts['boll_week'])}",
        'MACD': f"月{mark(parts['macd_month'])} 或 周{mark(parts['macd_week'])}",
        'OBV': f"月{mark(parts['obv_month'])} 周{mark(parts['obv_week'])}",
        'DMA': f"周{mark(conditions['DMA'])}",
        'AMO': mark(conditions['AMO']),
        'KDJ': f"月{mark(conditions['KDJ'])}",
        'MA5': f"月{mark(conditions['MA5'])}",
        'CAP': f"{float(circ_mv_wan) / 10_000:.2f}亿 {mark(conditions['CAP'])}",
        'GAP': f"月{mark(parts['gap_month'])} 或 周{mark(parts['gap_week'])}",
        'BOLL_HOLD': f"周{mark(conditions['BOLL_HOLD'])}",
    }


# ============================================================
# 主流程
# ============================================================

MIN_MONTH = 30  # MACD(26)等月线指标需要窗口前的预热数据。
MIN_WEEK  = 60  # DMA(50,10)等周线指标需要窗口前的预热数据。

FETCH_MONTH = 60
FETCH_WEEK  = 260


def run_strategy():
    print("=" * 60)
    print(f"  长周期鸭口选股 V6 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    prepare_market_data()
    stocks = get_all_a_stocks()
    if stocks.empty:
        print("无法获取股票列表，退出")
        return []

    selected = []
    total    = len(stocks)
    failed   = 0
    market_caps = {ts_code[:6]: float(value)
                   for ts_code, value in MARKET_DATA.raw['circ_mv'].items()}
    if set(stocks['代码']) - set(market_caps):
        raise MarketDataError("流通市值覆盖不完整，禁止发布")

    print(f"\n[2/4] 逐只计算策略信号（共 {total} 只）...")
    for idx, row in stocks.iterrows():
        code = row['代码']
        name = row['名称']

        if idx % 200 == 0:
            print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")

        df_month = get_kline(code, 'month', FETCH_MONTH)
        df_week  = get_kline(code, 'week',  FETCH_WEEK)

        if (df_month.empty or len(df_month) < MIN_MONTH or
                df_week.empty or len(df_week) < MIN_WEEK):
            failed += 1
            MARKET_DATA.stats["insufficient_history"] += 1
            continue

        try:
            circ_mv_wan = market_caps[code]
            hit = apply_strategy(df_month, df_week, circ_mv_wan)
            MARKET_DATA.stats["evaluated"] += 1
            if hit:
                detail = apply_strategy_detail(df_month, df_week, circ_mv_wan)
                selected.append({
                    'code':   code,
                    'name':   name,
                    'detail': detail,
                    'circulating_market_cap_yi': round(circ_mv_wan / 10_000, 2),
                })
                print(f"  ★ 选中: {code} {name}")
        except Exception as e:
            MARKET_DATA.stats["errors"] += 1
            raise MarketDataError(f"策略计算异常 {code}，禁止发布") from e

    print(f"\n  策略计算完成: 成功 {total - failed}, 失败 {failed}")

    print(f"\n[3/4] 获取选中股票的最新行情...")
    for item in selected:
        daily = get_daily_display(item['code'])
        item.update(daily)

    print(f"\n  共选出 {len(selected)} 只股票")
    return selected


# ============================================================
# HTML 生成
# ============================================================

def generate_html(selected_stocks, output_path):
    print(f"\n[4/4] 生成展示页面...")

    template_str = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>长周期鸭口选股 V6</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    background: #080c1a;
    color: #dde3ff;
    min-height: 100vh;
    padding-bottom: env(safe-area-inset-bottom);
}
.header {
    background: linear-gradient(135deg, #131836 0%, #0a0f28 100%);
    padding: 18px 16px 14px;
    border-bottom: 1px solid rgba(90, 120, 255, 0.18);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(20px);
}
.header h1 {
    font-size: 21px;
    font-weight: 800;
    background: linear-gradient(90deg, #7ba4ff, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1.5px;
}
.header .meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 7px;
    font-size: 12px;
    color: #6870a0;
}
.header .count {
    background: rgba(90,120,255,0.15);
    color: #8fa4ff;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
}
.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 9px;
}
.tag {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.tag-v4    { background: rgba(255,100,100,0.18); color: #ff7070; border: 1px solid rgba(255,100,100,0.35); }
.tag-boll  { background: rgba(100,150,255,0.12); color: #7ba4ff; }
.tag-macd  { background: rgba(52,211,153,0.10);  color: #34d399; }
.tag-obv   { background: rgba(251,191,36,0.10);  color: #fbbf24; }
.tag-dma   { background: rgba(244,114,182,0.10); color: #f472b6; }
.tag-amo   { background: rgba(34,211,238,0.10);  color: #22d3ee; }
.tag-kdj   { background: rgba(167,139,250,0.12); color: #a78bfa; }
.tag-cap   { background: rgba(45,212,191,0.12); color: #5eead4; }
.tag-gap   { background: rgba(251,146,60,0.12); color: #fb923c; }
.strategy-desc {
    background: rgba(90,120,255,0.05);
    border: 1px solid rgba(90,120,255,0.12);
    border-radius: 10px;
    padding: 11px 13px;
    margin: 10px 12px 4px;
    font-size: 11px;
    color: #6870a0;
    line-height: 1.85;
}
.strategy-desc strong { color: #a0b0ff; }
.strategy-desc .v4-highlight { color: #ff8080; font-weight: 700; }
.disclaimer {
    background: rgba(234,179,8,0.06);
    border: 1px solid rgba(234,179,8,0.14);
    border-radius: 10px;
    padding: 10px 13px;
    margin: 4px 12px 4px;
    font-size: 11px;
    color: #a89040;
    line-height: 1.5;
}
.stock-list { padding: 10px 12px; }
.stock-card {
    background: linear-gradient(135deg, rgba(20,26,60,0.85) 0%, rgba(10,14,36,0.92) 100%);
    border: 1px solid rgba(90,120,255,0.10);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s;
}
.stock-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,120,120,0.5), transparent);
}
.stock-card:active { transform: scale(0.985); }
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.stock-name { font-size: 17px; font-weight: 700; color: #e6eaff; }
.stock-code {
    font-size: 12px;
    color: #525880;
    margin-top: 2px;
    font-family: 'SF Mono','Fira Code',monospace;
}
.stock-price { text-align: right; }
.price-value {
    font-size: 22px;
    font-weight: 700;
    font-family: 'SF Mono','DIN Alternate',monospace;
}
.price-change { font-size: 13px; font-weight: 600; margin-top: 1px; }
.up   { color: #f43f5e; }
.down { color: #10b981; }
.flat { color: #6870a0; }
.card-bottom {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
    padding-top: 11px;
    border-top: 1px solid rgba(90,120,255,0.07);
}
.metric { text-align: center; }
.metric-label { font-size: 10px; color: #525880; letter-spacing: 0.4px; }
.metric-value {
    font-size: 13px;
    color: #a0aacc;
    margin-top: 2px;
    font-family: 'SF Mono',monospace;
}
.signal-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px solid rgba(90,120,255,0.07);
}
.sig {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    font-family: 'SF Mono',monospace;
    white-space: nowrap;
}
.sig-v4   { background: rgba(255,100,100,0.15); color: #ff8080; border: 1px solid rgba(255,100,100,0.25); }
.sig-boll { background: rgba(100,150,255,0.1); color: #7ba4ff; }
.sig-macd { background: rgba(52,211,153,0.1);  color: #34d399; }
.sig-obv  { background: rgba(251,191,36,0.1);  color: #fbbf24; }
.sig-dma  { background: rgba(244,114,182,0.1); color: #f472b6; }
.sig-amo  { background: rgba(34,211,238,0.1);  color: #22d3ee; }
.sig-kdj  { background: rgba(167,139,250,0.1); color: #a78bfa; }
.sig-cap  { background: rgba(45,212,191,0.1); color: #5eead4; }
.sig-gap  { background: rgba(251,146,60,0.1); color: #fb923c; }
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #525880;
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; }
.empty-state p { font-size: 14px; line-height: 1.7; }
.footer {
    text-align: center;
    padding: 18px;
    font-size: 11px;
    color: #363b5a;
    border-top: 1px solid rgba(90,120,255,0.06);
    margin-top: 8px;
}
</style>
</head>
<body>
<div class="header">
    <h1>长周期鸭口选股 V6</h1>
    <div class="meta">
        <span>{{ update_time }}</span>
        <span class="count">{{ stock_count }} 只</span>
    </div>
    <div class="tags">
        <span class="tag tag-v4">★ 十组条件</span>
        <span class="tag tag-boll">BOLL 月12/周52</span>
        <span class="tag tag-macd">MACD 月或周</span>
        <span class="tag tag-obv">OBV 月/周</span>
        <span class="tag tag-dma">DMA 周</span>
        <span class="tag tag-amo">量能 月12/周52</span>
        <span class="tag tag-kdj">KDJ 月12</span>
        <span class="tag tag-v4">月收盘价 &gt; MA5</span>
        <span class="tag tag-cap">流通市值 20～200亿</span>
        <span class="tag tag-gap">一年内月/周向上跳空</span>
        <span class="tag tag-boll">周鸭口后收盘始终高于中轨</span>
    </div>
</div>

<div class="strategy-desc">
    <strong>策略逻辑（V6）：</strong>
    十组条件全部通过：近一年月线鸭口 + 周线鸭口或三轨向上；月线或零轴上周线 MACD
    金叉后保持；月周 OBV 均线上方；周 DMA；一年内月量≥3倍且周量≥2倍；
    一年内月 KDJ 先后上穿并形成 J&gt;K&gt;D；一年内月收盘价曾高于 MA5；
    流通市值20～200亿元；一年内月线或周线至少一次向上跳空≥0.02元；
    最近一次周线鸭口出现后至今，每周收盘价始终严格高于周BOLL中轨。
</div>

<div class="disclaimer" id="data-status">
    {{ data_status }}<br>价格：前复权；放量指标：成交量（非成交额）。
    <br><span id="freshness-warning"></span>
</div>
<script>
const dataDay = "{{ data_status }}".match(/\d{4}-\d{2}-\d{2}/);
if (dataDay && Date.now() - Date.parse(dataDay[0] + "T15:00:00+08:00") > 4*86400000) {
  document.getElementById("freshness-warning").textContent =
    "提示：行情日期距今超过4天，可能为休市或任务未更新，请核对运行状态。";
}
</script>
<div class="disclaimer">
    本页面仅为量化策略筛选结果展示，不构成任何投资建议。股市有风险，投资需谨慎。
</div>

<div class="stock-list">
{% if stocks %}
{% for s in stocks %}
<div class="stock-card">
    <div class="card-top">
        <div>
            <div class="stock-name">{{ s.name }}</div>
            <div class="stock-code">{{ s.code }}</div>
        </div>
        <div class="stock-price">
            {% if s.price %}
            <div class="price-value {% if s.change_pct > 0 %}up{% elif s.change_pct < 0 %}down{% else %}flat{% endif %}">
                {{ "%.2f"|format(s.price) }}
            </div>
            <div class="price-change {% if s.change_pct > 0 %}up{% elif s.change_pct < 0 %}down{% else %}flat{% endif %}">
                {% if s.change_pct > 0 %}+{% endif %}{{ "%.2f"|format(s.change_pct) }}%
            </div>
            {% else %}
            <div class="price-value flat">--</div>
            {% endif %}
        </div>
    </div>
    {% if s.price %}
    <div class="card-bottom">
        <div class="metric">
            <div class="metric-label">开盘</div>
            <div class="metric-value">{{ "%.2f"|format(s.open) }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">最高</div>
            <div class="metric-value">{{ "%.2f"|format(s.high) }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">最低</div>
            <div class="metric-value">{{ "%.2f"|format(s.low) }}</div>
        </div>
    </div>
    {% endif %}
    {% if s.detail %}
    <div class="signal-row">
        <span class="sig sig-boll">BOLL {{ s.detail.BOLL }}</span>
        <span class="sig sig-macd">MACD {{ s.detail.MACD }}</span>
        <span class="sig sig-obv">OBV {{ s.detail.OBV }}</span>
        <span class="sig sig-dma">DMA {{ s.detail.DMA }}</span>
        <span class="sig sig-amo">成交量 {{ s.detail.AMO }}</span>
        <span class="sig sig-kdj">KDJ {{ s.detail.KDJ }}</span>
        <span class="sig sig-v4">MA5 {{ s.detail.MA5 }}</span>
        <span class="sig sig-cap">流通市值 {{ s.detail.CAP }}</span>
        <span class="sig sig-gap">跳空 {{ s.detail.GAP }}</span>
        <span class="sig sig-boll">鸭口后周收盘&gt;中轨 {{ s.detail.BOLL_HOLD }}</span>
    </div>
    {% endif %}
</div>
{% endfor %}
{% else %}
<div class="empty-state">
    <div class="icon">📊</div>
    <p>今日暂无符合策略的股票<br>策略每个交易日收盘后自动更新</p>
</div>
{% endif %}
</div>

<div class="footer">
    <p>长周期鸭口选股 V6 · 十组筛选条件 · 数据来源：腾讯财经 / Tushare</p>
    <p style="margin-top:4px;">每个交易日收盘后自动更新</p>
</div>
</body>
</html>"""

    template = Template(template_str)
    html = template.render(
        stocks=selected_stocks,
        stock_count=len(selected_stocks),
        update_time=beijing_now().strftime('%Y年%m月%d日 %H:%M 北京时间更新'),
        data_status=MARKET_DATA.page_status(),
    )
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  页面已生成: {output_path}")


def save_data_json(selected_stocks, output_path):
    """保存选股结果为 JSON"""
    data = {
        'update_time': beijing_now().strftime('%Y-%m-%d %H:%M:%S'),
        'timezone': 'Asia/Shanghai',
        'trade_date': MARKET_DATA.trade_date,
        'data_quality': MARKET_DATA.metadata(),
        'strategy': '长周期鸭口选股 V6',
        'conditions': {
            'BOLL': '月12期内UB↑/MID↑/LB↓，且周52期内同形态或三轨均↑',
            'MACD': '月12期金叉保持，或周52期零轴上金叉保持',
            'OBV': '最新月线及周线 OBV>MAOBV(20)',
            'DMA': '最新周线 DIF_DMA>DIFMA',
            'AMO': '成交量（非成交额）：月12期内≥前月3倍且周52期内≥前周2倍',
            'KDJ': '月12期内J/K与K/D允许先后上穿，最终形成J>K>D',
            'MA5': '月12期内至少一期收盘价>月MA5',
            'CAP': '交易日流通市值20亿～200亿元（含边界）',
            'GAP': '近12个月或52周内至少一次本期最低价≥前一期最高价+0.02元',
            'BOLL_HOLD': '近52周最近一次周线鸭口出现后，每周收盘价始终严格高于周BOLL中轨',
        },
        'adjustment': 'qfq',
        'data_source': SOURCE,
        'count': len(selected_stocks),
        'stocks': selected_stocks,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  数据已保存: {output_path}")


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)

    results = run_strategy()

    html_path = os.path.join(output_dir, 'index.html')
    generate_html(results, html_path)

    json_path = os.path.join(output_dir, 'data.json')
    save_data_json(results, json_path)

    print(f"\n{'=' * 60}")
    print(f"  完成! 共选出 {len(results)} 只股票")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print(f"{'=' * 60}")
