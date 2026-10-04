"""
鸭口选股 V4 — 历史回测脚本 v2
==============================================
改进：
1. 直接调用腾讯接口获取历史周线/月线（不从日线聚合）
2. 修复沪深300基准数据获取
3. 滑窗回测：对历史上每个交易日截断K线数据，模拟策略信号
4. 统计选出后 5/10/22 个交易日（约1周/2周/1月）的收益表现

运行：
    python backtest.py                   # 全量（约需1~2小时）
    python backtest.py --sample 300      # 采样300只，快速验证
    python backtest.py --workers 4       # 多线程
"""

import os, sys, re, time, json, math, argparse, threading, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from jinja2 import Template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import (
    SESSION, get_kline, get_all_a_stocks,
    calc_boll, calc_boll_current, calc_macd, calc_obv,
    calc_dma, calc_amo, calc_kdj, exist,
)

# ============================================================
# 配置
# ============================================================
HOLD_DAYS    = [5, 10, 22]      # 持有期（交易日）
BENCHMARK    = "sh000300"        # 沪深300

# 策略计算所需的最少K线数
MIN_DAY   = 80
MIN_WEEK  = 60
MIN_MONTH = 30

# 回测时每只股票拉多少根K线（够两年+ 热身期）
FETCH_DAY   = 700
FETCH_WEEK  = 200
FETCH_MONTH = 80

# 滑窗起始偏移（至少保证策略有足够热身数据）
SCAN_START_DAY   = 100   # 日线从第100根开始扫
SCAN_START_WEEK  = 60
SCAN_START_MONTH = 30

# 最近回测区间（日线，约2年）
BACKTEST_WINDOW = 500

# ============================================================
# 工具：从 DataFrame 构造日期->收盘价 映射
# ============================================================

def build_close_map(df):
    return dict(zip(df['date'], df['close']))


def nth_future_close(close_list, current_idx, n):
    """在 close_list 中，current_idx 之后第 n 个交易日的收盘价"""
    target = current_idx + n
    if target < len(close_list):
        return close_list[target][1]
    return None


# ============================================================
# 沪深300基准
# ============================================================

def get_benchmark():
    """返回 {date_str: close} 字典"""
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
        f"_var=kline_bm&param={BENCHMARK},day,,,800,qfq"
    )
    try:
        r = SESSION.get(url, timeout=30)
        text = r.text.strip()
        if '=' in text:
            text = text.split('=', 1)[1]
        data = json.loads(text)
        code_key = BENCHMARK
        # 尝试两种键名
        klines = (data['data'].get(code_key, {}).get('qfqday')
                  or data['data'].get(code_key, {}).get('day') or [])
        return {k[0]: float(k[2]) for k in klines if len(k) >= 3}
    except Exception as e:
        print(f"  [WARN] 基准获取失败: {e}")
        return {}


# ============================================================
# 核心：对单只股票做历史信号扫描
# ============================================================

def scan_one_stock(code, name, benchmark_map):
    """
    对一只股票执行滑窗回测：
    - 获取足量日线/周线/月线
    - 从第 SCAN_START_DAY 根日线开始，逐日截断数据，运行策略
    - 记录信号日 & 之后 5/10/22 日收益
    返回信号列表（list of dict）
    """
    # 获取 K 线
    df_day   = get_kline(code, 'day',   FETCH_DAY)
    df_week  = get_kline(code, 'week',  FETCH_WEEK)
    df_month = get_kline(code, 'month', FETCH_MONTH)

    if (df_day.empty   or len(df_day)   < MIN_DAY + max(HOLD_DAYS)
     or df_week.empty  or len(df_week)  < MIN_WEEK
     or df_month.empty or len(df_month) < MIN_MONTH):
        return []

    # 日线数据（带索引方便定位）
    day_dates  = df_day['date'].tolist()
    day_closes = df_day['close'].tolist()
    n_days     = len(day_dates)

    signals = []
    prev_hit = False  # 连续信号去重

    # 滑窗范围：只回测最近 BACKTEST_WINDOW 根日线
    scan_start = max(SCAN_START_DAY, n_days - BACKTEST_WINDOW)
    scan_end   = n_days - max(HOLD_DAYS) - 1

    if scan_start >= scan_end:
        return []

    for i in range(scan_start, scan_end):
        current_date = day_dates[i]

        # 截断日线
        df_d_slice = df_day.iloc[:i+1].reset_index(drop=True)

        # ── 最轻量判断：当前日线鸭口（先过滤，节省时间）
        if not calc_boll_current(df_d_slice):
            prev_hit = False
            continue

        # ── 找对应的周线/月线截断点（按日期）
        # 周线截断：取日期 <= current_date 的所有周线
        df_w_slice = df_week[df_week['date'] <= current_date].reset_index(drop=True)
        df_m_slice = df_month[df_month['date'] <= current_date].reset_index(drop=True)

        if len(df_w_slice) < MIN_WEEK or len(df_m_slice) < MIN_MONTH:
            prev_hit = False
            continue

        # ── 完整策略判断
        try:
            hit = _full_strategy(df_m_slice, df_w_slice, df_d_slice)
        except Exception:
            prev_hit = False
            continue

        if not hit:
            prev_hit = False
            continue

        # 连续信号去重：只记录信号起始日
        if prev_hit:
            prev_hit = True
            continue

        prev_hit = True

        # ── 计算持有收益
        entry_price = day_closes[i]
        if not entry_price or entry_price <= 0:
            continue

        record = {
            'code':        code,
            'name':        name,
            'signal_date': current_date,
            'entry_price': round(entry_price, 3),
        }

        bm_entry = benchmark_map.get(current_date)
        for hold in HOLD_DAYS:
            future_idx = i + hold
            if future_idx < n_days:
                exit_price = day_closes[future_idx]
                future_date = day_dates[future_idx]
                if exit_price and exit_price > 0:
                    ret = (exit_price - entry_price) / entry_price * 100
                    bm_exit = benchmark_map.get(future_date)
                    if bm_entry and bm_entry > 0 and bm_exit and bm_exit > 0:
                        bm_ret = (bm_exit - bm_entry) / bm_entry * 100
                        alpha  = ret - bm_ret
                    else:
                        bm_ret = alpha = None
                    record[f'ret_{hold}d']    = round(ret, 3)
                    record[f'bm_{hold}d']     = round(bm_ret, 3) if bm_ret is not None else None
                    record[f'alpha_{hold}d']  = round(alpha, 3)  if alpha  is not None else None
                else:
                    record[f'ret_{hold}d'] = record[f'bm_{hold}d'] = record[f'alpha_{hold}d'] = None
            else:
                record[f'ret_{hold}d'] = record[f'bm_{hold}d'] = record[f'alpha_{hold}d'] = None

        signals.append(record)

    return signals


def _full_strategy(df_month, df_week, df_day):
    """完整 V4 策略判断（与 strategy.py apply_strategy 逻辑一致）"""
    # V4：当前三周期鸭口
    if not (calc_boll_current(df_day)
         and calc_boll_current(df_week)
         and calc_boll_current(df_month)):
        return False

    # BOLL 历史
    if not bool(exist(calc_boll(df_month), 12).iloc[-1]): return False
    if not bool(exist(calc_boll(df_week),  26).iloc[-1]): return False
    if not bool(exist(calc_boll(df_day),   22).iloc[-1]): return False

    # MACD
    if not bool(exist(calc_macd(df_month, False), 12).iloc[-1]): return False
    if not bool(exist(calc_macd(df_week,  True),  26).iloc[-1]): return False
    if not bool(exist(calc_macd(df_day,   False), 22).iloc[-1]): return False

    # OBV
    if not bool(calc_obv(df_month).iloc[-1]): return False
    if not bool(calc_obv(df_week).iloc[-1]):  return False
    if not bool(calc_obv(df_day).iloc[-1]):   return False

    # DMA（仅月/周）
    if not bool(calc_dma(df_month).iloc[-1]): return False
    if not bool(calc_dma(df_week).iloc[-1]):  return False

    # AMO
    if not calc_amo(df_week, df_day): return False

    # KDJ
    if not bool(exist(calc_kdj(df_month), 24).iloc[-1]): return False
    if not bool(exist(calc_kdj(df_week),  26).iloc[-1]): return False
    if not bool(exist(calc_kdj(df_day),   22).iloc[-1]): return False

    return True


# ============================================================
# 主流程
# ============================================================

def run_backtest(max_stocks=None, max_workers=3):
    print("=" * 60)
    print(f"  鸭口选股 V4 — 历史回测 v2")
    print(f"  回测区间：最近约 {BACKTEST_WINDOW} 个交易日（≈2年）")
    print(f"  持有期：{HOLD_DAYS} 个交易日 ≈ 1周/2周/1月")
    print("=" * 60)

    # 基准
    print("\n[1/4] 获取沪深300基准...")
    bm_map = get_benchmark()
    print(f"  沪深300：{len(bm_map)} 个交易日")

    # 股票列表
    print("\n[2/4] 获取A股列表...")
    stocks_df = get_all_a_stocks()
    if stocks_df.empty:
        print("  失败，退出"); return None

    if max_stocks:
        stocks_df = stocks_df.head(max_stocks)
        print(f"  [采样] 仅处理前 {max_stocks} 只")
    print(f"  共 {len(stocks_df)} 只股票")

    # 扫描
    print(f"\n[3/4] 历史信号扫描（workers={max_workers}）...")
    all_signals = []
    lock = threading.Lock()
    done = [0]
    total = len(stocks_df)

    def process(row):
        sigs = []
        try:
            sigs = scan_one_stock(row['代码'], row['名称'], bm_map)
        except Exception as e:
            pass
        with lock:
            all_signals.extend(sigs)
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == total:
                print(f"  进度: {done[0]}/{total} ({done[0]/total*100:.1f}%)  "
                      f"信号: {len(all_signals)}")
        time.sleep(0.05)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process, row) for _, row in stocks_df.iterrows()]
        for f in as_completed(futures):
            pass

    print(f"\n  扫描完成，共 {len(all_signals)} 个历史信号")
    if not all_signals:
        print("  未发现信号（策略条件严苛或样本太小）")
        return None

    # 统计
    print("\n[4/4] 统计分析...")
    df = pd.DataFrame(all_signals)
    stats = compute_stats(df)
    return df, stats


# ============================================================
# 统计
# ============================================================

def compute_stats(df):
    stats = {
        'total_signals': len(df),
        'total_stocks':  df['code'].nunique(),
        'date_range':    f"{df['signal_date'].min()} ~ {df['signal_date'].max()}",
    }
    for hold in HOLD_DAYS:
        col   = f'ret_{hold}d'
        a_col = f'alpha_{hold}d'
        sub   = df[col].dropna()
        asub  = df[a_col].dropna()
        if len(sub) == 0:
            stats[hold] = {}; continue
        stats[hold] = {
            'n':          len(sub),
            'win_rate':   round((sub > 0).mean() * 100, 1),
            'avg_ret':    round(sub.mean(), 2),
            'median_ret': round(sub.median(), 2),
            'beat_bm':    round((asub > 0).mean() * 100, 1) if len(asub) > 0 else None,
            'avg_alpha':  round(asub.mean(), 2) if len(asub) > 0 else None,
            'p10':  round(sub.quantile(.10), 2),
            'p25':  round(sub.quantile(.25), 2),
            'p75':  round(sub.quantile(.75), 2),
            'p90':  round(sub.quantile(.90), 2),
            'max':  round(sub.max(), 2),
            'min':  round(sub.min(), 2),
            'sharpe': round(sub.mean() / sub.std() * (252/hold)**0.5, 2) if sub.std() > 0 else None,
            'dist': _dist(sub),
        }
    return stats


def _dist(series, bins=22):
    counts, edges = np.histogram(series.clip(-30, 50), bins=bins)
    return {
        'labels': [f"{edges[i]:.1f}~{edges[i+1]:.1f}%" for i in range(len(edges)-1)],
        'counts': counts.tolist(),
        'mid':    [round((edges[i]+edges[i+1])/2, 1) for i in range(len(edges)-1)],
    }


# ============================================================
# 保存 & HTML
# ============================================================

def save_and_report(df, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path  = os.path.join(out_dir, 'backtest_signals.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    json_path = os.path.join(out_dir, 'backtest_stats.json')
    _save_json(stats, json_path)

    html_path = os.path.join(out_dir, 'backtest_report.html')
    _gen_html(df, stats, html_path)

    return csv_path, json_path, html_path


def _save_json(stats, path):
    out = {k: v for k, v in stats.items() if not isinstance(k, int)}
    for h in HOLD_DAYS:
        if h in stats and stats[h]:
            s = {k: v for k, v in stats[h].items() if k != 'dist'}
            out[f'{h}d'] = s
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


TMPL = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>鸭口选股 V4 — 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     background:#080c1a;color:#dde3ff;min-height:100vh}
.hdr{background:linear-gradient(135deg,#131836,#0a0f28);padding:22px 24px 16px;
     border-bottom:1px solid rgba(90,120,255,.18)}
.hdr h1{font-size:22px;font-weight:800;
  background:linear-gradient(90deg,#7ba4ff,#c084fc,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr .sub{font-size:12px;color:#6870a0;margin-top:6px;line-height:1.6}
.sec{padding:20px 20px 4px}
.sec h2{font-size:15px;font-weight:700;color:#a0b0ff;margin-bottom:14px;
         border-left:3px solid #7ba4ff;padding-left:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}
.card{background:linear-gradient(135deg,rgba(20,26,60,.85),rgba(10,14,36,.92));
      border:1px solid rgba(90,120,255,.12);border-radius:12px;padding:16px}
.card .lbl{font-size:11px;color:#6870a0;letter-spacing:.4px}
.card .val{font-size:26px;font-weight:700;margin-top:4px;font-family:'SF Mono',monospace}
.card .sub{font-size:11px;color:#525880;margin-top:2px}
.red{color:#f43f5e}.green{color:#34d399}.blue{color:#7ba4ff}.purple{color:#c084fc}.yellow{color:#fbbf24}
.period-block{margin-bottom:24px}
.period-title{font-size:14px;font-weight:700;color:#c084fc;margin-bottom:12px;
              display:flex;align-items:center;gap:8px}
.period-title span{font-size:11px;color:#6870a0;font-weight:400}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:12px}
.si{background:rgba(13,17,40,.6);border:1px solid rgba(90,120,255,.08);
    border-radius:8px;padding:10px 12px}
.si .sl{font-size:10px;color:#525880}
.si .sv{font-size:17px;font-weight:700;margin-top:3px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px}
.cb{background:rgba(13,17,40,.7);border:1px solid rgba(90,120,255,.1);
    border-radius:10px;padding:14px}
.cb h3{font-size:11px;color:#6870a0;margin-bottom:10px}
.cb canvas{max-height:200px}
@media(max-width:700px){.chart-row{grid-template-columns:1fr}}
.tbl-wrap{overflow-x:auto;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:rgba(20,26,60,.9);color:#6870a0;padding:8px 10px;text-align:left;
   border-bottom:1px solid rgba(90,120,255,.1);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid rgba(90,120,255,.05);color:#a0aacc}
tr:hover td{background:rgba(90,120,255,.05)}
.pos{color:#f43f5e;font-weight:600}.neg{color:#34d399;font-weight:600}
.footer{text-align:center;padding:20px;font-size:11px;color:#363b5a;
        border-top:1px solid rgba(90,120,255,.06);margin-top:10px}
</style>
</head>
<body>
<div class="hdr">
  <h1>鸭口选股 V4 — 历史回测报告</h1>
  <div class="sub">
    回测区间：{{ stats.date_range }} ·
    信号总数：{{ stats.total_signals }} · 涉及股票：{{ stats.total_stocks }} 只 ·
    生成时间：{{ gen_time }}
  </div>
</div>

<div class="sec">
  <h2>总览</h2>
  <div class="cards">
    <div class="card">
      <div class="lbl">历史信号总数</div>
      <div class="val blue">{{ stats.total_signals }}</div>
      <div class="sub">涉及 {{ stats.total_stocks }} 只个股</div>
    </div>
    {% for h,lbl in [(5,'1周(5日)'),(10,'2周(10日)'),(22,'1月(22日)')] %}{% if stats[h] %}
    <div class="card">
      <div class="lbl">{{ lbl }} 平均收益</div>
      <div class="val {% if stats[h].avg_ret > 0 %}red{% elif stats[h].avg_ret < 0 %}green{% else %}blue{% endif %}">
        {% if stats[h].avg_ret > 0 %}+{% endif %}{{ stats[h].avg_ret }}%</div>
      <div class="sub">胜率 {{ stats[h].win_rate }}%
        {% if stats[h].avg_alpha is not none %} · 超额 {{ stats[h].avg_alpha }}%{% endif %}</div>
    </div>
    {% endif %}{% endfor %}
  </div>
</div>

{% for h,lbl in [(5,'1周 (持有5个交易日)'),(10,'2周 (持有10个交易日)'),(22,'1个月 (持有22个交易日)')] %}
{% if stats[h] %}{% set s=stats[h] %}
<div class="sec">
  <h2>{{ lbl }}</h2>
  <div class="period-block">
    <div class="sg">
      <div class="si"><div class="sl">样本数</div><div class="sv blue">{{ s.n }}</div></div>
      <div class="si"><div class="sl">胜率（正收益）</div>
        <div class="sv {% if s.win_rate>=50 %}red{% else %}green{% endif %}">{{ s.win_rate }}%</div></div>
      <div class="si"><div class="sl">平均收益</div>
        <div class="sv {% if s.avg_ret>0 %}red{% else %}green{% endif %}">
          {% if s.avg_ret>0 %}+{% endif %}{{ s.avg_ret }}%</div></div>
      <div class="si"><div class="sl">中位数</div>
        <div class="sv {% if s.median_ret>0 %}red{% else %}green{% endif %}">
          {% if s.median_ret>0 %}+{% endif %}{{ s.median_ret }}%</div></div>
      <div class="si"><div class="sl">跑赢基准比例</div>
        <div class="sv {% if s.beat_bm and s.beat_bm>=50 %}red{% else %}green{% endif %}">
          {{ s.beat_bm if s.beat_bm else '--' }}%</div></div>
      <div class="si"><div class="sl">平均超额收益</div>
        <div class="sv {% if s.avg_alpha and s.avg_alpha>0 %}red{% else %}green{% endif %}">
          {% if s.avg_alpha %}{% if s.avg_alpha>0 %}+{% endif %}{{ s.avg_alpha }}%{% else %}--{% endif %}</div></div>
      <div class="si"><div class="sl">P10 / P25</div>
        <div class="sv" style="font-size:14px;color:#a0aacc">{{ s.p10 }}% / {{ s.p25 }}%</div></div>
      <div class="si"><div class="sl">P75 / P90</div>
        <div class="sv" style="font-size:14px;color:#a0aacc">{{ s.p75 }}% / {{ s.p90 }}%</div></div>
      <div class="si"><div class="sl">最大 / 最小</div>
        <div class="sv" style="font-size:14px;color:#a0aacc">{{ s.max }}% / {{ s.min }}%</div></div>
      <div class="si"><div class="sl">年化 Sharpe</div>
        <div class="sv blue">{{ s.sharpe if s.sharpe else '--' }}</div></div>
    </div>
    <div class="chart-row">
      <div class="cb">
        <h3>收益率分布直方图（-30%～+50%）</h3>
        <canvas id="hist{{ h }}"></canvas>
      </div>
      <div class="cb">
        <h3>分位数概览</h3>
        <canvas id="pct{{ h }}"></canvas>
      </div>
    </div>
  </div>
</div>
{% endif %}{% endfor %}

<div class="sec">
  <h2>最近信号明细（最新 {{ recent|length }} 条）</h2>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>日期</th><th>代码</th><th>名称</th><th>入场价</th>
        <th>5日收益</th><th>超额</th>
        <th>10日收益</th><th>超额</th>
        <th>22日收益</th><th>超额</th>
      </tr></thead>
      <tbody>
      {% for r in recent %}
      <tr>
        <td>{{ r.signal_date }}</td>
        <td>{{ r.code }}</td>
        <td>{{ r.name }}</td>
        <td>{{ "%.2f"|format(r.entry_price) }}</td>
        {% for h in [5,10,22] %}
        {% set ret=r.get('ret_'+h|string+'d') %}
        {% set alp=r.get('alpha_'+h|string+'d') %}
        <td class="{% if ret is not none and ret>0 %}pos{% elif ret is not none and ret<0 %}neg{% endif %}">
          {% if ret is not none %}{% if ret>0 %}+{% endif %}{{ "%.2f"|format(ret) }}%{% else %}--{% endif %}</td>
        <td style="color:#525880;font-size:11px">
          {% if alp is not none %}{% if alp>0 %}+{% endif %}{{ "%.2f"|format(alp) }}%{% else %}--{% endif %}</td>
        {% endfor %}
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<div class="footer">鸭口选股 V4 回测报告 · 仅供研究参考，不构成投资建议</div>

<script>
const D = {{ chart_json }};
function hist(id, dist) {
  if (!dist || !document.getElementById(id)) return;
  const colors = dist.mid.map(m => m >= 0 ? 'rgba(244,63,94,0.72)' : 'rgba(16,185,129,0.72)');
  new Chart(document.getElementById(id), {
    type:'bar',
    data:{ labels:dist.labels, datasets:[{data:dist.counts,backgroundColor:colors,borderWidth:0,borderRadius:2}] },
    options:{ responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#525880',font:{size:8},maxRotation:45}, grid:{color:'rgba(90,120,255,.04)'}},
               y:{ticks:{color:'#525880',font:{size:9}}, grid:{color:'rgba(90,120,255,.04)'}} } }
  });
}
function pct(id, s) {
  if (!s || !document.getElementById(id)) return;
  new Chart(document.getElementById(id), {
    type:'bar',
    data:{ labels:['P10','P25','中位数','平均','P75','P90'],
      datasets:[{
        data:[s.p10,s.p25,s.median_ret,s.avg_ret,s.p75,s.p90],
        backgroundColor:['#3b82f6','#6366f1','#8b5cf6','#f59e0b','#f43f5e','#ef4444'],
        borderWidth:0, borderRadius:4
      }]
    },
    options:{ responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#525880',font:{size:9}}, grid:{color:'rgba(90,120,255,.04)'}},
               y:{ticks:{color:'#525880',font:{size:9}}, grid:{color:'rgba(90,120,255,.04)'}} } }
  });
}
{% for h in [5,10,22] %}
hist('hist{{ h }}', D['dist{{ h }}']);
pct('pct{{ h }}',  D['stats{{ h }}']);
{% endfor %}
</script>
</body></html>"""


def _gen_html(df, stats, path):
    recent = df.sort_values('signal_date', ascending=False).head(120).to_dict('records')
    chart = {}
    for h in HOLD_DAYS:
        if h in stats and stats[h]:
            chart[f'dist{h}'] = stats[h].get('dist')
            chart[f'stats{h}'] = {k: v for k, v in stats[h].items() if k != 'dist'}
    tmpl = Template(TMPL)
    html = tmpl.render(stats=stats, recent=recent,
                       chart_json=json.dumps(chart, ensure_ascii=False),
                       gen_time=datetime.now().strftime('%Y-%m-%d %H:%M'))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',  type=int, default=None, help='仅前N只股票')
    parser.add_argument('--workers', type=int, default=3,    help='线程数')
    args = parser.parse_args()

    result = run_backtest(max_stocks=args.sample, max_workers=args.workers)
    if result is None:
        print("回测无结果"); sys.exit(0)

    df, stats = result
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results')
    csv_p, json_p, html_p = save_and_report(df, stats, out_dir)

    print(f"\n{'='*60}")
    print(f"  回测完成 | 信号:{stats['total_signals']}  股票:{stats['total_stocks']}")
    for h in HOLD_DAYS:
        if h in stats and stats[h]:
            s = stats[h]
            print(f"  {h:2d}日: 胜率={s['win_rate']}%  均收益={s['avg_ret']}%  超额={s['avg_alpha']}%  Sharpe={s['sharpe']}")
    print(f"\n  CSV   : {csv_p}")
    print(f"  JSON  : {json_p}")
    print(f"  HTML  : {html_p}")
    print('='*60)
