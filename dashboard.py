#!/usr/bin/env python3

import json
import pandas as pd
import streamlit as st
from datetime import date

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from crsi_backtrader import run_backtest, optimize_with_optuna, walk_forward

st.set_page_config(page_title="cRSI Strategy Dashboard", layout="wide")
st.title("cRSI Strategy Dashboard")

with st.sidebar:
    st.header("Data & Params")
    symbols = st.text_input("Symbols (comma)", value="^GSPC,SPY,QQQ")
    start = st.date_input("Start", value=date(2018,1,1))
    end = st.date_input("End", value=date.today())
    interval = st.selectbox("Interval", options=["1d","1h","4h"], index=0)

    st.subheader("Strategy")
    domcycle = st.slider("domcycle", 10, 60, 20, step=2)
    vibration = st.slider("vibration", 5, 20, 10)
    leveling = st.slider("leveling", 1.0, 50.0, 10.0, step=1.0)
    sl = st.slider("SL %", 0.1, 10.0, 2.0, 0.1)
    tp = st.slider("TP %", 0.1, 15.0, 4.0, 0.1)
    bars = st.slider("Max bars in trade (0=off)", 0, 200, 0)
    enable_short = st.checkbox("Enable Short", value=False)

    st.subheader("Regime Filter")
    regime = st.checkbox("Use Regime Filter", value=True)
    regimema = st.slider("Regime MA", 20, 400, 200)
    regime_exit = st.checkbox("Exit on Regime Flip", value=True)

    st.subheader("ATR/Trailing")
    use_atr = st.checkbox("Use ATR exits", value=False)
    atr_period = st.slider("ATR Period", 7, 40, 14)
    atr_sl = st.slider("ATR SL Mult", 1.0, 5.0, 2.0, 0.1)
    atr_tp = st.slider("ATR TP Mult", 1.0, 10.0, 3.0, 0.1)
    trailing = st.checkbox("Trailing Stop", value=False)
    rearm = st.checkbox("Re-arm exits per bar", value=False)

    st.subheader("Risk & Costs")
    cash = st.number_input("Cash", min_value=1000.0, value=50000.0, step=1000.0)
    max_leverage = st.slider("Max Leverage", 1, 100, 50)
    risk_trade = st.number_input("Max Loss per Trade ($)", min_value=0.0, value=500.0, step=50.0)
    commission = st.number_input("Commission", min_value=0.0, value=0.0005, step=0.0001, format="%f")
    slippage = st.number_input("Slippage %", min_value=0.0, value=0.0, step=0.01)
    daily_cap = st.number_input("Daily Loss Cap ($)", min_value=0.0, value=0.0, step=50.0)
    daily_cap_pct = st.number_input("Daily Loss Cap (%)", min_value=0.0, value=0.0, step=0.5)

    st.subheader("Actions")
    run_btn = st.button("Run Backtests")
    opt_btn = st.button("Optimize (single symbol)")
    wf_btn = st.button("Walk-Forward (single symbol)")

col1, col2 = st.columns([1,1])

if run_btn:
    rows = []
    for sym in [s.strip() for s in symbols.split(',') if s.strip()]:
        try:
            res = run_backtest(
                symbol=sym,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval=interval,
                cash=cash,
                commission=commission,
                slippage_perc=slippage/100.0,
                domcycle=domcycle,
                vibration=vibration,
                leveling=leveling,
                enable_short=enable_short,
                sl_pct=sl,
                tp_pct=tp,
                max_bars_in_trade=bars,
                max_leverage=max_leverage,
                max_loss_per_trade=risk_trade,
                regime_filter=regime,
                regime_ma_len=regimema,
                exit_on_regime_flip=regime_exit,
                use_atr_exits=use_atr,
                atr_period=atr_period,
                atr_sl_mult=atr_sl,
                atr_tp_mult=atr_tp,
                trailing_stop=trailing,
                rearm_exits=rearm,
                collect_equity=True,
            )
            rows.append({**{'Symbol': sym}, **res})
        except Exception as e:
            st.error(f"{sym}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        with col1:
            st.subheader("Metrics")
            show_cols = ['Symbol','FinalValue','NetProfit','MaxDrawdownPct','Sharpe','WinRatePct','TotalTrades']
            st.dataframe(df[show_cols])
        with col2:
            st.subheader("Equity Curves")
            for r in rows:
                if r.get('EquityCurve'):
                    eq = pd.DataFrame(r['EquityCurve'], columns=['datetime','equity']).set_index('datetime')
                    st.line_chart(eq, height=250)

if opt_btn:
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    if len(syms) != 1:
        st.warning("Optimization runs on a single symbol. Enter exactly one symbol.")
    else:
        with st.spinner("Running optimization (few trials for demo)..."):
            study = optimize_with_optuna(symbol=syms[0], start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval=interval, n_trials=20)
        st.json(study.best_params)

if wf_btn:
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    if len(syms) != 1:
        st.warning("Walk-forward runs on a single symbol.")
    else:
        with st.spinner("Running walk-forward (may take time)..."):
            wf = walk_forward(symbol=syms[0], interval=interval, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), window_years=2, step_months=6, trials=15)
        st.write(wf)