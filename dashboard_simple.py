#!/usr/bin/env python3

import json
import pandas as pd
import streamlit as st
from datetime import date

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from crsi_backtrader import optimize_with_optuna, run_backtest

st.set_page_config(page_title="cRSI Optimizer (Minimal)", layout="centered")
st.title("cRSI Optimizer (Minimal)")

# Fixed defaults (kept simple)
DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date.today()
DEFAULT_INTERVAL = "1d"
DEFAULT_TRIALS = 40

symbol = st.text_input("Ticker (Yahoo Finance)", value="QQQ")
run_btn = st.button("Find Best Parameters")

if run_btn:
    if not symbol.strip():
        st.warning("Enter a ticker symbol.")
    else:
        with st.spinner("Optimizing... This may take a few minutes."):
            study = optimize_with_optuna(
                symbol=symbol.strip(),
                start=DEFAULT_START.strftime('%Y-%m-%d'),
                end=DEFAULT_END.strftime('%Y-%m-%d'),
                interval=DEFAULT_INTERVAL,
                n_trials=DEFAULT_TRIALS,
            )
            best = study.best_params

            st.subheader("Best Parameters")
            st.json(best)

            # Run a final backtest with the best params
            st.subheader("Backtest with Best Parameters")
            res = run_backtest(
                symbol=symbol.strip(),
                start=DEFAULT_START.strftime('%Y-%m-%d'),
                end=DEFAULT_END.strftime('%Y-%m-%d'),
                interval=DEFAULT_INTERVAL,
                domcycle=best.get('domcycle', 20),
                vibration=best.get('vibration', 10),
                leveling=best.get('leveling', 10.0),
                sl_pct=best.get('sl_pct', 2.0),
                tp_pct=best.get('tp_pct', 4.0),
                max_bars_in_trade=best.get('max_bars_in_trade', 0),
                regime_filter=best.get('regime_filter', True),
                regime_ma_len=best.get('regime_ma_len', 200),
                exit_on_regime_flip=best.get('exit_on_regime_flip', True),
                use_atr_exits=best.get('use_atr_exits', False),
                atr_period=best.get('atr_period', 14),
                atr_sl_mult=best.get('atr_sl_mult', 2.0),
                atr_tp_mult=best.get('atr_tp_mult', 3.0),
                trailing_stop=best.get('trailing_stop', False),
                rearm_exits=best.get('rearm_exits', False),
                collect_equity=True,
            )

            metrics = {
                'FinalValue': res['FinalValue'],
                'NetProfit': res['NetProfit'],
                'MaxDrawdownPct': res['MaxDrawdownPct'],
                'Sharpe': res['Sharpe'],
                'WinRatePct': res['WinRatePct'],
                'TotalTrades': res['TotalTrades'],
            }
            st.table(pd.DataFrame([metrics]))

            if res.get('EquityCurve'):
                eq = pd.DataFrame(res['EquityCurve'], columns=['datetime','equity']).set_index('datetime')
                st.line_chart(eq)

st.caption("Defaults: interval=1d, start=2018-01-01, trials=40. Only ticker is configurable.")