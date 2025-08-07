#!/usr/bin/env python3
import argparse
import json
from crsi_backtrader import optimize_with_optuna, run_backtest

def main():
    ap = argparse.ArgumentParser(description='Optimize cRSI strategy for a ticker (minimal).')
    ap.add_argument('symbol', help='Ticker symbol (Yahoo Finance)')
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--interval', default='1d')
    ap.add_argument('--trials', type=int, default=40)
    args = ap.parse_args()

    study = optimize_with_optuna(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval, n_trials=args.trials)
    best = study.best_params
    print('Best parameters:')
    print(json.dumps(best, indent=2))

    res = run_backtest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
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
    )

    summary = {k: res[k] for k in ['FinalValue','NetProfit','MaxDrawdownPct','Sharpe','WinRatePct','TotalTrades']}
    print('Backtest summary:')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()