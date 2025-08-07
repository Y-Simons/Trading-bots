#!/usr/bin/env python3

# pip install backtrader yfinance optuna pandas numpy

import argparse
import math
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt

try:
    import optuna  # optional; only required for --optimize
except Exception:
    optuna = None


class RiskSizer(bt.Sizer):
    params = dict(
        max_leverage=100.0,   # maximum notional = broker value * max_leverage
        max_loss_per_trade=500.0,
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Use strategy parameters for stop distance
        close_price = float(data.close[0])
        sl_pct = float(getattr(self.strategy.p, 'sl_pct', 2.0))
        if sl_pct <= 0 or close_price <= 0:
            return 0

        per_share_risk = close_price * sl_pct / 100.0
        if per_share_risk <= 0:
            return 0

        # Risk-based shares
        shares_by_risk = int(self.p.max_loss_per_trade // per_share_risk)

        # Leverage cap in notional terms
        try:
            account_value = float(self.strategy.broker.getvalue())
        except Exception:
            account_value = cash
        max_notional = account_value * float(self.p.max_leverage)
        shares_by_leverage = int(max_notional // close_price)

        shares = max(0, min(shares_by_risk, shares_by_leverage))
        if shares <= 0:
            return 0
        return shares if isbuy else -shares


class cRSIIndicator(bt.Indicator):
    lines = ('crsi', 'db', 'ub')
    params = dict(domcycle=20, vibration=10, leveling=10.0)

    def __init__(self):
        domcycle = int(self.p.domcycle)
        if domcycle % 2 == 1:
            domcycle += 1
        self.cyclelen = max(1, domcycle // 2)
        self.cyclicmemory = max(1, domcycle * 2)
        self.torque = 2.0 / (self.p.vibration + 1.0)
        self.phasingLag = int(math.floor((self.p.vibration - 1) / 2))

        self.rsi = bt.indicators.RSI(self.data, period=self.cyclelen, safediv=True)

        self.addminperiod = max(self.cyclicmemory + self.phasingLag + 2, self.cyclelen + 2)

    def next(self):
        rsi_now = float(self.rsi[0])
        rsi_lag = float(self.rsi[-self.phasingLag]) if len(self) > self.phasingLag else rsi_now

        prev_crsi = float(self.lines.crsi[-1]) if len(self) > 0 and not np.isnan(self.lines.crsi[-1]) else rsi_now
        crsi_now = self.torque * (2.0 * rsi_now - rsi_lag) + (1.0 - self.torque) * prev_crsi
        self.lines.crsi[0] = crsi_now

        if len(self) < self.cyclicmemory:
            self.lines.db[0] = np.nan
            self.lines.ub[0] = np.nan
            return

        window = [float(self.lines.crsi[-i]) for i in range(self.cyclicmemory)]
        lmax = max(window)
        lmin = min(window)
        mstep = (lmax - lmin) / 100.0
        aperc = self.p.leveling / 100.0

        db_val = lmin
        if mstep > 0:
            for steps in range(101):
                testvalue = lmin + mstep * steps
                below = sum(1 for val in window if val < testvalue)
                ratio = below / float(self.cyclicmemory)
                if ratio >= aperc:
                    db_val = testvalue
                    break

        ub_val = lmax
        if mstep > 0:
            for steps in range(101):
                testvalue = lmax - mstep * steps
                above = sum(1 for val in window if val >= testvalue)
                ratio = above / float(self.cyclicmemory)
                if ratio >= aperc:
                    ub_val = testvalue
                    break

        self.lines.db[0] = db_val
        self.lines.ub[0] = ub_val


class CRSIStrategy(bt.Strategy):
    params = dict(
        domcycle=20,
        vibration=10,
        leveling=10.0,
        enable_short=False,
        sl_pct=2.0,
        tp_pct=4.0,
        commission=0.0005,
        max_bars_in_trade=0,  # 0 disables time-based exit
    )

    def __init__(self):
        self.ind = cRSIIndicator(
            self.data,
            domcycle=self.p.domcycle,
            vibration=self.p.vibration,
            leveling=self.p.leveling,
        )

        self.cross_db = bt.indicators.CrossOver(self.ind.crsi, self.ind.db)
        self.cross_ub = bt.indicators.CrossOver(self.ind.crsi, self.ind.ub)

        self.entry_order = None
        self.stop_order = None
        self.limit_order = None
        self.entry_bar_index = None

    def cancel_children(self):
        if self.stop_order:
            try:
                self.cancel(self.stop_order)
            except Exception:
                pass
            self.stop_order = None
        if self.limit_order:
            try:
                self.cancel(self.limit_order)
            except Exception:
                pass
            self.limit_order = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected, order.Margin]:
            if order is self.entry_order and order.status == order.Completed:
                self.entry_bar_index = len(self)
            if order is self.entry_order and order.status != order.Submitted:
                self.entry_order = None
            if order is self.stop_order and order.status != order.Submitted:
                self.stop_order = None
            if order is self.limit_order and order.status != order.Submitted:
                self.limit_order = None

    def next(self):
        close_price = float(self.data.close[0])
        if np.isnan(self.ind.db[0]) or np.isnan(self.ind.ub[0]):
            return

        # Time-based exit
        if self.position and self.p.max_bars_in_trade and self.entry_bar_index is not None:
            bars_in_trade = len(self) - int(self.entry_bar_index)
            if bars_in_trade >= int(self.p.max_bars_in_trade):
                self.cancel_children()
                self.close()
                return

        # Signal-based exit
        if self.position:
            if self.position.size > 0:
                if self.cross_ub[0] < 0:
                    self.cancel_children()
                    self.close()
                    return
            elif self.position.size < 0 and self.p.enable_short:
                if self.cross_db[0] > 0:
                    self.cancel_children()
                    self.close()
                    return

        # If any order alive, wait
        if any(o for o in [self.entry_order, self.stop_order, self.limit_order] if o and o.status in [o.Submitted, o.Accepted]):
            return

        # Bracket exits from current price
        sl_long = close_price * (1.0 - self.p.sl_pct / 100.0)
        tp_long = close_price * (1.0 + self.p.tp_pct / 100.0)
        sl_short = close_price * (1.0 + self.p.sl_pct / 100.0)
        tp_short = close_price * (1.0 - self.p.tp_pct / 100.0)

        if not self.position:
            if self.cross_db[0] > 0:
                mainside, stopside, limitside = self.buy_bracket(
                    price=None, stopprice=sl_long, limitprice=tp_long
                )
                self.entry_order, self.stop_order, self.limit_order = mainside, stopside, limitside
                return

            if self.p.enable_short and self.cross_ub[0] < 0:
                mainside, stopside, limitside = self.sell_bracket(
                    price=None, stopprice=sl_short, limitprice=tp_short
                )
                self.entry_order, self.stop_order, self.limit_order = mainside, stopside, limitside
                return


def fetch_data(symbol: str, start: str, end: str = None, interval: str = "1d") -> pd.DataFrame:
    # First try: Ticker().history which returns a clean single-level OHLCV
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
    except Exception:
        df = None

    # Fallback: download()
    if df is None or df.empty:
        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by='column')

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol/date/interval.")

    # Some yfinance versions return 'Date' as a column
    if 'Date' in df.columns:
        df = df.set_index('Date')

    df.index = pd.to_datetime(df.index)

    # Normalize columns to single-level OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))
            if symbol in lvl1:
                df = df.xs(symbol, axis=1, level=1, drop_level=True)
            elif symbol in lvl0:
                df = df.xs(symbol, axis=1, level=0, drop_level=True)
            else:
                unique_lvl1 = sorted(set(lvl1))
                if len(unique_lvl1) == 1:
                    df = df.xs(unique_lvl1[0], axis=1, level=1, drop_level=True)
                else:
                    df.columns = [str(a).lower().replace(' ', '') for a, _ in df.columns]
        else:
            df.columns = ['_'.join([str(x) for x in tup]).lower() for tup in df.columns]
    else:
        df.columns = [str(c).lower().replace(' ', '') for c in df.columns]

    # Map adjclose to close if needed
    if 'close' not in df.columns and 'adjclose' in df.columns:
        df['close'] = df['adjclose']

    # Ensure required columns exist; create volume=0 if missing
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            if col == 'volume':
                df['volume'] = 0
            else:
                raise ValueError(f"Missing required columns for {symbol}: {[c for c in required if c not in df.columns]}")

    df = df[required].astype(float)

    # Rename to Backtrader's expected case-sensitive columns
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    # Ensure OpenInterest exists
    if 'OpenInterest' not in df.columns:
        df['OpenInterest'] = 0.0

    return df


def run_backtest(
    symbol: str,
    start: str = "2015-01-01",
    end: str = None,
    interval: str = "1d",
    cash: float = 50000.0,
    commission: float = 0.0005,
    slippage_perc: float = 0.0,
    domcycle: int = 20,
    vibration: int = 10,
    leveling: float = 10.0,
    enable_short: bool = False,
    sl_pct: float = 2.0,
    tp_pct: float = 4.0,
    max_bars_in_trade: int = 0,
    max_leverage: float = 100.0,
    max_loss_per_trade: float = 500.0,
) -> Dict[str, Any]:
    data_df = fetch_data(symbol, start, end, interval)
    data = bt.feeds.PandasData(dataname=data_df,
                                open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest='OpenInterest')

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    # Commission and margin to allow leverage
    if max_leverage and max_leverage > 0:
        margin = 1.0 / float(max_leverage)
        cerebro.broker.setcommission(commission=commission, margin=margin)
    else:
        cerebro.broker.setcommission(commission=commission)
    if slippage_perc > 0:
        cerebro.broker.set_slippage_perc(perc=slippage_perc)

    # Risk-based sizer
    cerebro.addsizer(RiskSizer, max_leverage=max_leverage, max_loss_per_trade=max_loss_per_trade)

    cerebro.addstrategy(
        CRSIStrategy,
        domcycle=domcycle,
        vibration=vibration,
        leveling=leveling,
        enable_short=enable_short,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        commission=commission,
        max_bars_in_trade=max_bars_in_trade,
    )

    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, factor=252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    results = cerebro.run()
    strat = results[0]

    sharpe_an = strat.analyzers.sharpe.get_analysis() if hasattr(strat.analyzers, 'sharpe') else {}
    sharpe = sharpe_an.get('sharperatio', None)

    dd_an = strat.analyzers.dd.get_analysis() if hasattr(strat.analyzers, 'dd') else {}
    max_dd = None
    if isinstance(dd_an, dict):
        max_dd = dd_an.get('max', {}).get('drawdown', None)

    trades_an = strat.analyzers.trades.get_analysis() if hasattr(strat.analyzers, 'trades') else {}
    total_closed = 0
    win_trades = 0
    if isinstance(trades_an, dict):
        total_closed = trades_an.get('total', {}).get('closed', 0) or 0
        win_trades = trades_an.get('won', {}).get('total', 0) or 0

    netprofit = (cerebro.broker.getvalue() - cash)
    win_rate = (win_trades / total_closed) * 100.0 if total_closed > 0 else None

    return dict(
        FinalValue=round(cerebro.broker.getvalue(), 2),
        NetProfit=round(netprofit, 2),
        MaxDrawdownPct=round(max_dd, 2) if max_dd is not None else None,
        Sharpe=sharpe if sharpe is None else round(sharpe, 3),
        WinRatePct=round(win_rate, 2) if win_rate is not None else None,
        TotalTrades=int(total_closed) if total_closed is not None else 0,
        Params=dict(domcycle=domcycle, vibration=vibration, leveling=leveling,
                    sl_pct=sl_pct, tp_pct=tp_pct, enable_short=enable_short,
                    max_bars_in_trade=max_bars_in_trade, max_leverage=max_leverage,
                    max_loss_per_trade=max_loss_per_trade, interval=interval, symbol=symbol),
    )


def optimize_with_optuna(
    symbol: str,
    start: str = "2015-01-01",
    end: str = None,
    interval: str = "1d",
    n_trials: int = 30,
    enable_short: bool = False,
    cash: float = 50000.0,
    max_leverage: float = 100.0,
    max_loss_per_trade: float = 500.0,
):
    if optuna is None:
        raise RuntimeError("optuna is not installed. Please `pip install optuna`.\n")

    def objective(trial: 'optuna.Trial'):
        domcycle = trial.suggest_int("domcycle", 10, 60, step=2)
        vibration = trial.suggest_int("vibration", 5, 20)
        leveling = trial.suggest_float("leveling", 5.0, 30.0)
        sl_pct = trial.suggest_float("sl_pct", 0.2, 5.0)
        tp_pct = trial.suggest_float("tp_pct", 0.5, 15.0)
        max_bars_in_trade = trial.suggest_int("max_bars_in_trade", 0, 120)

        try:
            res = run_backtest(
                symbol=symbol, start=start, end=end, interval=interval, cash=cash,
                domcycle=domcycle, vibration=vibration, leveling=leveling,
                sl_pct=sl_pct, tp_pct=tp_pct, enable_short=enable_short,
                max_bars_in_trade=max_bars_in_trade,
                max_leverage=max_leverage, max_loss_per_trade=max_loss_per_trade
            )
        except Exception:
            return -1e12

        # Prefer Sharpe; fallback to NetProfit
        score = res['Sharpe']
        if score is None or (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
            score = res['NetProfit']
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_global(
    symbols: List[str],
    intervals: List[str],
    start: str = "2018-01-01",
    end: str = None,
    n_trials: int = 50,
    cash: float = 50000.0,
    max_loss_per_trade: float = 500.0,
):
    if optuna is None:
        raise RuntimeError("optuna is not installed. Please `pip install optuna`.\n")

    def objective(trial: 'optuna.Trial'):
        symbol = trial.suggest_categorical("symbol", symbols)
        interval = trial.suggest_categorical("interval", intervals)
        enable_short = trial.suggest_categorical("enable_short", [False, True])
        max_leverage = trial.suggest_int("max_leverage", 1, 100)

        domcycle = trial.suggest_int("domcycle", 10, 60, step=2)
        vibration = trial.suggest_int("vibration", 5, 20)
        leveling = trial.suggest_float("leveling", 5.0, 30.0)
        sl_pct = trial.suggest_float("sl_pct", 0.2, 5.0)
        tp_pct = trial.suggest_float("tp_pct", 0.5, 15.0)
        max_bars_in_trade = trial.suggest_int("max_bars_in_trade", 0, 180)

        # Skip intraday for Yahoo index symbols (caret) to avoid errors
        if interval != '1d' and symbol.startswith('^'):
            return -1e12

        try:
            res = run_backtest(
                symbol=symbol, start=start, end=end, interval=interval, cash=cash,
                domcycle=domcycle, vibration=vibration, leveling=leveling,
                sl_pct=sl_pct, tp_pct=tp_pct, enable_short=enable_short,
                max_bars_in_trade=max_bars_in_trade,
                max_leverage=float(max_leverage), max_loss_per_trade=max_loss_per_trade,
            )
        except Exception:
            return -1e12

        score = res['Sharpe']
        if score is None or (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
            score = res['NetProfit']
        # Penalize too few trades
        if res['TotalTrades'] < 5:
            score = float(score) - 1e3
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cRSI Backtrader strategy on symbols")
    parser.add_argument('--symbols', type=str, default='^GSPC,^DJI,^IXIC,^NDX,SPY,DIA,QQQ', help='Comma-separated symbols')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--interval', type=str, default='1d', help='1d, 1h, 4h (indices generally daily only)')
    parser.add_argument('--cash', type=float, default=50000.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0)

    parser.add_argument('--domcycle', type=int, default=20)
    parser.add_argument('--vibration', type=int, default=10)
    parser.add_argument('--leveling', type=float, default=10.0)
    parser.add_argument('--sl', type=float, default=2.0)
    parser.add_argument('--tp', type=float, default=4.0)
    parser.add_argument('--bars', type=int, default=0, help='Max bars in trade (0 disables)')
    parser.add_argument('--short', action='store_true', help='Enable short entries')

    parser.add_argument('--maxleverage', type=float, default=100.0)
    parser.add_argument('--maxloss', type=float, default=500.0)

    parser.add_argument('--optimize', action='store_true', help='Run Optuna optimization (single symbol/interval)')
    parser.add_argument('--trials', type=int, default=30)

    parser.add_argument('--optimize-global', action='store_true', help='Search symbols and intervals with Optuna')
    parser.add_argument('--intervals', type=str, default='1d,1h,4h', help='Comma-separated intervals for global optimize')

    parser.add_argument('--csv', type=str, default=None, help='Optional path to save results CSV')

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    results_rows = []

    if args.optimize_global:
        if optuna is None:
            print("optuna is not installed. Run: pip install optuna", file=sys.stderr)
            return 2
        intervals = [i.strip() for i in args.intervals.split(',') if i.strip()]
        study = optimize_global(
            symbols=symbols,
            intervals=intervals,
            start=args.start,
            end=args.end,
            n_trials=args.trials,
            cash=args.cash,
            max_loss_per_trade=args.maxloss,
        )
        print("Best global params:", study.best_params)
        return 0

    if args.optimize and len(symbols) != 1:
        print("For optimization, please provide exactly one symbol via --symbols SYM", file=sys.stderr)
        return 2

    if args.optimize:
        if optuna is None:
            print("optuna is not installed. Run: pip install optuna", file=sys.stderr)
            return 2
        study = optimize_with_optuna(
            symbol=symbols[0], start=args.start, end=args.end, interval=args.interval,
            n_trials=args.trials, enable_short=args.short, cash=args.cash,
            max_leverage=args.maxleverage, max_loss_per_trade=args.maxloss,
        )
        print("Best params:", study.best_params)
        return 0

    for sym in symbols:
        try:
            res = run_backtest(
                symbol=sym,
                start=args.start,
                end=args.end,
                interval=args.interval,
                cash=args.cash,
                commission=args.commission,
                slippage_perc=args.slippage,
                domcycle=args.domcycle,
                vibration=args.vibration,
                leveling=args.leveling,
                enable_short=args.short,
                sl_pct=args.sl,
                tp_pct=args.tp,
                max_bars_in_trade=args.bars,
                max_leverage=args.maxleverage,
                max_loss_per_trade=args.maxloss,
            )
            row = {'Symbol': sym}
            row.update(res)
            results_rows.append(row)
            print(f"{sym}: {res}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"{sym}: ERROR - {e}\n{tb}", file=sys.stderr)

    if args.csv and results_rows:
        pd.DataFrame(results_rows).to_csv(args.csv, index=False)
        print(f"Saved CSV to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))