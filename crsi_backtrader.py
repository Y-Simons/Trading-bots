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
        # Regime filter params
        regime_filter=True,
        regime_ma_len=200,
        exit_on_regime_flip=True,
        # ATR/trailing
        use_atr_exits=False,
        atr_period=14,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        trailing_stop=False,
        rearm_exits=False,
        # Daily risk cap
        daily_loss_cap=0.0,          # absolute currency (0 disables)
        daily_loss_cap_pct=0.0,      # percentage of day-start equity (0 disables)
        # Equity capture
        collect_equity=False,
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

        # Regime indicator (SMA)
        self.regime_ma = None
        if self.p.regime_filter:
            self.regime_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=int(self.p.regime_ma_len))

        # ATR for ATR-based exits / trailing
        self.atr = None
        if self.p.use_atr_exits or self.p.trailing_stop:
            self.atr = bt.indicators.ATR(self.data, period=int(self.p.atr_period))

        self.entry_order = None
        self.stop_order = None
        self.limit_order = None
        self.entry_bar_index = None

        # Track dynamic trailing levels
        self.current_stop = None
        self.current_limit = None

        # Daily risk tracking
        self.day_start_value = None
        self.current_day = None
        self.locked_until_next_day = False

        # Equity capture
        self.equity_curve = [] if self.p.collect_equity else None

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

    def _compute_exit_levels(self, close_price: float, is_long: bool) -> (float, float):
        if self.p.use_atr_exits and self.atr is not None and not np.isnan(self.atr[0]):
            atr_val = float(self.atr[0])
            if is_long:
                sl = close_price - self.p.atr_sl_mult * atr_val
                tp = close_price + self.p.atr_tp_mult * atr_val
            else:
                sl = close_price + self.p.atr_sl_mult * atr_val
                tp = close_price - self.p.atr_tp_mult * atr_val
            return sl, tp
        else:
            if is_long:
                return close_price * (1.0 - self.p.sl_pct / 100.0), close_price * (1.0 + self.p.tp_pct / 100.0)
            else:
                return close_price * (1.0 + self.p.sl_pct / 100.0), close_price * (1.0 - self.p.tp_pct / 100.0)

    def _apply_trailing(self, close_price: float, is_long: bool, sl: float, tp: float) -> (float, float):
        if not self.p.trailing_stop:
            return sl, tp
        if self.current_stop is None:
            self.current_stop = sl
        if is_long:
            # trail upward only
            self.current_stop = max(self.current_stop, close_price - (abs(close_price - sl)))
        else:
            # trail downward only
            self.current_stop = min(self.current_stop, close_price + (abs(close_price - sl)))
        return self.current_stop, tp

    def _rearm_exit_orders(self, new_sl: float, new_tp: float):
        changed = False
        eps = 1e-8
        if self.current_stop is None or abs(self.current_stop - new_sl) > eps:
            changed = True
        if self.current_limit is None or abs(self.current_limit - new_tp) > eps:
            changed = True
        if not changed:
            return
        self.cancel_children()
        if self.position.size > 0:
            self.stop_order = self.sell(exectype=bt.Order.Stop, price=new_sl)
            self.limit_order = self.sell(exectype=bt.Order.Limit, price=new_tp)
        elif self.position.size < 0:
            self.stop_order = self.buy(exectype=bt.Order.Stop, price=new_sl)
            self.limit_order = self.buy(exectype=bt.Order.Limit, price=new_tp)
        self.current_stop = new_sl
        self.current_limit = new_tp

    def next(self):
        # Equity capture
        if self.p.collect_equity and self.equity_curve is not None:
            dt = bt.num2date(self.datas[0].datetime[0])
            self.equity_curve.append((dt, float(self.broker.getvalue())))

        close_price = float(self.data.close[0])
        if np.isnan(self.ind.db[0]) or np.isnan(self.ind.ub[0]):
            return

        # Daily risk cap handling
        dt = bt.num2date(self.datas[0].datetime[0])
        day_key = (dt.year, dt.month, dt.day)
        if self.current_day != day_key:
            self.current_day = day_key
            self.day_start_value = float(self.broker.getvalue())
            self.locked_until_next_day = False
        if not self.locked_until_next_day and self.day_start_value is not None:
            drop_abs = self.day_start_value - float(self.broker.getvalue())
            drop_pct = drop_abs / self.day_start_value * 100.0 if self.day_start_value > 0 else 0.0
            if (self.p.daily_loss_cap and drop_abs >= self.p.daily_loss_cap) or \
               (self.p.daily_loss_cap_pct and drop_pct >= self.p.daily_loss_cap_pct):
                self.cancel_children()
                if self.position:
                    self.close()
                self.locked_until_next_day = True
                return
        if self.locked_until_next_day:
            return

        # Compute regime flags
        if self.p.regime_filter and self.regime_ma is not None:
            ma_val = float(self.regime_ma[0]) if not np.isnan(self.regime_ma[0]) else np.nan
            if np.isnan(ma_val):
                in_bull = False
                in_bear = False
            else:
                in_bull = close_price >= ma_val
                in_bear = close_price <= ma_val
        else:
            in_bull = True
            in_bear = True

        # Time-based exit
        if self.position and self.p.max_bars_in_trade and self.entry_bar_index is not None:
            bars_in_trade = len(self) - int(self.entry_bar_index)
            if bars_in_trade >= int(self.p.max_bars_in_trade):
                self.cancel_children()
                self.close()
                return

        # Regime flip exit
        if self.position and self.p.regime_filter and self.p.exit_on_regime_flip:
            if self.position.size > 0 and not in_bull:
                self.cancel_children()
                self.close()
                return
            if self.position.size < 0 and self.p.enable_short and not in_bear:
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

            # Per-bar re-arming / trailing for active position
            is_long = self.position.size > 0
            sl, tp = self._compute_exit_levels(close_price, is_long)
            sl, tp = self._apply_trailing(close_price, is_long, sl, tp)
            if self.p.rearm_exits or self.p.trailing_stop or self.p.use_atr_exits:
                self._rearm_exit_orders(sl, tp)
            return

        # If any order alive, wait
        if any(o for o in [self.entry_order, self.stop_order, self.limit_order] if o and o.status in [o.Submitted, o.Accepted]):
            return

        # Compute exits for potential new position
        sl_long, tp_long = self._compute_exit_levels(close_price, True)
        sl_short, tp_short = self._compute_exit_levels(close_price, False)

        if not self.position:
            if self.cross_db[0] > 0 and in_bull:
                if self.p.rearm_exits or self.p.trailing_stop or self.p.use_atr_exits:
                    # Market/close entry then attach exits we manage
                    self.entry_order = self.buy()
                    self.current_stop, self.current_limit = None, None
                    self._rearm_exit_orders(sl_long, tp_long)
                else:
                    mainside, stopside, limitside = self.buy_bracket(
                        price=None, stopprice=sl_long, limitprice=tp_long
                    )
                    self.entry_order, self.stop_order, self.limit_order = mainside, stopside, limitside
                return

            if self.p.enable_short and self.cross_ub[0] < 0 and in_bear:
                if self.p.rearm_exits or self.p.trailing_stop or self.p.use_atr_exits:
                    self.entry_order = self.sell()
                    self.current_stop, self.current_limit = None, None
                    self._rearm_exit_orders(sl_short, tp_short)
                else:
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
    # Regime filter params
    regime_filter: bool = True,
    regime_ma_len: int = 200,
    exit_on_regime_flip: bool = True,
    # ATR/trailing
    use_atr_exits: bool = False,
    atr_period: int = 14,
    atr_sl_mult: float = 2.0,
    atr_tp_mult: float = 3.0,
    trailing_stop: bool = False,
    rearm_exits: bool = False,
    # Daily risk cap
    daily_loss_cap: float = 0.0,
    daily_loss_cap_pct: float = 0.0,
    # Export
    collect_equity: bool = False,
    export_equity_csv: str = None,
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
        # Regime
        regime_filter=regime_filter,
        regime_ma_len=regime_ma_len,
        exit_on_regime_flip=exit_on_regime_flip,
        # ATR/trailing
        use_atr_exits=use_atr_exits,
        atr_period=atr_period,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        trailing_stop=trailing_stop,
        rearm_exits=rearm_exits,
        # Daily
        daily_loss_cap=daily_loss_cap,
        daily_loss_cap_pct=daily_loss_cap_pct,
        # Equity
        collect_equity=collect_equity,
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

    # Export equity if requested
    if collect_equity and export_equity_csv and getattr(strat, 'equity_curve', None):
        eq_df = pd.DataFrame(strat.equity_curve, columns=['datetime', 'equity'])
        try:
            eq_df.to_csv(export_equity_csv, index=False)
        except Exception:
            pass

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
                    max_loss_per_trade=max_loss_per_trade, interval=interval, symbol=symbol,
                    regime_filter=regime_filter, regime_ma_len=regime_ma_len, exit_on_regime_flip=exit_on_regime_flip,
                    use_atr_exits=use_atr_exits, atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                    trailing_stop=trailing_stop, rearm_exits=rearm_exits,
                    daily_loss_cap=daily_loss_cap, daily_loss_cap_pct=daily_loss_cap_pct),
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
        # Regime
        regime_filter = trial.suggest_categorical("regime_filter", [True, False])
        regime_ma_len = trial.suggest_int("regime_ma_len", 50, 300)
        exit_on_regime_flip = trial.suggest_categorical("exit_on_regime_flip", [True, False])
        # ATR/trailing
        use_atr_exits = trial.suggest_categorical("use_atr_exits", [False, True])
        atr_period = trial.suggest_int("atr_period", 7, 40)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 5.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.0, 10.0)
        trailing_stop = trial.suggest_categorical("trailing_stop", [False, True])
        rearm_exits = trial.suggest_categorical("rearm_exits", [False, True])

        try:
            res = run_backtest(
                symbol=symbol, start=start, end=end, interval=interval, cash=cash,
                domcycle=domcycle, vibration=vibration, leveling=leveling,
                sl_pct=sl_pct, tp_pct=tp_pct, enable_short=enable_short,
                max_bars_in_trade=max_bars_in_trade,
                max_leverage=max_leverage, max_loss_per_trade=max_loss_per_trade,
                regime_filter=regime_filter, regime_ma_len=regime_ma_len, exit_on_regime_flip=exit_on_regime_flip,
                use_atr_exits=use_atr_exits, atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                trailing_stop=trailing_stop, rearm_exits=rearm_exits,
            )
        except Exception:
            return -1e12

        # Multi-objective score
        max_dd = res.get('MaxDrawdownPct', None) or 0.0
        sharpe = res.get('Sharpe', None) or 0.0
        trades = res.get('TotalTrades', 0)
        score = float(sharpe) - 0.05 * float(max_dd)
        if trades < 10:
            score -= 1.0
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
        # Regime
        regime_filter = trial.suggest_categorical("regime_filter", [True, False])
        regime_ma_len = trial.suggest_int("regime_ma_len", 50, 300)
        exit_on_regime_flip = trial.suggest_categorical("exit_on_regime_flip", [True, False])
        # ATR/trailing
        use_atr_exits = trial.suggest_categorical("use_atr_exits", [False, True])
        atr_period = trial.suggest_int("atr_period", 7, 40)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 5.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.0, 10.0)
        trailing_stop = trial.suggest_categorical("trailing_stop", [False, True])
        rearm_exits = trial.suggest_categorical("rearm_exits", [False, True])

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
                regime_filter=regime_filter, regime_ma_len=regime_ma_len, exit_on_regime_flip=exit_on_regime_flip,
                use_atr_exits=use_atr_exits, atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                trailing_stop=trailing_stop, rearm_exits=rearm_exits,
            )
        except Exception:
            return -1e12

        max_dd = res.get('MaxDrawdownPct', None) or 0.0
        sharpe = res.get('Sharpe', None) or 0.0
        trades = res.get('TotalTrades', 0)
        score = float(sharpe) - 0.05 * float(max_dd)
        if trades < 10:
            score -= 1.0
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


# Walk-forward validation

def walk_forward(
    symbol: str,
    interval: str,
    start: str,
    end: str = None,
    window_years: int = 2,
    step_months: int = 6,
    trials: int = 30,
) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("optuna is not installed. Please `pip install optuna`.")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today()

    from dateutil.relativedelta import relativedelta

    test_results = []
    cur_train_start = start_dt
    while True:
        train_end = cur_train_start + relativedelta(years=window_years)
        test_end = train_end + relativedelta(months=step_months)
        if train_end >= end_dt or cur_train_start >= end_dt:
            break
        test_end = min(test_end, end_dt)

        study = optimize_with_optuna(
            symbol=symbol,
            start=cur_train_start.strftime('%Y-%m-%d'),
            end=train_end.strftime('%Y-%m-%d'),
            interval=interval,
            n_trials=trials,
        )
        params = study.best_params
        res_test = run_backtest(
            symbol=symbol,
            start=train_end.strftime('%Y-%m-%d'),
            end=test_end.strftime('%Y-%m-%d'),
            interval=interval,
            domcycle=params.get('domcycle', 20),
            vibration=params.get('vibration', 10),
            leveling=params.get('leveling', 10.0),
            sl_pct=params.get('sl_pct', 2.0),
            tp_pct=params.get('tp_pct', 4.0),
            max_bars_in_trade=params.get('max_bars_in_trade', 0),
            regime_filter=params.get('regime_filter', True),
            regime_ma_len=params.get('regime_ma_len', 200),
            exit_on_regime_flip=params.get('exit_on_regime_flip', True),
            use_atr_exits=params.get('use_atr_exits', False),
            atr_period=params.get('atr_period', 14),
            atr_sl_mult=params.get('atr_sl_mult', 2.0),
            atr_tp_mult=params.get('atr_tp_mult', 3.0),
            trailing_stop=params.get('trailing_stop', False),
            rearm_exits=params.get('rearm_exits', False),
        )
        res_test['TrainStart'] = cur_train_start.strftime('%Y-%m-%d')
        res_test['TrainEnd'] = train_end.strftime('%Y-%m-%d')
        res_test['TestEnd'] = test_end.strftime('%Y-%m-%d')
        test_results.append(res_test)

        cur_train_start = cur_train_start + relativedelta(months=step_months)

    # Aggregate
    if not test_results:
        return {'WF': 'No windows', 'Windows': 0}
    df = pd.DataFrame(test_results)
    return {
        'WF': 'OK',
        'Windows': len(test_results),
        'AvgNetProfit': float(df['NetProfit'].mean()),
        'AvgSharpe': float(pd.to_numeric(df['Sharpe'], errors='coerce').mean()),
        'AvgMaxDD': float(pd.to_numeric(df['MaxDrawdownPct'], errors='coerce').mean()),
        'TotalTrades': int(df['TotalTrades'].sum()),
        'Details': test_results,
    }

# Sensitivity analysis around given parameters

def sensitivity_scan(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    center: Dict[str, Any],
    span: Dict[str, Any] = None,
) -> pd.DataFrame:
    span = span or {}
    domcycle_vals = sorted(set([center.get('domcycle', 20) + d for d in [-4, -2, 0, 2, 4]]))
    vibration_vals = sorted(set([center.get('vibration', 10) + d for d in [-3, -1, 0, 1, 3]]))
    leveling_vals = sorted(set([center.get('leveling', 10.0) + d for d in [-5.0, -2.0, 0.0, 2.0, 5.0]]))
    sl_vals = sorted(set([center.get('sl_pct', 2.0) + d for d in [-0.5, 0.0, 0.5]]))
    tp_vals = sorted(set([center.get('tp_pct', 4.0) + d for d in [-1.0, 0.0, 1.0]]))

    rows = []
    for dc in domcycle_vals:
        for vib in vibration_vals:
            for lev in leveling_vals:
                for slp in sl_vals:
                    for tpp in tp_vals:
                        res = run_backtest(
                            symbol=symbol, start=start, end=end, interval=interval,
                            domcycle=int(max(10, min(60, dc))),
                            vibration=int(max(5, min(20, vib))),
                            leveling=float(max(1.0, min(50.0, lev))),
                            sl_pct=max(0.1, slp), tp_pct=max(0.1, tpp)
                        )
                        res_row = dict(dc=dc, vib=vib, lev=lev, sl=slp, tp=tpp,
                                        Sharpe=res['Sharpe'], Net=res['NetProfit'], DD=res['MaxDrawdownPct'], Trades=res['TotalTrades'])
                        rows.append(res_row)
    return pd.DataFrame(rows)


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
    # Regime filter CLI
    parser.add_argument('--regime', action='store_true', help='Enable SMA regime filter (price above MA for longs, below for shorts)')
    parser.add_argument('--regimema', type=int, default=200, help='Regime SMA length')
    parser.add_argument('--regime-exit', action='store_true', help='Exit when regime flips against the position')

    parser.add_argument('--maxleverage', type=float, default=100.0)
    parser.add_argument('--maxloss', type=float, default=500.0)

    # ATR/trailing/rearm
    parser.add_argument('--atr', action='store_true', help='Use ATR-based exits instead of percentage')
    parser.add_argument('--atrperiod', type=int, default=14)
    parser.add_argument('--atrsl', type=float, default=2.0)
    parser.add_argument('--atrtp', type=float, default=3.0)
    parser.add_argument('--trail', action='store_true', help='Enable trailing stop')
    parser.add_argument('--rearm', action='store_true', help='Re-arm exits each bar')

    # Daily risk cap
    parser.add_argument('--dailycap', type=float, default=0.0, help='Absolute daily loss cap (0 disables)')
    parser.add_argument('--dailycap-pct', type=float, default=0.0, help='Daily loss cap as percent of day-start equity (0 disables)')

    # Optimization controls
    parser.add_argument('--optimize', action='store_true', help='Run Optuna optimization (single symbol/interval)')
    parser.add_argument('--trials', type=int, default=30)

    parser.add_argument('--optimize-global', action='store_true', help='Search symbols and intervals with Optuna')
    parser.add_argument('--intervals', type=str, default='1d,1h,4h', help='Comma-separated intervals for global optimize')

    # Walk-forward
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation (single symbol/interval)')
    parser.add_argument('--wf-window-years', type=int, default=2)
    parser.add_argument('--wf-step-months', type=int, default=6)

    # Sensitivity scan
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity scan around provided parameters (single symbol/interval)')

    # Export
    parser.add_argument('--export-equity', type=str, default=None, help='Path to save equity curve CSV for single-symbol runs')

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

    if args.walk_forward:
        if len(symbols) != 1:
            print("Walk-forward requires exactly one symbol.", file=sys.stderr)
            return 2
        wf = walk_forward(
            symbol=symbols[0], interval=args.interval, start=args.start, end=args.end,
            window_years=args.wf_window_years, step_months=args.wf_step_months, trials=args.trials
        )
        print(wf)
        return 0

    if args.sensitivity:
        if len(symbols) != 1:
            print("Sensitivity scan requires exactly one symbol.", file=sys.stderr)
            return 2
        center = dict(domcycle=args.domcycle, vibration=args.vibration, leveling=args.leveling,
                      sl_pct=args.sl, tp_pct=args.tp)
        df = sensitivity_scan(symbol=symbols[0], interval=args.interval, start=args.start, end=args.end, center=center)
        print(df.head(30).to_string(index=False))
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
                regime_filter=args.regime,
                regime_ma_len=args.regimema,
                exit_on_regime_flip=args.regime_exit,
                use_atr_exits=args.atr,
                atr_period=args.atrperiod,
                atr_sl_mult=args.atrsl,
                atr_tp_mult=args.atrtp,
                trailing_stop=args.trail,
                rearm_exits=args.rearm,
                daily_loss_cap=args.dailycap,
                daily_loss_cap_pct=args.dailycap_pct,
                collect_equity=bool(args.export_equity),
                export_equity_csv=args.export_equity,
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