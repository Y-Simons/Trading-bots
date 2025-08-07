#!/usr/bin/env python3

# pip install backtrader yfinance optuna pandas numpy

import argparse
import math
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt
import os

try:
    import optuna  # optional; only required for --optimize
except Exception:
    optuna = None

from math import sqrt
from scipy.stats import norm

DEBUG_FETCH = False


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
    # Clip start date for intraday intervals to Yahoo limits
    try:
        start_dt = pd.to_datetime(start) if start else None
        now = pd.Timestamp.utcnow().normalize()
        intraday = interval.endswith('m') or interval.endswith('h')
        if intraday and start_dt is not None:
            lim_days = 365
            if interval == '1m':
                lim_days = 7
            elif interval in {'2m', '5m'}:
                lim_days = 60
            elif interval in {'15m', '30m'}:
                lim_days = 60
            elif interval in {'1h'}:
                lim_days = 730
            clip = now - pd.Timedelta(days=lim_days)
            if start_dt < clip:
                start_dt = clip
                start = start_dt.strftime('%Y-%m-%d')
    except Exception:
        pass

    # Choose period-based retrieval for intraday to satisfy Yahoo constraints
    use_period = interval in {'1m', '2m', '5m', '15m', '30m', '1h'}
    period_map = {
        '1m': '7d',
        '2m': '60d',
        '5m': '60d',
        '15m': '60d',
        '30m': '60d',
        '1h': '2y',
    }
    period = period_map.get(interval, None)

    if DEBUG_FETCH:
        print(f"[fetch_data] symbol={symbol} interval={interval} start={start} end={end} use_period={use_period} period={period}")

    # First try: Ticker().history
    df = None
    for attempt in range(3):
        try:
            if use_period and period:
                df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            else:
                df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                break
        except Exception as e:
            if DEBUG_FETCH:
                print(f"[fetch_data] history attempt {attempt+1} failed: {e}")
        # brief backoff
        try:
            import time; time.sleep(0.5 * (attempt + 1))
        except Exception:
            pass

    # Fallback: download()
    if df is None or df.empty:
        for attempt in range(3):
            try:
                if use_period and period:
                    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, group_by='column', threads=False)
                else:
                    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by='column', threads=False)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                if DEBUG_FETCH:
                    print(f"[fetch_data] download attempt {attempt+1} failed: {e}")
            try:
                import time; time.sleep(0.5 * (attempt + 1))
            except Exception:
                pass

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol/date/interval.")

    # Some yfinance versions return 'Date' as a column
    if 'Date' in df.columns:
        df = df.set_index('Date')

    df.index = pd.to_datetime(df.index)

    if DEBUG_FETCH:
        try:
            print(f"[fetch_data] fetched rows={len(df)} first={df.index[0]} last={df.index[-1]} cols={list(df.columns)[:6]}")
        except Exception:
            pass

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
    slippage_perc: float = 0.0002,
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
    # Execution realism
    cheat_on_close: bool = False,
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

    # Execution realism toggle
    cerebro.broker.set_coc(bool(cheat_on_close))

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

    # Equity and returns
    equity_curve = strat.equity_curve if collect_equity else None
    returns_series = None
    if equity_curve and len(equity_curve) > 1:
        eq_df = pd.DataFrame(equity_curve, columns=['datetime', 'equity']).set_index('datetime')
        returns_series = (eq_df['equity'].pct_change().dropna()).reset_index().values.tolist()

    payload = dict(
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
                    daily_loss_cap=daily_loss_cap, daily_loss_cap_pct=daily_loss_cap_pct,
                    cheat_on_close=cheat_on_close, slippage_perc=slippage_perc),
        EquityCurve=equity_curve,
        Returns=returns_series,
    )

    # Export equity if requested
    if collect_equity and export_equity_csv and equity_curve:
        eq_df = pd.DataFrame(equity_curve, columns=['datetime', 'equity'])
        try:
            eq_df.to_csv(export_equity_csv, index=False)
        except Exception:
            pass

    return payload


# Statistical utilities: PSR / Deflated Sharpe (approx) and Reality Check

def compute_sharpe_from_returns(ret: pd.Series, freq_per_year: int = 252) -> float:
    if ret.empty:
        return float('nan')
    mu = ret.mean() * freq_per_year
    sigma = ret.std(ddof=1) * math.sqrt(freq_per_year)
    if sigma == 0:
        return float('nan')
    return mu / sigma


def probabilistic_sharpe_ratio(sr_hat: float, n: int, sr0: float = 0.0) -> float:
    if n <= 1 or not np.isfinite(sr_hat):
        return float('nan')
    z = (sr_hat - sr0) * math.sqrt(n)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(sr_hat: float, n: int, skew: float, kurt: float, m: int) -> float:
    # Approximate DSR per Bailey & Lopez de Prado; simplified
    if n <= 1 or not np.isfinite(sr_hat):
        return float('nan')
    sigma_sr = math.sqrt((1 - skew * sr_hat + (kurt - 1) * sr_hat * sr_hat / 4) / (n - 1))
    # Expected max under M trials (approx of Gaussian max)
    if m <= 1:
        sr_max = 0.0
    else:
        p = (m - 0.3) / (m + 0.4)
        sr_max = norm.ppf(p) * sigma_sr
    z = (sr_hat - sr_max) / sigma_sr if sigma_sr > 0 else float('inf')
    return float(norm.cdf(z))


def reality_check_pvalue(ret: pd.Series, B: int = 500, block: int = 10) -> float:
    # Simple block bootstrap p-value for mean(ret) > 0
    if ret.empty:
        return float('nan')
    n = len(ret)
    mu = ret.mean()
    stats = []
    rng = np.random.default_rng(42)
    for _ in range(B):
        idx = []
        i = 0
        while i < n:
            start = rng.integers(0, n)
            L = min(block, n - i)
            seg = list(range(start, min(start + L, n)))
            if len(seg) < L:
                seg += list(range(0, L - len(seg)))
            idx.extend(seg)
            i += L
        boot = ret.iloc[idx].reset_index(drop=True)
        stats.append(boot.mean())
    p = float((np.sum(np.array(stats) >= mu) + 1) / (B + 1))
    return p

# Purged / embargoed time-series CV

def time_series_cv_windows(index: pd.DatetimeIndex, n_splits: int = 5, purge_frac: float = 0.1, embargo_frac: float = 0.05) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    # Returns list of (train_start, train_end, test_start, test_end) dates
    n = len(index)
    fold_size = n // (n_splits + 1)
    windows = []
    for k in range(n_splits):
        train_end_idx = fold_size * (k + 1)
        test_end_idx = fold_size * (k + 2)
        train_start_idx = 0
        # Purge last portion of train
        purge = int(fold_size * purge_frac)
        embargo = int(fold_size * embargo_frac)
        train_end_idx_adj = max(train_end_idx - purge, 1)
        test_start_idx = train_end_idx + embargo
        test_end_idx = min(test_end_idx, n - 1)
        if test_start_idx >= test_end_idx:
            continue
        windows.append((index[train_start_idx], index[train_end_idx_adj], index[test_start_idx], index[test_end_idx]))
    return windows


def evaluate_params_cv(symbol: str, interval: str, start: str, end: str, params: Dict[str, Any], n_splits: int = 5, purge_frac: float = 0.1, embargo_frac: float = 0.05) -> Dict[str, Any]:
    df = fetch_data(symbol, start, end, interval)
    idx = df.index
    windows = time_series_cv_windows(idx, n_splits=n_splits, purge_frac=purge_frac, embargo_frac=embargo_frac)
    scores = []
    dd_list = []
    trades_list = []
    psr_list = []
    for (tr_s, tr_e, te_s, te_e) in windows:
        res = run_backtest(symbol=symbol, start=str(tr_s.date()), end=str(te_e.date()), interval=interval, collect_equity=True, **params)
        scores.append(res.get('Sharpe') or 0.0)
        dd_list.append(res.get('MaxDrawdownPct') or 0.0)
        trades_list.append(res.get('TotalTrades') or 0)
        # PSR from returns
        ret = res.get('Returns')
        if ret:
            ret_series = pd.DataFrame(ret, columns=['dt','r']).set_index('dt')['r']
            sr_hat = compute_sharpe_from_returns(ret_series)
            psr_list.append(probabilistic_sharpe_ratio(sr_hat, max(2, len(ret_series))))
    return dict(
        mean_sharpe=float(np.nanmean(scores)) if scores else float('nan'),
        var_sharpe=float(np.nanvar(scores)) if scores else float('nan'),
        mean_dd=float(np.nanmean(dd_list)) if dd_list else float('nan'),
        mean_trades=float(np.nanmean(trades_list)) if trades_list else float('nan'),
        mean_psr=float(np.nanmean(psr_list)) if psr_list else float('nan'),
        folds=len(windows),
    )

# Multi-objective optimization via Optuna NSGA-II

def optimize_moo(symbol: str, start: str, end: str, interval: str = '1d', n_trials: int = 40):
    if optuna is None:
        raise RuntimeError("optuna is not installed. Please `pip install optuna`.")

    def objective(trial: 'optuna.Trial'):
        domcycle = trial.suggest_int("domcycle", 10, 60, step=2)
        vibration = trial.suggest_int("vibration", 5, 20)
        leveling = trial.suggest_float("leveling", 5.0, 30.0)
        sl_pct = trial.suggest_float("sl_pct", 0.2, 5.0)
        tp_pct = trial.suggest_float("tp_pct", 0.5, 15.0)
        max_bars_in_trade = trial.suggest_int("max_bars_in_trade", 0, 120)
        regime_filter = trial.suggest_categorical("regime_filter", [True, False])
        regime_ma_len = trial.suggest_int("regime_ma_len", 50, 300)
        exit_on_regime_flip = trial.suggest_categorical("exit_on_regime_flip", [True, False])
        use_atr_exits = trial.suggest_categorical("use_atr_exits", [False, True])
        atr_period = trial.suggest_int("atr_period", 7, 40)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 5.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.0, 10.0)
        trailing_stop = trial.suggest_categorical("trailing_stop", [False, True])
        rearm_exits = trial.suggest_categorical("rearm_exits", [False, True])

        res = run_backtest(
            symbol=symbol, start=start, end=end, interval=interval,
            domcycle=domcycle, vibration=vibration, leveling=leveling,
            sl_pct=sl_pct, tp_pct=tp_pct,
            max_bars_in_trade=max_bars_in_trade,
            regime_filter=regime_filter, regime_ma_len=regime_ma_len, exit_on_regime_flip=exit_on_regime_flip,
            use_atr_exits=use_atr_exits, atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
            trailing_stop=trailing_stop, rearm_exits=rearm_exits,
        )
        sharpe = res.get('Sharpe') or 0.0
        dd = res.get('MaxDrawdownPct') or 0.0
        trades = res.get('TotalTrades') or 0
        return sharpe, dd, trades

    study = optuna.create_study(directions=["maximize", "minimize", "maximize"])
    study.optimize(objective, n_trials=n_trials)
    return study

# Parameter stability: local perturbation robustness

def stability_score(symbol: str, interval: str, start: str, end: str, params: Dict[str, Any]) -> float:
    base = run_backtest(symbol=symbol, start=start, end=end, interval=interval, **params)
    base_score = (base.get('Sharpe') or 0.0) - 0.05 * (base.get('MaxDrawdownPct') or 0.0) + 0.001 * (base.get('TotalTrades') or 0)
    scores = []
    deltas = [(-2,0,0), (2,0,0), (0,-2,0), (0,2,0), (0,0,-2.0), (0,0,2.0)]
    for d_dc, d_v, d_lev in deltas:
        p2 = params.copy()
        p2['domcycle'] = int(max(10, min(60, p2.get('domcycle', 20) + d_dc)))
        p2['vibration'] = int(max(5, min(20, p2.get('vibration', 10) + d_v)))
        p2['leveling'] = float(max(1.0, min(50.0, p2.get('leveling', 10.0) + d_lev)))
        res = run_backtest(symbol=symbol, start=start, end=end, interval=interval, **p2)
        score = (res.get('Sharpe') or 0.0) - 0.05 * (res.get('MaxDrawdownPct') or 0.0) + 0.001 * (res.get('TotalTrades') or 0)
        scores.append(score)
    # Higher stability if avg near base and low variance
    if not scores:
        return 0.0
    return float(max(0.0, 1.0 - np.std(scores + [base_score])))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cRSI Backtrader strategy on symbols")
    parser.add_argument('--symbols', type=str, default='^GSPC,^DJI,^IXIC,^NDX,SPY,DIA,QQQ', help='Comma-separated symbols')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--interval', type=str, default='1d', help='1d, 1h, 4h (indices generally daily only)')
    parser.add_argument('--cash', type=float, default=50000.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--slippage', type=float, default=0.0002)

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
    parser.add_argument('--coc', action='store_true', help='Cheat-on-close execution (fills on bar close). Off=next-bar open fills')
    parser.add_argument('--debug-data', action='store_true', help='Print detailed data fetching diagnostics')

    # CV / MOO / Stats
    parser.add_argument('--cv', action='store_true', help='Evaluate current params with purged/embargoed time-series CV')
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--cv-purge', type=float, default=0.1)
    parser.add_argument('--cv-embargo', type=float, default=0.05)
    parser.add_argument('--moo', action='store_true', help='Run multi-objective (NSGA-II) optimization')
    parser.add_argument('--psr', action='store_true', help='Compute PSR/DSR and Reality Check p-value for best params (single symbol)')

    parser.add_argument('--csv', type=str, default=None, help='Optional path to save results CSV')

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    global DEBUG_FETCH
    DEBUG_FETCH = bool(args.debug_data) or os.environ.get('CRSI_DEBUG') == '1'

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

    # CV evaluation
    if args.cv:
        if len(symbols) != 1:
            print("CV requires exactly one symbol.", file=sys.stderr)
            return 2
        params = dict(domcycle=args.domcycle, vibration=args.vibration, leveling=args.leveling,
                      sl_pct=args.sl, tp_pct=args.tp, enable_short=args.short,
                      max_bars_in_trade=args.bars, max_leverage=args.maxleverage,
                      max_loss_per_trade=args.maxloss, regime_filter=args.regime,
                      regime_ma_len=args.regimema, exit_on_regime_flip=args.regime_exit,
                      use_atr_exits=args.atr, atr_period=args.atrperiod, atr_sl_mult=args.atrsl, atr_tp_mult=args.atrtp,
                      trailing_stop=args.trail, rearm_exits=args.rearm, daily_loss_cap=args.dailycap,
                      daily_loss_cap_pct=args.dailycap_pct, cheat_on_close=args.coc)
        cvres = evaluate_params_cv(symbol=symbols[0], interval=args.interval, start=args.start, end=args.end,
                                    params=params, n_splits=args.cv_splits, purge_frac=args.cv_purge, embargo_frac=args.cv_embargo)
        print(cvres)
        return 0

    # Multi-objective optimization
    if args.moo:
        if len(symbols) != 1:
            print("MOO requires exactly one symbol.", file=sys.stderr)
            return 2
        study = optimize_moo(symbol=symbols[0], start=args.start, end=args.end, interval=args.interval, n_trials=args.trials)
        pareto = []
        for t in study.best_trials:
            pareto.append({'values': t.values, 'params': t.params})
        print({'pareto': pareto})
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
        # Optional: PSR/DSR and Reality Check on best
        if args.psr:
            res = run_backtest(symbol=symbols[0], start=args.start, end=args.end, interval=args.interval, collect_equity=True, **study.best_params)
            ret = res.get('Returns')
            if ret:
                s = pd.DataFrame(ret, columns=['dt','r']).set_index('dt')['r']
                sr = compute_sharpe_from_returns(s)
                psr = probabilistic_sharpe_ratio(sr, len(s))
                skew = float(s.skew())
                kurt = float(s.kurtosis() + 3)
                dsr = deflated_sharpe_ratio(sr, len(s), skew, kurt, args.trials)
                pval = reality_check_pvalue(s)
                print({'Sharpe': sr, 'PSR': psr, 'DSR': dsr, 'RealityCheckP': pval})
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
                cheat_on_close=args.coc,
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