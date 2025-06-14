#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2024 Ran Aroussi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ˜
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io as _io
import datetime as _dt
import pandas as _pd
import numpy as _np
# import yfinance as _yf - 在中国大陆无法使用
from . import stats as _stats
import inspect
import warnings


def _mtd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-%m-01")]


def _qtd(df):
    date = _dt.datetime.now()
    for q in [1, 4, 7, 10]:
        if date.month <= q:
            return df[df.index >= _dt.datetime(date.year, q, 1).strftime("%Y-%m-01")]
    return df[df.index >= date.strftime("%Y-%m-01")]


def _ytd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-01-01")]


def _pandas_date(df, dates):
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    n = _dt.datetime.now()
    daterange = _pd.date_range(_dt.date(n.year, n.month, 1), n)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    """Get last N rows relative to another row in pandas"""
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    dfs = [df.shift(i) for i in _np.arange(shift)]
    for ix, dfi in enumerate(dfs[1:]):
        dfs[ix + 1].columns = [str(col) for col in dfi.columns + str(ix + 1)]
    return _pd.concat(dfs, axis=1, sort=True)


def to_returns(prices, rf=0.0):
    """Calculates the simple arithmetic returns of a price series"""
    return _prepare_returns(prices, rf)


def to_prices(returns, base=1e5):
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    return base + base * _stats.compsum(returns)


def log_returns(returns, rf=0.0, nperiods=None):
    """Shorthand for to_log_returns"""
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns, rf=0.0, nperiods=None):
    """Converts returns series to log returns"""
    returns = _prepare_returns(returns, rf, nperiods)
    try:
        return _np.log(returns + 1).replace([_np.inf, -_np.inf], float("NaN"))
    except Exception:
        return 0.0


def exponential_stdev(returns, window=30, is_halflife=False):
    """Returns series representing exponential volatility of returns"""
    returns = _prepare_returns(returns)
    halflife = window if is_halflife else None
    return returns.ewm(
        com=None, span=window, halflife=halflife, min_periods=window
    ).std()


def rebase(prices, base=100.0):
    """
    Rebase all series to a given intial base.
    This makes comparing/plotting different series together easier.
    Args:
        * prices: Expects a price series/dataframe
        * base (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().iloc[0] * base


def group_returns(returns, groupby, compounded=False):
    """Summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """Aggregates returns based on date periods"""
    if period is None or "day" in period:
        return returns
    index = returns.index

    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "YE" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    if "eom" in period or period == "ME":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    if "eoq" in period or period == "QE":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    return returns


def to_excess_returns(returns, rf, nperiods=None):
    """
    Calculates excess returns by subtracting
    risk-free returns from total returns

    Args:
        * returns (Series, DataFrame): Returns
        * rf (float, Series, DataFrame): Risk-Free rate(s)
        * nperiods (int): Optional. If provided, will convert rf to different
            frequency using deannualize
    Returns:
        * excess_returns (Series, DataFrame): Returns - rf
    """
    if isinstance(rf, int):
        rf = float(rf)

    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]

    if nperiods is not None:
        # deannualize
        rf = _np.power(1 + rf, 1.0 / nperiods) - 1.0

    df = returns - rf
    df = df.tz_localize(None)
    return df


def _prepare_prices(data, base=1.0):
    """Converts return data into prices + cleanup"""
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))

    # 只有pandas对象且索引是DatetimeIndex或PeriodIndex时才调用tz_localize方法
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        if isinstance(data.index, (_pd.DatetimeIndex, _pd.PeriodIndex)):
            data = data.tz_localize(None)
    return data


def _prepare_returns(data, rf=0.0, nperiods=None):
    """Converts price data into returns + cleanup"""
    data = data.copy()
    function = inspect.stack()[1][3]
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() >= 0 and data[col].dropna().max() > 1:
                data[col] = data[col].pct_change()
    elif isinstance(data, _pd.Series):
        if data.min() >= 0 and data.max() > 1:
            data = data.pct_change()
    elif isinstance(data, _np.ndarray):
        if data.min() >= 0 and data.max() > 1:
            # 对NumPy数组计算百分比变化
            data = _np.diff(data) / data[:-1]
            # 在开头添加一个0，保持数组长度一致
            data = _np.insert(data, 0, 0)

    # cleanup data
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.replace([_np.inf, -_np.inf], float("NaN"))
        data = data.fillna(0).replace([_np.inf, -_np.inf], float("NaN"))
    elif isinstance(data, _np.ndarray):
        # 处理numpy数组
        data = _np.where(_np.isinf(data), _np.nan, data)
        data = _np.nan_to_num(data, nan=0.0)
    unnecessary_function_calls = [
        "_prepare_benchmark",
        "cagr",
        "gain_to_pain_ratio",
        "rolling_volatility",
    ]

    if function not in unnecessary_function_calls:
        if rf > 0:
            # 只对pandas对象应用to_excess_returns
            if isinstance(data, (_pd.DataFrame, _pd.Series)):
                return to_excess_returns(data, rf, nperiods)

    # 只有pandas对象才调用tz_localize方法
    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        if isinstance(data.index, (_pd.DatetimeIndex, _pd.PeriodIndex)):
            data = data.tz_localize(None)
    return data


def download_returns(ticker, period="max", proxy=None):
    warnings.warn(
        "yfinance在中国大陆无法使用。请手动提供数据或使用其他数据源。"
        "\n您可以通过以下方式提供数据："
        "\n1. 使用本地CSV文件加载数据"
        "\n2. 使用其他可用的数据API"
        "\n3. 手动创建pandas.Series或DataFrame对象"
        "\n4. 使用quantstats.synthetic_data模块创建合成数据",
        UserWarning
    )
    
    # 返回空的Series，用户需要自行提供数据
    if isinstance(period, _pd.DatetimeIndex):
        index = period
    else:
        # 创建一个默认的日期范围
        index = _pd.date_range(start='2000-01-01', periods=10, freq='D')
    
    return _pd.Series(index=index, dtype=float, name=ticker)


def _prepare_benchmark(benchmark=None, period="max", rf=0.0, prepare_returns=True):
    """
    Fetch benchmark if ticker is provided, and pass through
    _prepare_returns()

    period can be options or (expected) _pd.DatetimeIndex range
    
    由于在中国大陆无法使用yfinance，如果benchmark是字符串，将返回警告并创建空的Series。
    请直接提供benchmark数据作为Series或DataFrame。
    """
    if benchmark is None:
        return None

    if isinstance(benchmark, str):
        warnings.warn(
            f"由于在中国大陆无法使用yfinance，无法下载{benchmark}的数据。"
            "\n请直接提供benchmark数据作为Series或DataFrame。"
            "\n您也可以使用quantstats.synthetic_data模块创建合成基准数据。",
            UserWarning
        )
        # 创建一个空的Series作为占位符
        if isinstance(period, _pd.DatetimeIndex):
            index = period
        else:
            # 创建一个默认的日期范围
            index = _pd.date_range(start='2000-01-01', periods=10, freq='D')
        
        benchmark = _pd.Series(index=index, dtype=float, name=benchmark)

    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()
        
    # 确保benchmark有一个日期索引
    if not isinstance(benchmark.index, _pd.DatetimeIndex):
        # 创建一个日期索引
        date_index = _pd.date_range(start='2000-01-01', periods=len(benchmark))
        benchmark = _pd.Series(benchmark.values, index=date_index, name=benchmark.name)

    if isinstance(period, _pd.DatetimeIndex) and set(period) != set(benchmark.index):

        # Adjust Benchmark to Strategy frequency
        benchmark_prices = to_prices(benchmark, base=1)
        new_index = _pd.date_range(start=period[0], end=period[-1], freq="D")
        benchmark = (
            benchmark_prices.reindex(new_index, method="bfill")
            .reindex(period)
            .pct_change()
            .fillna(0)
        )
        benchmark = benchmark[benchmark.index.isin(period)]

    benchmark = benchmark.tz_localize(None)

    if prepare_returns:
        return _prepare_returns(benchmark.dropna(), rf=rf)
    return benchmark.dropna()


def _round_to_closest(val, res, decimals=None):
    """Round to closest resolution"""
    # 处理无穷大和NaN值
    if _np.isinf(val) or _np.isnan(val):
        return val
    
    if decimals is None and "." in str(res):
        decimals = len(str(res).split(".")[1])
    return round(round(val / res) * res, decimals)


def _file_stream():
    """Returns a file stream"""
    return _io.BytesIO()


def _in_notebook(matplotlib_inline=False):
    """Identify enviroment (notebook, terminal, etc)"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            if matplotlib_inline:
                get_ipython().run_line_magic("matplotlib", "inline")
            return True
        if shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        # Other type (?)
        return False
    except NameError:
        # Probably standard Python interpreter
        return False


def _count_consecutive(data):
    """Counts consecutive data (like cumsum() with reset on zeroes)"""

    def _count(data):
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def _score_str(val):
    """Returns + sign for positive values (used in plots)"""
    return ("" if "-" in val else "+") + str(val)


def make_index(
    ticker_weights, rebalance="1M", period="max", returns=None, match_dates=False
):
    """
    Makes an index out of the given tickers and weights.
    Optionally you can pass a dataframe with the returns.
    
    由于在中国大陆无法使用yfinance，您必须提供returns参数。

    Args:
        * ticker_weights (Dict): A python dict with tickers as keys
            and weights as values
        * rebalance: Pandas resample interval or None for never
        * period: time period of the returns to be downloaded
        * returns: Optional, DataFrame or dict of returns to use instead of
            downloading from Yahoo
        * match_dates: Optional, match the dates of all returns to the first ticker

    Returns:
        * A portfolio Series
    """

    if returns is None:
        warnings.warn(
            "由于在中国大陆无法使用yfinance，必须提供returns参数。"
            "\n您可以使用quantstats.synthetic_data模块创建合成数据，或提供自己的数据。",
            UserWarning
        )
        raise ValueError("必须提供returns参数，因为无法使用yfinance下载数据。")

    # Convert any dicts to DataFrame
    if isinstance(returns, dict):
        returns = _pd.DataFrame(returns)

    # Ensure all tickers are in returns
    missing_tickers = [ticker for ticker in ticker_weights.keys() if ticker not in returns.columns]
    if missing_tickers:
        warnings.warn(
            f"以下ticker在returns中不存在: {', '.join(missing_tickers)}"
            "\n请确保所有需要的ticker都在returns中提供。",
            UserWarning
        )

    # Use only tickers that are in returns
    valid_tickers = [ticker for ticker in ticker_weights.keys() if ticker in returns.columns]
    if not valid_tickers:
        raise ValueError("没有有效的ticker可用于创建指数。请检查ticker_weights和returns。")

    # Normalize weights to sum to 1 for valid tickers
    weights = {ticker: weight for ticker, weight in ticker_weights.items() if ticker in valid_tickers}
    total_weight = sum(weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

    # Create portfolio
    portfolio = _pd.DataFrame(index=returns.index)

    # Match dates of all returns to the first ticker if requested
    if match_dates and len(valid_tickers) > 1:
        first_ticker = valid_tickers[0]
        first_ticker_dates = returns[first_ticker].dropna().index
        returns = returns.loc[first_ticker_dates]

    # Add ticker returns to portfolio
    for ticker, weight in weights.items():
        portfolio[ticker] = returns[ticker] * weight

    # Resample portfolio based on rebalance interval
    if rebalance is not None:
        portfolio = portfolio.dropna()
        portfolio = portfolio.resample(rebalance).apply(
            lambda x: _stats.comp(x) if len(x) > 0 else 0
        )

    # Sum up all ticker returns for each period
    portfolio = portfolio.sum(axis=1)
    portfolio.name = "Portfolio"

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """Dirty method for flattening multi-index dataframe"""
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df = df.set_index(set_index)
    return df


def make_portfolio(
    returns, start_balance=1e5, mode="comp", round_to=None, verbose=True
):
    """
    Calculates compounded value of portfolio

    Args:
        * returns (Series, DataFrame): Returns
        * start_balance (float): Starting balance, default 1e5 (100,000)
        * mode (str): Compounding mode, either "comp" for compounded or "sum" for sum
        * round_to (float): Round to the nearest number
        * verbose (bool): Print progress

    Returns:
        * Portfolio value in currency
    """

    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            raise ValueError("returns must be a Series or one-column DataFrame")
        returns = returns[returns.columns[0]]

    returns = returns.fillna(0).replace([_np.inf, -_np.inf], 0)

    if mode.lower() == "comp":
        portfolio = start_balance * (1 + returns).cumprod()
    else:  # "sum"
        portfolio = start_balance * (1 + returns.cumsum())

    if round_to:
        portfolio = _round_to_closest(portfolio, round_to)

    if verbose:
        print(
            "%s starting balance of %s grew to %s (%.2f%%)"
            % (
                returns.name or "Portfolio",
                start_balance,
                portfolio.iloc[-1].round(2),
                (portfolio.iloc[-1] / start_balance - 1) * 100,
            )
        )

    return portfolio


def make_portfolio_from_prices(
    prices, weights=None, start_balance=1e5, mode="shares", round_to=None, verbose=True
):
    """
    Calculates compounded value of portfolio

    Args:
        * prices (Series, DataFrame): Prices
        * weights (list, dict): List of ticker weights with same length
            as prices.columns. If None, equal weights assumed.
        * start_balance (float): Starting balance, default 1e5 (100,000)
        * mode (str): Share allocation mode, either "shares" for shares or
            "currency" for currency
        * round_to (float): Round to the nearest number
        * verbose (bool): Print progress

    Returns:
        * Portfolio value in currency
    """

    if isinstance(prices, _pd.Series):
        prices = _pd.DataFrame(prices)

    # Make sure we're working with DataFrame
    if not isinstance(prices, _pd.DataFrame):
        raise ValueError("prices must be a Series or DataFrame")

    # Ensure we have weights
    if weights is None:
        weights = [1 / len(prices.columns) for _ in range(len(prices.columns))]

    # Convert weights to dict if list provided
    if isinstance(weights, list):
        if len(weights) != len(prices.columns):
            raise ValueError("weights must have same length as prices.columns")
        weights = dict(zip(prices.columns, weights))

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

    # Calculate portfolio value based on mode
    if mode.lower() == "shares":
        # Allocate based on number of shares
        shares = {}
        for ticker, weight in weights.items():
            ticker_price = prices[ticker].iloc[0]
            shares[ticker] = (start_balance * weight) / ticker_price

        # Calculate portfolio value over time
        portfolio = _pd.Series(index=prices.index, dtype=float)
        for i, date in enumerate(prices.index):
            portfolio[date] = sum(shares[ticker] * prices[ticker][date] for ticker in weights.keys())

    else:  # "currency"
        # Allocate based on currency amount
        portfolio = _pd.Series(index=prices.index, dtype=float)
        portfolio.iloc[0] = start_balance

        # Rebalance at each step
        for i in range(1, len(prices.index)):
            prev_date = prices.index[i-1]
            curr_date = prices.index[i]
            prev_value = portfolio[prev_date]

            # Calculate returns for each ticker
            ticker_returns = {}
            for ticker in weights.keys():
                prev_price = prices[ticker][prev_date]
                curr_price = prices[ticker][curr_date]
                ticker_returns[ticker] = (curr_price / prev_price - 1) * weights[ticker]

            # Apply returns to portfolio
            portfolio[curr_date] = prev_value * (1 + sum(ticker_returns.values()))

    if round_to:
        portfolio = _round_to_closest(portfolio, round_to)

    if verbose:
        print(
            "Portfolio starting balance of %s grew to %s (%.2f%%)"
            % (
                start_balance,
                portfolio.iloc[-1].round(2),
                (portfolio.iloc[-1] / start_balance - 1) * 100,
            )
        )

    return portfolio
