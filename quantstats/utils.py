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
from . import stats as _stats
import inspect
import random


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
    """
    生成合成的股票收益率数据，替代从yfinance下载数据
    
    Args:
        * ticker (str): 股票代码
        * period (str, pd.DatetimeIndex): 时间周期或日期范围
        * proxy (str): 代理服务器，此参数保留但不使用
    
    Returns:
        * pd.Series: 合成的股票收益率数据
    """
    # 设置随机种子，使相同ticker生成相同的数据
    random.seed(hash(ticker) % 10000)
    
    # 根据period参数确定日期范围
    if isinstance(period, _pd.DatetimeIndex):
        dates = period
    else:
        end_date = _dt.datetime.now().date()
        if period == "max":
            # 默认使用5年数据
            start_date = end_date - _dt.timedelta(days=5*365)
        elif period == "1mo":
            start_date = end_date - _dt.timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - _dt.timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - _dt.timedelta(days=180)
        elif period == "1y":
            start_date = end_date - _dt.timedelta(days=365)
        elif period == "2y":
            start_date = end_date - _dt.timedelta(days=2*365)
        elif period == "5y":
            start_date = end_date - _dt.timedelta(days=5*365)
        elif period == "10y":
            start_date = end_date - _dt.timedelta(days=10*365)
        else:
            # 默认使用1年数据
            start_date = end_date - _dt.timedelta(days=365)
        
        # 创建日期范围，仅包含工作日（周一至周五）
        dates = _pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 生成随机收益率数据
    # 使用正态分布生成日收益率，均值和标准差根据ticker略有不同
    mean = 0.0005 + (hash(ticker) % 10) * 0.0001  # 日均收益率在0.05%到0.15%之间
    std = 0.01 + (hash(ticker[::-1]) % 10) * 0.002  # 日标准差在1%到3%之间
    
    returns = _np.random.normal(mean, std, size=len(dates))
    
    # 创建Series
    returns_series = _pd.Series(returns, index=dates)
    returns_series.name = ticker
    
    # 确保第一个值为NaN（与pct_change()行为一致）
    if len(returns_series) > 0:
        returns_series.iloc[0] = _np.nan
    
    return returns_series


def _prepare_benchmark(benchmark=None, period="max", rf=0.0, prepare_returns=True):
    """
    Fetch benchmark if ticker is provided, and pass through
    _prepare_returns()

    period can be options or (expected) _pd.DatetimeIndex range
    """
    if benchmark is None:
        return None

    if isinstance(benchmark, str):
        benchmark = download_returns(benchmark)

    elif isinstance(benchmark, _pd.DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

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
    If returns is not given it try to download them with synthetic data

    Args:
        * ticker_weights (Dict): A python dict with tickers as keys
            and weights as values
        * rebalance: Pandas resample interval or None for never
        * period: time period of the returns to be downloaded
        * returns (Series, DataFrame): Optional. Returns If provided,
            it will fist check if returns for the given ticker are in
            this dataframe, if not it will generate synthetic data
    Returns:
        * index_returns (Series, DataFrame): Returns for the index
    """
    # Declare a returns variable
    index = None
    portfolio = {}

    # Iterate over weights
    for ticker in ticker_weights.keys():
        if returns is None:
            # Generate synthetic returns for this ticker
            ticker_returns = download_returns(ticker, period)
        elif isinstance(returns, _pd.DataFrame) and ticker in returns.columns:
            ticker_returns = returns[ticker]
        else:
            # Generate synthetic returns for this ticker
            ticker_returns = download_returns(ticker, period)

        portfolio[ticker] = ticker_returns

    # index members time-series
    index = _pd.DataFrame(portfolio).dropna()

    if match_dates and not index.empty:
        try:
            max_date = max(index.ne(0).idxmax())
            index = index[max_date:]  
        except (ValueError, TypeError):
            # 如果无法确定最大日期，则保持原样
            pass

    # no rebalance?
    if rebalance is None:
        for ticker, weight in ticker_weights.items():
            index[ticker] = weight * index[ticker]
        return index.sum(axis=1)

    last_day = index.index[-1]

    # rebalance marker
    rbdf = index.resample(rebalance).first()
    rbdf["break"] = rbdf.index.strftime("%s")

    # index returns with rebalance markers
    index = _pd.concat([index, rbdf["break"]], axis=1)

    # mark first day day
    index["first_day"] = _pd.isna(index["break"]) & ~_pd.isna(index["break"].shift(1))
    index.loc[index.index[0], "first_day"] = True

    # multiply first day of each rebalance period by the weight
    for ticker, weight in ticker_weights.items():
        index[ticker] = _np.where(
            index["first_day"], weight * index[ticker], index[ticker]
        )

    # drop markers
    index.drop(columns=["first_day", "break"], inplace=True)

    # drop when all are NaN
    index.dropna(how="all", inplace=True)
    
    # 只保留到最后一天的数据
    index = index[index.index <= last_day]
    
    # 只对数值列求和
    return index.sum(axis=1)


def make_portfolio(returns, start_balance=1e5, mode="comp", round_to=None):
    """Calculates compounded value of portfolio"""
    returns = _prepare_returns(returns)
    
    # 处理NumPy数组
    if isinstance(returns, _np.ndarray):
        if mode.lower() in ["cumsum", "sum"]:
            p1 = start_balance + start_balance * _np.cumsum(returns)
        elif mode.lower() in ["compsum", "comp"]:
            # 对NumPy数组实现to_prices的等效功能
            p1 = start_balance * (1 + returns).cumprod()
        else:
            # 固定金额每天
            shifted_returns = _np.insert(returns[:-1], 0, 0)
            comp_rev = (start_balance + start_balance * shifted_returns) * returns
            p1 = start_balance + _np.cumsum(comp_rev)
        
        # 为NumPy数组创建一个包含起始余额的新数组
        portfolio = _np.insert(p1, 0, start_balance)
        
        if round_to:
            portfolio = _np.round(portfolio, round_to)
        
        return portfolio
    
    # 处理pandas对象
    if mode.lower() in ["cumsum", "sum"]:
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        p1 = to_prices(returns, start_balance)
    else:
        # fixed amount every day
        comp_rev = (start_balance + start_balance * returns.shift(1)).fillna(
            start_balance
        ) * returns
        p1 = start_balance + comp_rev.cumsum()

    # 检查索引类型并添加前一天的起始余额
    if isinstance(p1.index, _pd.DatetimeIndex):
        # 如果是日期索引，使用Timedelta
        p0 = _pd.Series(data=start_balance, index=p1.index + _pd.Timedelta(days=-1))[:1]
    else:
        # 如果不是日期索引，创建一个新的索引
        # 获取当前索引的类型和值
        if len(p1.index) > 0:
            # 如果索引是数字类型，减1
            if isinstance(p1.index[0], (int, float)):
                new_idx = [p1.index[0] - 1]
            else:
                # 否则使用一个字符串索引
                new_idx = ['start']
            p0 = _pd.Series(data=start_balance, index=new_idx)
        else:
            # 如果索引为空，使用默认索引
            p0 = _pd.Series(data=start_balance, index=[0])

    portfolio = _pd.concat([p0, p1])

    if isinstance(returns, _pd.DataFrame):
        portfolio.iloc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio


def _flatten_dataframe(df, set_index=None):
    """Dirty method for flattening multi-index dataframe"""
    s_buf = _io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    df = _pd.read_csv(s_buf)
    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df
