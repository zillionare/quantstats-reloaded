#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantstats import utils

# 创建测试数据
@pytest.fixture
def returns_data():
    # 创建一个包含正负收益的Series，用于测试
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')  # 工作日
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    return returns

@pytest.fixture
def price_data():
    # 创建价格数据用于测试
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    np.random.seed(44)
    prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates)
    return prices

@pytest.fixture
def sample_dataframe():
    # 创建测试用的DataFrame
    dates = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, len(dates)),
        'B': np.random.normal(0, 1, len(dates))
    }, index=dates)
    return data

# 测试数据转换函数

def test_to_returns(price_data):
    """测试价格转收益率"""
    returns = utils.to_returns(price_data)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(price_data)
    
    # 测试带风险自由利率
    returns_rf = utils.to_returns(price_data, rf=0.02)
    assert isinstance(returns_rf, pd.Series)

def test_to_prices(returns_data):
    """测试收益率转价格"""
    prices = utils.to_prices(returns_data)
    assert isinstance(prices, pd.Series)
    assert len(prices) == len(returns_data)
    
    # 测试不同基准价格
    prices_base = utils.to_prices(returns_data, base=1000)
    assert isinstance(prices_base, pd.Series)
    assert prices_base.iloc[0] > 900  # 应该接近1000

def test_to_log_returns(returns_data):
    """测试转换为对数收益率"""
    log_returns = utils.to_log_returns(returns_data)
    assert isinstance(log_returns, pd.Series)
    assert len(log_returns) == len(returns_data)
    
    # 测试带风险自由利率
    log_returns_rf = utils.to_log_returns(returns_data, rf=0.02)
    assert isinstance(log_returns_rf, pd.Series)

def test_log_returns(returns_data):
    """测试log_returns简写函数"""
    log_ret1 = utils.log_returns(returns_data)
    log_ret2 = utils.to_log_returns(returns_data)
    
    # 两个函数应该返回相同结果
    pd.testing.assert_series_equal(log_ret1, log_ret2)

def test_exponential_stdev(returns_data):
    """测试指数标准差"""
    exp_std = utils.exponential_stdev(returns_data)
    assert isinstance(exp_std, pd.Series)
    assert len(exp_std) == len(returns_data)

    # 测试不同窗口大小
    exp_std_60 = utils.exponential_stdev(returns_data, window=60)
    assert isinstance(exp_std_60, pd.Series)

    # 测试半衰期模式（不能同时指定span和halflife）
    exp_std_hl = utils.exponential_stdev(returns_data, window=30, is_halflife=True)
    assert isinstance(exp_std_hl, pd.Series)

def test_rebase(price_data):
    """测试重新基准化"""
    rebased = utils.rebase(price_data)
    assert isinstance(rebased, pd.Series)
    assert abs(rebased.iloc[0] - 100.0) < 1e-10  # 第一个值应该是100
    
    # 测试不同基准值
    rebased_200 = utils.rebase(price_data, base=200.0)
    assert abs(rebased_200.iloc[0] - 200.0) < 1e-10

def test_aggregate_returns(returns_data):
    """测试聚合收益率"""
    # 测试日收益率（默认）
    daily = utils.aggregate_returns(returns_data)
    pd.testing.assert_series_equal(daily, returns_data)

    # 测试月度收益率（使用正确的期间标识符）
    monthly = utils.aggregate_returns(returns_data, period='ME')  # Month End
    assert isinstance(monthly, pd.Series)
    assert len(monthly) <= 12  # 最多12个月

    # 测试季度收益率
    quarterly = utils.aggregate_returns(returns_data, period='QE')  # Quarter End
    assert isinstance(quarterly, pd.Series)

    # 测试年度收益率
    yearly = utils.aggregate_returns(returns_data, period='YE')  # Year End
    assert isinstance(yearly, pd.Series)

    # 测试周收益率
    weekly = utils.aggregate_returns(returns_data, period='W')
    assert isinstance(weekly, pd.Series)

    # 测试非复合模式
    monthly_simple = utils.aggregate_returns(returns_data, period='ME', compounded=False)
    assert isinstance(monthly_simple, pd.Series)

def test_to_excess_returns(returns_data):
    """测试超额收益率"""
    # 测试固定风险自由利率
    excess = utils.to_excess_returns(returns_data, rf=0.02)
    assert isinstance(excess, pd.Series)
    assert len(excess) == len(returns_data)
    
    # 测试Series风险自由利率
    rf_series = pd.Series(0.02, index=returns_data.index)
    excess_series = utils.to_excess_returns(returns_data, rf=rf_series)
    assert isinstance(excess_series, pd.Series)
    
    # 测试带期间数的情况
    excess_periods = utils.to_excess_returns(returns_data, rf=0.02, nperiods=252)
    assert isinstance(excess_periods, pd.Series)

def test_multi_shift(sample_dataframe):
    """测试多重移位"""
    # 测试DataFrame
    shifted_df = utils.multi_shift(sample_dataframe, shift=3)
    assert isinstance(shifted_df, pd.DataFrame)
    assert shifted_df.shape[1] == sample_dataframe.shape[1] * 3  # 应该有3倍的列数
    
    # 测试Series
    series = sample_dataframe['A']
    shifted_series = utils.multi_shift(series, shift=2)
    assert isinstance(shifted_series, pd.DataFrame)
    assert shifted_series.shape[1] == 2

# 测试日期相关函数

def test_pandas_date_functions(sample_dataframe):
    """测试pandas日期函数"""
    # 测试_pandas_date
    specific_dates = ['2020-01-01', '2020-01-15']
    filtered = utils._pandas_date(sample_dataframe, specific_dates)
    assert isinstance(filtered, pd.DataFrame)
    assert len(filtered) <= 2
    
    # 测试单个日期
    single_date = utils._pandas_date(sample_dataframe, '2020-01-01')
    assert isinstance(single_date, pd.DataFrame)

def test_make_portfolio(returns_data):
    """测试组合构建"""
    # 测试默认复合模式
    portfolio = utils.make_portfolio(returns_data)
    assert isinstance(portfolio, pd.Series)
    assert len(portfolio) == len(returns_data) + 1  # 包含起始值
    
    # 测试累计模式
    portfolio_sum = utils.make_portfolio(returns_data, mode='sum')
    assert isinstance(portfolio_sum, pd.Series)
    
    # 测试不同起始余额
    portfolio_1m = utils.make_portfolio(returns_data, start_balance=1e6)
    assert isinstance(portfolio_1m, pd.Series)
    assert portfolio_1m.iloc[0] == 1e6
    
    # 测试四舍五入
    portfolio_rounded = utils.make_portfolio(returns_data, round_to=2)
    assert isinstance(portfolio_rounded, pd.Series)
    
    # 测试NumPy数组输入
    returns_array = returns_data.values
    portfolio_array = utils.make_portfolio(returns_array)
    assert isinstance(portfolio_array, np.ndarray)

# 测试辅助函数

def test_round_to_closest():
    """测试四舍五入到最近值"""
    # 测试正常值
    result = utils._round_to_closest(1.234, 0.1)
    assert abs(result - 1.2) < 1e-10
    
    # 测试无穷大
    result_inf = utils._round_to_closest(np.inf, 0.1)
    assert np.isinf(result_inf)
    
    # 测试NaN
    result_nan = utils._round_to_closest(np.nan, 0.1)
    assert np.isnan(result_nan)
    
    # 测试指定小数位数
    result_decimals = utils._round_to_closest(1.2345, 0.01, decimals=2)
    assert abs(result_decimals - 1.23) < 1e-10

def test_file_stream():
    """测试文件流"""
    stream = utils._file_stream()
    assert hasattr(stream, 'read')
    assert hasattr(stream, 'write')

def test_in_notebook():
    """测试notebook环境检测"""
    # 在测试环境中应该返回False
    result = utils._in_notebook()
    assert isinstance(result, bool)

def test_count_consecutive(returns_data):
    """测试连续计数"""
    # 创建测试数据：正负交替
    test_data = pd.Series([1, 1, 0, 1, 1, 1, 0, 0, 1])
    result = utils._count_consecutive(test_data > 0)
    assert isinstance(result, pd.Series)
    
    # 测试DataFrame
    test_df = pd.DataFrame({'A': [1, 1, 0, 1], 'B': [0, 1, 1, 0]})
    result_df = utils._count_consecutive(test_df > 0)
    assert isinstance(result_df, pd.DataFrame)

def test_score_str():
    """测试分数字符串格式化"""
    # 测试正数
    result_pos = utils._score_str("1.5")
    assert result_pos == "+1.5"
    
    # 测试负数
    result_neg = utils._score_str("-1.5")
    assert result_neg == "-1.5"
    
    # 测试零
    result_zero = utils._score_str("0.0")
    assert result_zero == "+0.0"

def test_flatten_dataframe(sample_dataframe):
    """测试DataFrame扁平化"""
    # 创建多级索引DataFrame
    multi_index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['first', 'second'])
    multi_df = pd.DataFrame(np.random.randn(3, 2), index=multi_index, columns=['X', 'Y'])
    
    flattened = utils._flatten_dataframe(multi_df)
    assert isinstance(flattened, pd.DataFrame)
    
    # 测试设置索引
    flattened_with_index = utils._flatten_dataframe(multi_df, set_index='first')
    assert isinstance(flattened_with_index, pd.DataFrame)

# 测试数据准备函数

def test_prepare_prices(returns_data):
    """测试价格数据准备"""
    # 这是一个内部函数，但我们可以测试它
    prices = utils._prepare_prices(returns_data)
    assert isinstance(prices, pd.Series)

def test_prepare_returns(price_data):
    """测试收益率数据准备"""
    # 这是一个内部函数，但我们可以测试它
    returns = utils._prepare_returns(price_data)
    assert isinstance(returns, pd.Series)
    
    # 测试带风险自由利率
    returns_rf = utils._prepare_returns(price_data, rf=0.02)
    assert isinstance(returns_rf, pd.Series)

def test_prepare_benchmark(returns_data):
    """测试基准数据准备"""
    # 测试字符串ticker
    benchmark = utils._prepare_benchmark('SPY', period='1mo')
    assert isinstance(benchmark, pd.Series)
    
    # 测试Series输入
    benchmark_series = utils._prepare_benchmark(returns_data)
    assert isinstance(benchmark_series, pd.Series)
    
    # 测试DataFrame输入
    df = pd.DataFrame({'returns': returns_data})
    benchmark_df = utils._prepare_benchmark(df)
    assert isinstance(benchmark_df, pd.Series)
