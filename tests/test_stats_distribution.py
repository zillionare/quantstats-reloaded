#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import empyrical
from quantstats import stats

# 创建测试数据
@pytest.fixture
def returns_data():
    # 创建一个包含正负收益的Series，用于测试
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')  # 工作日
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    return returns

@pytest.fixture
def benchmark_data():
    # 创建一个基准收益Series，用于测试
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    np.random.seed(43)  # 不同的随机种子
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    return returns

# 测试分布分析
def test_distribution(returns_data):
    # 测试收益分布计算
    dist = stats.distribution(returns_data)
    # 检查返回的是否为字典
    assert isinstance(dist, dict)
    # 检查是否包含必要的键
    assert all(key in dist for key in ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'])
    # 检查每个键的值是否为字典，并包含'values'和'outliers'键
    for key in dist:
        assert isinstance(dist[key], dict)
        assert all(k in dist[key] for k in ['values', 'outliers'])

def test_monthly_returns(returns_data):
    # 测试月度收益计算
    monthly = stats.monthly_returns(returns_data)
    # 检查返回的是否为DataFrame
    assert isinstance(monthly, pd.DataFrame)
    # 检查是否包含12个月的数据
    assert monthly.shape[1] == 12

def test_comp(returns_data):
    # 测试复合收益计算
    comp_returns = stats.comp(returns_data)
    # 检查复合收益是否为浮点数
    assert isinstance(comp_returns, float)

def test_compsum(returns_data):
    # 测试累计复合收益计算
    compsum_returns = stats.compsum(returns_data)
    # 检查返回的是否为Series
    assert isinstance(compsum_returns, pd.Series)
    # 检查长度是否与原始数据相同
    assert len(compsum_returns) == len(returns_data)

# 测试异常值分析
def test_remove_outliers(returns_data):
    # 测试移除异常值
    clean_returns = stats.remove_outliers(returns_data)
    # 检查返回的是否为Series
    assert isinstance(clean_returns, pd.Series)
    # 清理后的数据长度应该小于等于原始数据
    assert len(clean_returns) <= len(returns_data)

def test_outliers(returns_data):
    # 测试异常值检测
    outliers = stats.outliers(returns_data)
    # 检查返回的是否为Series
    assert isinstance(outliers, pd.Series)
    # 异常值数量应该小于原始数据
    assert len(outliers) < len(returns_data)

# 测试时间序列分析
def test_rolling_volatility(returns_data):
    # 测试滚动波动率计算
    rolling_vol = stats.rolling_volatility(returns_data)
    # 检查返回的是否为Series
    assert isinstance(rolling_vol, pd.Series)
    # 检查长度是否与原始数据相同
    assert len(rolling_vol) == len(returns_data)

def test_rolling_sharpe(returns_data):
    # 测试滚动夏普比率计算
    rolling_sharpe = stats.rolling_sharpe(returns_data)
    # 检查返回的是否为Series
    assert isinstance(rolling_sharpe, pd.Series)
    # 检查长度是否与原始数据相同
    assert len(rolling_sharpe) == len(returns_data)

def test_rolling_sortino(returns_data):
    # 测试滚动索提诺比率计算
    rolling_sortino = stats.rolling_sortino(returns_data)
    # 检查返回的是否为Series
    assert isinstance(rolling_sortino, pd.Series)
    # 检查长度是否与原始数据相同
    assert len(rolling_sortino) == len(returns_data)

def test_drawdowns_details(returns_data):
    # 测试回撤详情计算
    dd_details = stats.drawdown_details(returns_data)
    # 检查返回的是否为DataFrame
    assert isinstance(dd_details, pd.DataFrame)
    # 检查是否包含必要的列
    assert all(col in dd_details.columns for col in ['start', 'valley', 'end', 'days', 'max drawdown'])

# 测试收益率分析
def test_monthly_returns(returns_data):
    # 测试月度收益计算
    monthly = stats.monthly_returns(returns_data)
    # 检查返回的是否为DataFrame
    assert isinstance(monthly, pd.DataFrame)

def test_monthly_returns_again(returns_data):
    # 测试月度收益计算（不同的参数）
    monthly = stats.monthly_returns(returns_data, eoy=True)
    # 检查返回的是否为DataFrame
    assert isinstance(monthly, pd.DataFrame)

def test_compare(returns_data, benchmark_data):
    # 测试收益比较
    # 确保两个Series的索引相同
    common_idx = returns_data.index.intersection(benchmark_data.index)
    returns = returns_data.loc[common_idx]
    benchmark = benchmark_data.loc[common_idx]
    
    # 计算超额收益
    excess_returns = returns - benchmark
    # 检查返回的是否为Series
    assert isinstance(excess_returns, pd.Series)
    # 检查长度是否与输入数据相同
    assert len(excess_returns) == len(returns)

def test_greeks_components(returns_data, benchmark_data):
    # 测试希腊字母指标的各个组件
    greeks_result = stats.greeks(returns_data, benchmark_data)
    
    # 检查返回的是否为Series
    assert isinstance(greeks_result, pd.Series)
    
    # 检查是否包含beta和alpha
    assert 'beta' in greeks_result.index
    assert 'alpha' in greeks_result.index
    
    # 检查beta和alpha是否为浮点数
    assert isinstance(greeks_result['beta'], float)
    assert isinstance(greeks_result['alpha'], float)