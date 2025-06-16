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

# 测试基本统计函数
def test_expected_return(returns_data):
    # 测试预期收益率计算
    qs_result = stats.expected_return(returns_data)
    
    # 验证结果是否为浮点数
    assert isinstance(qs_result, float)
    
    # 验证不同参数下的结果
    # 测试compounded=False的情况
    qs_result_non_compounded = stats.expected_return(returns_data, compounded=False)
    assert isinstance(qs_result_non_compounded, float)
    
    # 测试不同的aggregate参数
    qs_result_monthly = stats.expected_return(returns_data, aggregate='M')
    assert isinstance(qs_result_monthly, float)
    
    # 验证结果是否在合理范围内（根据测试数据的特性）
    assert -1 < qs_result < 1

def test_geometric_mean(returns_data):
    # 测试几何平均收益率
    qs_result = stats.geometric_mean(returns_data)
    
    # 验证结果是否为浮点数
    assert isinstance(qs_result, float)
    
    # 验证geometric_mean是expected_return的简写
    expected_return_result = stats.expected_return(returns_data)
    assert qs_result == expected_return_result
    
    # 测试不同的aggregate参数
    qs_result_weekly = stats.geometric_mean(returns_data, aggregate='W-MON')
    assert isinstance(qs_result_weekly, float)
    
    # 验证结果是否在合理范围内
    assert -1 < qs_result < 1

def test_volatility(returns_data):
    # 测试波动率计算
    qs_result = stats.volatility(returns_data)
    emp_result = empyrical.annual_volatility(returns_data)
    
    # 检查结果是否在合理范围内
    assert abs(qs_result - emp_result) < 0.01

def test_sharpe(returns_data):
    # 测试夏普比率计算
    qs_result = stats.sharpe(returns_data)
    emp_result = empyrical.sharpe_ratio(returns_data)
    
    # 检查结果是否在合理范围内
    assert abs(qs_result - emp_result) < 0.1

def test_sortino(returns_data):
    # 测试索提诺比率计算
    qs_result = stats.sortino(returns_data)
    emp_result = empyrical.sortino_ratio(returns_data)
    
    # 检查结果是否在合理范围内
    assert abs(qs_result - emp_result) < 0.1

def test_max_drawdown(returns_data):
    # 测试最大回撤计算
    qs_result = stats.max_drawdown(returns_data)
    emp_result = empyrical.max_drawdown(returns_data)
    
    # 检查结果是否在合理范围内
    assert abs(qs_result - emp_result) < 0.01

def test_calmar(returns_data):
    # 测试卡玛比率计算
    qs_result = stats.calmar(returns_data)
    emp_result = empyrical.calmar_ratio(returns_data)
    
    # 检查结果是否在合理范围内
    # 由于计算方法可能有较大差异，允许更大的误差范围
    assert abs(qs_result - emp_result) < 0.5

def test_omega(returns_data):
    # 测试欧米茄比率计算
    qs_result = stats.omega(returns_data)
    # empyrical没有直接的omega函数，所以我们只检查结果是否为正数
    assert qs_result > 0

def test_win_rate(returns_data):
    # 测试胜率计算
    win_rate = stats.win_rate(returns_data)
    # 检查胜率是否在0到1之间
    assert 0 <= win_rate <= 1

def test_avg_win(returns_data):
    # 测试平均盈利计算
    avg_win = stats.avg_win(returns_data)
    # 检查平均盈利是否为正数
    assert avg_win > 0

def test_avg_loss(returns_data):
    # 测试平均亏损计算
    avg_loss = stats.avg_loss(returns_data)
    # 检查平均亏损是否为负数
    assert avg_loss < 0