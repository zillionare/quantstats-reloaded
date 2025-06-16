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

# 测试绩效评估指标
def test_information_ratio(returns_data, benchmark_data):
    # 测试信息比率计算
    ir = stats.information_ratio(returns_data, benchmark_data)
    # 检查信息比率是否为浮点数
    assert isinstance(ir, float)
    # 与empyrical比较
    emp_ir = empyrical.excess_sharpe(returns_data, benchmark_data)
    assert abs(ir - emp_ir) < 0.1

def test_treynor_ratio(returns_data, benchmark_data):
    # 测试特雷诺比率计算
    treynor = stats.treynor_ratio(returns_data, benchmark_data)
    # 检查特雷诺比率是否为浮点数
    assert isinstance(treynor, float)

def test_probabilistic_sharpe_ratio(returns_data):
    # 测试概率夏普比率计算
    prob_sharpe = stats.probabilistic_sharpe_ratio(returns_data)
    # 检查概率夏普比率是否为浮点数
    assert isinstance(prob_sharpe, float)

def test_adjusted_sortino(returns_data):
    # 测试调整后的索提诺比率计算
    adj_sortino = stats.adjusted_sortino(returns_data)
    # 检查调整后的索提诺比率是否为浮点数
    assert isinstance(adj_sortino, float)
    # 调整后的索提诺比率应该与原始索提诺比率有所不同
    original_sortino = stats.sortino(returns_data)
    assert adj_sortino != original_sortino

def test_skew(returns_data):
    # 测试偏度计算
    skew = stats.skew(returns_data)
    # 检查偏度是否为浮点数
    assert isinstance(skew, float)

def test_kurtosis(returns_data):
    # 测试峰度计算
    kurt = stats.kurtosis(returns_data)
    # 检查峰度是否为浮点数
    assert isinstance(kurt, float)

# 测试收益分析
def test_cagr(returns_data):
    # 测试复合年增长率计算
    cagr = stats.cagr(returns_data)
    # 检查CAGR是否为浮点数
    assert isinstance(cagr, float)
    # 与empyrical比较 - 允许更大的误差范围，因为计算方法可能不同
    emp_cagr = empyrical.annual_return(returns_data)
    assert abs(cagr - emp_cagr) < 0.1

def test_best(returns_data):
    # 测试最佳收益计算
    best = stats.best(returns_data)
    # 检查最佳收益是否为浮点数
    assert isinstance(best, float)
    # 最佳收益应该大于0
    assert best > 0

def test_worst(returns_data):
    # 测试最差收益计算
    worst = stats.worst(returns_data)
    # 检查最差收益是否为浮点数
    assert isinstance(worst, float)
    # 最差收益应该小于0
    assert worst < 0

def test_gain_to_pain_ratio(returns_data):
    # 测试收益痛苦比率计算
    gpr = stats.gain_to_pain_ratio(returns_data)
    # 检查收益痛苦比率是否为浮点数
    assert isinstance(gpr, float)

def test_ulcer_index(returns_data):
    # 测试溃疡指数计算
    ui = stats.ulcer_index(returns_data)
    # 检查溃疡指数是否为浮点数
    assert isinstance(ui, float)
    # 溃疡指数应该大于等于0
    assert ui >= 0

def test_ulcer_performance_index(returns_data):
    # 测试溃疡表现指数计算
    upi = stats.ulcer_performance_index(returns_data)
    # 检查溃疡表现指数是否为浮点数
    assert isinstance(upi, float)

# 测试风险调整收益
def test_risk_of_ruin(returns_data):
    # 测试破产风险计算
    ror = stats.risk_of_ruin(returns_data)
    # 检查破产风险是否为浮点数
    assert isinstance(ror, float)
    # 破产风险应该在0到1之间
    assert 0 <= ror <= 1

def test_risk_return_ratio(returns_data):
    # 测试风险收益比率计算
    rrr = stats.risk_return_ratio(returns_data)
    # 检查风险收益比率是否为浮点数
    assert isinstance(rrr, float)

def test_omega(returns_data):
    # 测试欧米茄比率计算
    omega = stats.omega(returns_data)
    # 检查欧米茄比率是否为浮点数
    assert isinstance(omega, float)
    # 欧米茄比率应该大于0
    assert omega > 0

def test_smart_sharpe(returns_data):
    # 测试智能夏普比率计算
    smart_sharpe = stats.smart_sharpe(returns_data)
    # 检查智能夏普比率是否为浮点数
    assert isinstance(smart_sharpe, float)
    # 智能夏普比率应该与原始夏普比率有所不同
    original_sharpe = stats.sharpe(returns_data)
    assert smart_sharpe != original_sharpe

def test_smart_sortino(returns_data):
    # 测试智能索提诺比率计算
    smart_sortino = stats.smart_sortino(returns_data)
    # 检查智能索提诺比率是否为浮点数
    assert isinstance(smart_sortino, float)
    # 智能索提诺比率应该与原始索提诺比率有所不同
    original_sortino = stats.sortino(returns_data)
    assert smart_sortino != original_sortino