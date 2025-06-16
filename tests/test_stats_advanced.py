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

# 测试风险指标
def test_value_at_risk(returns_data):
    # 测试风险价值(VaR)计算
    var = stats.value_at_risk(returns_data)
    # 检查VaR是否为负数（损失）
    assert var < 0
    # 检查不同置信水平的VaR
    var_95 = stats.value_at_risk(returns_data, sigma=1.96)  # 95%置信水平
    var_99 = stats.value_at_risk(returns_data, sigma=2.58)  # 99%置信水平
    # 99%置信水平的VaR应该比95%的更大（更负）
    assert var_99 < var_95

def test_conditional_value_at_risk(returns_data):
    # 测试条件风险价值(CVaR)计算
    cvar = stats.conditional_value_at_risk(returns_data)
    # 检查CVaR是否为负数（损失）
    assert cvar < 0
    # CVaR应该比VaR更大（更负）
    var = stats.value_at_risk(returns_data)
    assert cvar < var

def test_tail_ratio(returns_data):
    # 测试尾部比率计算
    tail_ratio = stats.tail_ratio(returns_data)
    # 尾部比率应该为正数
    assert tail_ratio > 0

def test_outliers(returns_data):
    # 测试异常值检测
    outliers = stats.outliers(returns_data)
    # 检查异常值是否为Series
    assert isinstance(outliers, pd.Series)
    # 异常值数量应该小于原始数据
    assert len(outliers) < len(returns_data)

# 测试回撤分析
def test_drawdown_details(returns_data):
    # 测试回撤详情计算
    dd_details = stats.drawdown_details(returns_data)
    # 检查返回的是否为DataFrame
    assert isinstance(dd_details, pd.DataFrame)
    # 检查是否包含必要的列
    assert all(col in dd_details.columns for col in ['start', 'valley', 'end', 'days', 'max drawdown'])

def test_to_drawdown_series(returns_data):
    # 测试转换为回撤序列
    dd_series = stats.to_drawdown_series(returns_data)
    # 检查返回的是否为Series
    assert isinstance(dd_series, pd.Series)
    # 回撤值应该小于等于0
    assert (dd_series <= 0).all()

def test_kelly_criterion(returns_data):
    # 测试凯利准则计算
    kelly = stats.kelly_criterion(returns_data)
    # 检查凯利值是否在合理范围内
    assert -1 <= kelly <= 1

# 测试相关性分析
def test_correlation(returns_data, benchmark_data):
    # 测试相关性计算
    # 使用pandas的corr函数计算相关性，因为quantstats没有单独的correlation函数
    corr = returns_data.corr(benchmark_data)
    # 相关系数应该在-1到1之间
    assert -1 <= corr <= 1

def test_beta(returns_data, benchmark_data):
    # 测试贝塔系数计算
    # 使用greeks函数获取beta值
    beta = stats.greeks(returns_data, benchmark_data)['beta']
    # 检查贝塔值是否为浮点数
    assert isinstance(beta, float)
    # 与empyrical比较
    emp_beta = empyrical.beta(returns_data, benchmark_data)
    assert abs(beta - emp_beta) < 0.01

def test_alpha(returns_data, benchmark_data):
    # 测试阿尔法系数计算
    # 使用greeks函数获取alpha值
    alpha = stats.greeks(returns_data, benchmark_data)['alpha']
    # 检查阿尔法值是否为浮点数
    assert isinstance(alpha, float)
    # 与empyrical比较
    emp_alpha = empyrical.alpha(returns_data, benchmark_data)
    # 由于计算方法可能不同，允许一定误差
    assert abs(alpha - emp_alpha) < 0.1

# 测试时间序列分析
def test_consecutive_wins(returns_data):
    # 测试连续盈利计算
    cons_wins = stats.consecutive_wins(returns_data)
    # 检查连续盈利是否为整数
    assert isinstance(cons_wins, (int, np.integer))
    # 连续盈利应该大于等于0
    assert cons_wins >= 0

def test_consecutive_losses(returns_data):
    # 测试连续亏损计算
    cons_losses = stats.consecutive_losses(returns_data)
    # 检查连续亏损是否为整数
    assert isinstance(cons_losses, (int, np.integer))
    # 连续亏损应该大于等于0
    assert cons_losses >= 0

def test_exposure(returns_data):
    # 测试市场暴露度计算
    exposure = stats.exposure(returns_data)
    # 暴露度应该在0到1之间
    assert 0 <= exposure <= 1