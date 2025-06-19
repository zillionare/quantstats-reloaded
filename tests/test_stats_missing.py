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

@pytest.fixture
def price_data():
    # 创建价格数据用于测试
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    np.random.seed(44)
    prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates)
    return prices

# 测试未覆盖的函数

def test_pct_rank(returns_data):
    """测试百分位排名"""
    # 转换为价格数据进行测试
    prices = (1 + returns_data).cumprod() * 100
    prices.name = 'price'  # 给Series一个名字

    # 测试默认参数
    rank = stats.pct_rank(prices)
    assert isinstance(rank, pd.Series)
    assert len(rank) == len(prices)
    # 检查非NaN值的范围
    valid_rank = rank.dropna()
    if len(valid_rank) > 0:
        assert (valid_rank >= 0).all() and (valid_rank <= 100).all()

    # 测试不同窗口大小
    rank_30 = stats.pct_rank(prices, window=30)
    assert isinstance(rank_30, pd.Series)

    # 测试窗口大小为1的情况
    rank_1 = stats.pct_rank(prices, window=1)
    assert isinstance(rank_1, pd.Series)

def test_ghpr(returns_data):
    """测试几何持有期收益率"""
    ghpr = stats.ghpr(returns_data)
    assert isinstance(ghpr, float)
    
    # GHPR应该等于geometric_mean
    geom_mean = stats.geometric_mean(returns_data)
    assert abs(ghpr - geom_mean) < 1e-10

def test_avg_return(returns_data):
    """测试平均收益率"""
    avg_ret = stats.avg_return(returns_data)
    assert isinstance(avg_ret, float)
    
    # 测试不同聚合方式
    avg_ret_monthly = stats.avg_return(returns_data, aggregate='M')
    assert isinstance(avg_ret_monthly, float)
    
    # 测试复合和非复合模式
    avg_ret_simple = stats.avg_return(returns_data, compounded=False)
    assert isinstance(avg_ret_simple, float)

def test_cvar(returns_data):
    """测试条件风险价值（CVaR的简写）"""
    cvar = stats.cvar(returns_data)
    cvar_full = stats.conditional_value_at_risk(returns_data)
    
    # cvar应该等于conditional_value_at_risk
    assert abs(cvar - cvar_full) < 1e-10

def test_expected_shortfall(returns_data):
    """测试期望损失"""
    es = stats.expected_shortfall(returns_data)
    cvar = stats.conditional_value_at_risk(returns_data)
    
    # expected_shortfall应该等于conditional_value_at_risk
    assert abs(es - cvar) < 1e-10

def test_var(returns_data):
    """测试风险价值（VaR的简写）"""
    var = stats.var(returns_data)
    var_full = stats.value_at_risk(returns_data)
    
    # var应该等于value_at_risk
    assert abs(var - var_full) < 1e-10

def test_rar(returns_data):
    """测试风险调整收益率"""
    rar = stats.rar(returns_data)
    assert isinstance(rar, float)
    
    # 测试不同的风险自由利率
    rar_rf = stats.rar(returns_data, rf=0.02)
    assert isinstance(rar_rf, float)

def test_upi(returns_data):
    """测试Ulcer Performance Index（UPI的简写）"""
    upi = stats.upi(returns_data)
    upi_full = stats.ulcer_performance_index(returns_data)
    
    # upi应该等于ulcer_performance_index
    assert abs(upi - upi_full) < 1e-10

def test_ror(returns_data):
    """测试破产风险（RoR的简写）"""
    ror = stats.ror(returns_data)
    ror_full = stats.risk_of_ruin(returns_data)
    
    # ror应该等于risk_of_ruin
    assert abs(ror - ror_full) < 1e-10

def test_r2(returns_data, benchmark_data):
    """测试R平方（r2的简写）"""
    r2 = stats.r2(returns_data, benchmark_data)
    r2_full = stats.r_squared(returns_data, benchmark_data)
    
    # r2应该等于r_squared
    assert abs(r2 - r2_full) < 1e-10

def test_r_squared(returns_data, benchmark_data):
    """测试R平方"""
    r2 = stats.r_squared(returns_data, benchmark_data)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1  # R平方应该在0到1之间

def test_payoff_ratio(returns_data):
    """测试收益比率"""
    payoff = stats.payoff_ratio(returns_data)
    assert isinstance(payoff, float)
    assert payoff > 0  # 收益比率应该为正数

def test_win_loss_ratio(returns_data):
    """测试盈亏比"""
    wl_ratio = stats.win_loss_ratio(returns_data)
    assert isinstance(wl_ratio, float)
    assert wl_ratio > 0  # 盈亏比应该为正数

def test_profit_ratio(returns_data):
    """测试利润比率"""
    profit_ratio = stats.profit_ratio(returns_data)
    assert isinstance(profit_ratio, float)

def test_profit_factor(returns_data):
    """测试利润因子"""
    pf = stats.profit_factor(returns_data)
    assert isinstance(pf, float)
    assert pf > 0  # 利润因子应该为正数

def test_cpc_index(returns_data):
    """测试CPC指数"""
    cpc = stats.cpc_index(returns_data)
    assert isinstance(cpc, float)
    assert cpc > 0  # CPC指数应该为正数

def test_common_sense_ratio(returns_data):
    """测试常识比率"""
    csr = stats.common_sense_ratio(returns_data)
    assert isinstance(csr, float)
    assert csr > 0  # 常识比率应该为正数

def test_outlier_win_ratio(returns_data):
    """测试异常值胜率"""
    owr = stats.outlier_win_ratio(returns_data)
    assert isinstance(owr, float)
    assert owr >= 0  # 异常值胜率应该为非负数

def test_outlier_loss_ratio(returns_data):
    """测试异常值败率"""
    olr = stats.outlier_loss_ratio(returns_data)
    assert isinstance(olr, float)
    assert olr >= 0  # 异常值败率应该为非负数

def test_recovery_factor(returns_data):
    """测试恢复因子"""
    rf = stats.recovery_factor(returns_data)
    assert isinstance(rf, float)

def test_serenity_index(returns_data):
    """测试宁静指数"""
    si = stats.serenity_index(returns_data)
    assert isinstance(si, float)

def test_autocorr_penalty(returns_data):
    """测试自相关惩罚"""
    penalty = stats.autocorr_penalty(returns_data)
    assert isinstance(penalty, float)
    assert penalty >= 0  # 惩罚应该为非负数

def test_implied_volatility(returns_data):
    """测试隐含波动率"""
    iv = stats.implied_volatility(returns_data)
    # implied_volatility返回的是Series，不是float
    assert isinstance(iv, pd.Series)
    assert len(iv) == len(returns_data)

    # 测试不同参数
    iv_30 = stats.implied_volatility(returns_data, periods=30)
    assert isinstance(iv_30, pd.Series)

def test_rolling_greeks(returns_data, benchmark_data):
    """测试滚动希腊字母"""
    rolling_greeks = stats.rolling_greeks(returns_data, benchmark_data)
    assert isinstance(rolling_greeks, pd.DataFrame)
    assert 'alpha' in rolling_greeks.columns
    assert 'beta' in rolling_greeks.columns

    # 测试不同窗口大小（参数名是periods，不是window）
    rolling_greeks_30 = stats.rolling_greeks(returns_data, benchmark_data, periods=30)
    assert isinstance(rolling_greeks_30, pd.DataFrame)

def test_compare(returns_data, benchmark_data):
    """测试比较函数"""
    comparison = stats.compare(returns_data, benchmark_data)
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison.columns) == 4  # 应该有4列数据：Benchmark, Returns, Multiplier, Won

def test_probabilistic_ratio(returns_data):
    """测试概率比率"""
    # 测试基于Sharpe的概率比率
    prob_sharpe = stats.probabilistic_ratio(returns_data, base='sharpe')
    assert isinstance(prob_sharpe, float)
    
    # 测试基于Sortino的概率比率
    prob_sortino = stats.probabilistic_ratio(returns_data, base='sortino')
    assert isinstance(prob_sortino, float)
    
    # 测试基于调整Sortino的概率比率
    prob_adj_sortino = stats.probabilistic_ratio(returns_data, base='adjusted_sortino')
    assert isinstance(prob_adj_sortino, float)

def test_probabilistic_sortino_ratio(returns_data):
    """测试概率Sortino比率"""
    prob_sortino = stats.probabilistic_sortino_ratio(returns_data)
    assert isinstance(prob_sortino, float)

def test_probabilistic_adjusted_sortino_ratio(returns_data):
    """测试概率调整Sortino比率"""
    prob_adj_sortino = stats.probabilistic_adjusted_sortino_ratio(returns_data)
    assert isinstance(prob_adj_sortino, float)
