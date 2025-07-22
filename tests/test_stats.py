#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import empyrical
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import quantstats as qs


def replay(recorded_file: Path|str|None = None)->Tuple[ NDArray, NDArray, dict, list]:
    # 读取已录制的测试结果
    if recorded_file is None:
        recorded_file = Path(__file__).parent / "assets/quantstats-recorded.pkl"
    else:
        recorded_file = Path(recorded_file)

    if not recorded_file.exists():
        raise FileNotFoundError(f"{recorded_file}不存在")

    with open(recorded_file, "rb") as f:
        data = pickle.load(f)
        expected = data["results"]
        returns = data["returns"]
        benchmark = data["benchmark"]
        unary_ops = data["unary_ops"]

        return returns, benchmark, expected, unary_ops


def record(dst: str):
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')

    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

    np.random.seed(43)
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

    # the following requires returns only
    unary_ops = ['adjusted_sortino', 'autocorr_penalty', 'avg_loss', 'avg_return', 'avg_win', 'best', 'cagr', 'calmar', 'common_sense_ratio', 'comp', 'compsum', 'conditional_value_at_risk', 'consecutive_losses', 'consecutive_wins', 'cpc_index', 'cvar', 'distribution', 'drawdown_details', 'expected_return', 'expected_shortfall', 'exposure', 'gain_to_pain_ratio', 'geometric_mean', 'ghpr', 'implied_volatility', 'kelly_criterion', 'kurtosis', 'max_drawdown', 'monthly_returns', 'omega', 'outlier_loss_ratio', 'outlier_win_ratio', 'outliers', 'payoff_ratio', 'pct_rank', 'probabilistic_adjusted_sortino_ratio', 'probabilistic_ratio', 'probabilistic_sharpe_ratio', 'probabilistic_sortino_ratio', 'profit_factor', 'profit_ratio', 'rar', 'recovery_factor', 'remove_outliers', 'risk_of_ruin', 'risk_return_ratio', 'rolling_sharpe', 'rolling_sortino', 'rolling_volatility', 'ror', 'serenity_index', 'sharpe', 'skew', 'smart_sharpe', 'smart_sortino', 'sortino', 'tail_ratio', 'to_drawdown_series', 'ulcer_index', 'ulcer_performance_index', 'upi', 'validate_input', 'value_at_risk', 'var', 'volatility', 'win_loss_ratio', 'win_rate', 'worst']

    # the following requires benchmark also
    binary_ops = [
        "compare",
        "greeks",
        "information_ratio",
        "r2",
        "r_squared",
        "treynor_ratio"
    ]

    # need speical handling, or not a stats
    excluded = ["rolling_greeks", "safe_concat"]
    results = {}
    for name in dir(qs.stats):
        if name[0] == '_':
            continue
            
        func = getattr(qs.stats, name)
        if name in unary_ops:
            results[name] = func(returns)
        elif name in binary_ops:
            results[name] = func(returns, benchmark_returns)
        else:
            print("no handled", name)

    data= {
        "returns": returns,
        "benchmark": benchmark_returns,
        "results": results,
        "unary_ops": unary_ops
    }

    with open(dst, "wb") as f:
        pickle.dump(data, f)

def is_equal(a, b, precision = 7):
    """
    判断两个对象是否相等，支持标量、可迭代对象和字典
    
    参数:
        a: 第一个比较对象
        b: 第二个比较对象
    
    返回:
        bool: 两个对象是否相等
    """
    # 处理numpy标量的特殊情况
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    
    # 类型不同直接返回False
    if type(a) != type(b):
        return False
    
    # 1. 字典类型比较
    if isinstance(a, dict):
        # 键的数量不同则不相等
        if len(a) != len(b):
            return False
        # 比较每个键值对
        for key in a:
            if key not in b:
                return False
            if not is_equal(a[key], b[key]):
                return False
        return True
    
    # 2. 可迭代对象比较（排除字符串，字符串当作标量处理）
    if isinstance(a, Iterable) and not isinstance(a, (str, bytes)):
        # 转换为列表进行长度和元素比较
        list_a = list(a)
        list_b = list(b)
        if len(list_a) != len(list_b):
            return False
        # 递归比较每个元素
        for item_a, item_b in zip(list_a, list_b):
            if not is_equal(item_a, item_b):
                return False
        return True
    
    if isinstance(a, float) and isinstance(b, float):
        return np.isclose(a, b, atol=10**(-precision))

    # 其他标量直接比较
    return a == b

def test_recorded():
    returns, benchmark, expected, unary_ops = replay()

    for name in expected.keys():
        try:
            func = getattr(qs.stats, name)
            if name in unary_ops:
                ret = func(returns)
                expected_ret = expected.get(name)
                msg = f"{name}结果不一致，期望{expected_ret}，实际{ret}"
                assert is_equal(ret, expected_ret), msg
            else:
                ret = func(returns, benchmark)
                expected_ret = expected.get(name, None)
                msg = f"{name}结果不一致，期望{expected_ret}，实际{ret}"
                assert is_equal(ret, expected_ret), msg
        except Exception as e:
            print(name, e)
