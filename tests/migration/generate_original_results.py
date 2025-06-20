#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
在原版quantstats环境中生成测试结果并保存到文件
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# 确保不导入当前目录的quantstats
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

# 设置随机种子确保可重现性
np.random.seed(42)

# 创建测试数据
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
benchmark = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

print("=== 在原版QuantStats环境中生成测试结果 ===")

# 导入原版quantstats
import quantstats as qs

print(f"QuantStats版本: {qs.__version__}")

# 存储测试结果的字典
results = {
    'version': qs.__version__,
    'test_data_info': {
        'returns_shape': returns.shape,
        'returns_mean': float(returns.mean()),
        'returns_std': float(returns.std()),
        'benchmark_shape': benchmark.shape,
        'benchmark_mean': float(benchmark.mean()),
        'benchmark_std': float(benchmark.std()),
        'date_range': [str(returns.index[0]), str(returns.index[-1])]
    },
    'stats_functions': {},
    'utils_functions': {},
    'edge_cases': {}
}

# 测试统计函数
print("\n测试统计函数...")
stats_functions = [
    'sharpe', 'sortino', 'max_drawdown', 'volatility', 'cagr', 'calmar', 
    'omega', 'skew', 'kurtosis', 'value_at_risk', 'conditional_value_at_risk',
    'tail_ratio', 'payoff_ratio', 'profit_factor', 'win_rate', 'avg_win',
    'avg_loss', 'best', 'worst', 'kelly_criterion', 'risk_of_ruin',
    'ulcer_index', 'serenity_index', 'expected_return', 'geometric_mean',
    'gain_to_pain_ratio', 'common_sense_ratio', 'cpc_index', 'outlier_win_ratio',
    'outlier_loss_ratio', 'recovery_factor', 'ulcer_performance_index',
    'probabilistic_sharpe_ratio', 'autocorr_penalty'
]

for func_name in stats_functions:
    try:
        func = getattr(qs.stats, func_name)
        result = func(returns)
        
        # 处理不同类型的结果
        if isinstance(result, (pd.Series, pd.DataFrame)):
            # 转换为可序列化的格式
            if isinstance(result, pd.Series):
                # 处理Series的索引
                if hasattr(result.index, 'strftime'):
                    # 日期索引
                    index_data = [d.strftime('%Y-%m-%d') for d in result.index]
                else:
                    index_data = [str(i) for i in result.index]

                data_dict = {}
                for i, (idx, val) in enumerate(zip(index_data, result.values)):
                    if pd.isna(val):
                        data_dict[idx] = 'NaN'
                    elif np.isinf(val):
                        data_dict[idx] = 'inf' if val > 0 else '-inf'
                    else:
                        data_dict[idx] = float(val)

                results['stats_functions'][func_name] = {
                    'type': 'pandas_series',
                    'data': data_dict,
                    'shape': result.shape
                }
            else:
                # DataFrame处理
                results['stats_functions'][func_name] = {
                    'type': 'pandas_dataframe',
                    'data': result.to_dict(),
                    'shape': result.shape
                }
        else:
            # 标量结果
            if pd.isna(result):
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': 'NaN'}
            elif np.isinf(result):
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': 'inf' if result > 0 else '-inf'}
            else:
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': float(result)}
        
        print(f"  ✅ {func_name}: {type(result)}")
    except Exception as e:
        print(f"  ❌ {func_name}: {e}")
        results['stats_functions'][func_name] = {'type': 'error', 'error': str(e)}

# 测试需要基准的函数
print("\n测试需要基准的统计函数...")
benchmark_functions = ['information_ratio', 'treynor_ratio', 'r_squared', 'greeks']

for func_name in benchmark_functions:
    try:
        func = getattr(qs.stats, func_name)
        result = func(returns, benchmark)
        
        if isinstance(result, dict):
            # 处理字典结果（如greeks）
            processed_dict = {}
            for k, v in result.items():
                if pd.isna(v):
                    processed_dict[k] = 'NaN'
                elif np.isinf(v):
                    processed_dict[k] = 'inf' if v > 0 else '-inf'
                else:
                    processed_dict[k] = float(v)
            results['stats_functions'][func_name] = {'type': 'dict', 'value': processed_dict}
        elif isinstance(result, (pd.Series, pd.DataFrame)):
            if isinstance(result, pd.Series):
                # 处理Series
                if hasattr(result.index, 'strftime'):
                    index_data = [d.strftime('%Y-%m-%d') for d in result.index]
                else:
                    index_data = [str(i) for i in result.index]

                data_dict = {}
                for i, (idx, val) in enumerate(zip(index_data, result.values)):
                    if pd.isna(val):
                        data_dict[idx] = 'NaN'
                    elif np.isinf(val):
                        data_dict[idx] = 'inf' if val > 0 else '-inf'
                    else:
                        data_dict[idx] = float(val)

                results['stats_functions'][func_name] = {
                    'type': 'pandas_series',
                    'data': data_dict,
                    'shape': result.shape
                }
            else:
                results['stats_functions'][func_name] = {
                    'type': 'pandas_dataframe',
                    'data': result.to_dict(),
                    'shape': result.shape
                }
        else:
            if pd.isna(result):
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': 'NaN'}
            elif np.isinf(result):
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': 'inf' if result > 0 else '-inf'}
            else:
                results['stats_functions'][func_name] = {'type': 'scalar', 'value': float(result)}
        
        print(f"  ✅ {func_name}: {type(result)}")
    except Exception as e:
        print(f"  ❌ {func_name}: {e}")
        results['stats_functions'][func_name] = {'type': 'error', 'error': str(e)}

# 测试工具函数
print("\n测试工具函数...")
# 创建价格数据用于测试
prices = (1 + returns).cumprod() * 100

utils_tests = [
    ('to_returns', [prices]),
    ('to_prices', [returns]),
    ('aggregate_returns', [returns, 'M']),
    ('make_portfolio', [returns]),
    ('to_log_returns', [returns]),
    ('rebase', [prices]),
    ('to_excess_returns', [returns, 0.02]),
    ('exponential_stdev', [returns])
]

for func_name, args in utils_tests:
    try:
        func = getattr(qs.utils, func_name)
        result = func(*args)
        
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if isinstance(result, pd.Series):
                # 处理Series
                if hasattr(result.index, 'strftime'):
                    index_data = [d.strftime('%Y-%m-%d') for d in result.index]
                else:
                    index_data = [str(i) for i in result.index]

                data_dict = {}
                for i, (idx, val) in enumerate(zip(index_data, result.values)):
                    if pd.isna(val):
                        data_dict[idx] = 'NaN'
                    elif np.isinf(val):
                        data_dict[idx] = 'inf' if val > 0 else '-inf'
                    else:
                        data_dict[idx] = float(val)

                results['utils_functions'][func_name] = {
                    'type': 'pandas_series',
                    'data': data_dict,
                    'shape': result.shape
                }
            else:
                results['utils_functions'][func_name] = {
                    'type': 'pandas_dataframe',
                    'data': result.to_dict(),
                    'shape': result.shape
                }
        else:
            if pd.isna(result):
                results['utils_functions'][func_name] = {'type': 'scalar', 'value': 'NaN'}
            elif np.isinf(result):
                results['utils_functions'][func_name] = {'type': 'scalar', 'value': 'inf' if result > 0 else '-inf'}
            else:
                results['utils_functions'][func_name] = {'type': 'scalar', 'value': float(result)}
        
        print(f"  ✅ {func_name}: {type(result)}")
    except Exception as e:
        print(f"  ❌ {func_name}: {e}")
        results['utils_functions'][func_name] = {'type': 'error', 'error': str(e)}

# 测试边界条件
print("\n测试边界条件...")
edge_cases = {
    'zero_returns': pd.Series([0.0] * 100, index=pd.date_range('2020-01-01', periods=100)),
    'positive_returns': pd.Series([0.01] * 100, index=pd.date_range('2020-01-01', periods=100)),
    'negative_returns': pd.Series([-0.01] * 100, index=pd.date_range('2020-01-01', periods=100)),
    'extreme_returns': pd.Series([0.5, -0.5] * 50, index=pd.date_range('2020-01-01', periods=100)),
    'single_return': pd.Series([0.01], index=pd.date_range('2020-01-01', periods=1))
}

edge_test_functions = ['sharpe', 'volatility', 'max_drawdown', 'sortino', 'win_rate', 'skew', 'kurtosis']

for case_name, test_data in edge_cases.items():
    results['edge_cases'][case_name] = {}
    print(f"\n  测试 {case_name}:")
    
    for func_name in edge_test_functions:
        try:
            func = getattr(qs.stats, func_name)
            result = func(test_data)
            
            if pd.isna(result):
                results['edge_cases'][case_name][func_name] = {'type': 'scalar', 'value': 'NaN'}
            elif np.isinf(result):
                results['edge_cases'][case_name][func_name] = {'type': 'scalar', 'value': 'inf' if result > 0 else '-inf'}
            else:
                results['edge_cases'][case_name][func_name] = {'type': 'scalar', 'value': float(result)}
            
            print(f"    ✅ {func_name}: {result}")
        except Exception as e:
            print(f"    ❌ {func_name}: {e}")
            results['edge_cases'][case_name][func_name] = {'type': 'error', 'error': str(e)}

# 保存结果
print("\n保存测试结果...")
with open('/tmp/original_quantstats_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# 同时保存原始数据用于验证
test_data = {
    'returns': returns,
    'benchmark': benchmark,
    'prices': prices
}

with open('/tmp/original_test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print(f"\n✅ 测试结果已保存到:")
print(f"  - /tmp/original_quantstats_results.json")
print(f"  - /tmp/original_test_data.pkl")
print(f"\n统计函数测试: {len([k for k, v in results['stats_functions'].items() if v.get('type') != 'error'])}/{len(results['stats_functions'])}")
print(f"工具函数测试: {len([k for k, v in results['utils_functions'].items() if v.get('type') != 'error'])}/{len(results['utils_functions'])}")
print(f"边界条件测试: {sum(len([f for f, r in case.items() if r.get('type') != 'error']) for case in results['edge_cases'].values())}")
