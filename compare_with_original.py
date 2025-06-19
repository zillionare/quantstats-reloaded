#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
对比当前实现与原版quantstats 0.0.64的结果
"""

import pandas as pd
import numpy as np
import sys
import os

# 创建测试数据
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
benchmark = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

print("=== 对比测试：当前实现 vs 原版quantstats 0.0.64 ===\n")

# 导入原版quantstats
try:
    import quantstats as qs_original
    print(f"原版quantstats版本: {qs_original.__version__}")
except ImportError:
    print("无法导入原版quantstats")
    sys.exit(1)

# 导入当前实现
sys.path.insert(0, '/mnt/persist/workspace')
try:
    import quantstats as qs_current
    print(f"当前实现版本: {getattr(qs_current, '__version__', 'unknown')}")
except ImportError:
    print("无法导入当前实现")
    sys.exit(1)

print("\n" + "="*60)

def compare_function(func_name, *args, **kwargs):
    """比较两个版本的函数结果"""
    try:
        # 获取原版函数
        original_func = getattr(qs_original.stats, func_name, None)
        if original_func is None:
            original_func = getattr(qs_original.utils, func_name, None)
        if original_func is None:
            original_func = getattr(qs_original, func_name, None)
            
        # 获取当前版本函数
        current_func = getattr(qs_current.stats, func_name, None)
        if current_func is None:
            current_func = getattr(qs_current.utils, func_name, None)
        if current_func is None:
            current_func = getattr(qs_current, func_name, None)
            
        if original_func is None or current_func is None:
            print(f"❌ {func_name}: 函数不存在")
            return False
            
        # 执行函数
        try:
            original_result = original_func(*args, **kwargs)
        except Exception as e:
            print(f"❌ {func_name}: 原版执行失败 - {e}")
            return False
            
        try:
            current_result = current_func(*args, **kwargs)
        except Exception as e:
            print(f"❌ {func_name}: 当前版本执行失败 - {e}")
            return False
        
        # 比较结果
        if isinstance(original_result, (pd.Series, pd.DataFrame)):
            if isinstance(current_result, (pd.Series, pd.DataFrame)):
                try:
                    if isinstance(original_result, pd.Series):
                        pd.testing.assert_series_equal(original_result, current_result, rtol=1e-10, atol=1e-10)
                    else:
                        pd.testing.assert_frame_equal(original_result, current_result, rtol=1e-10, atol=1e-10)
                    print(f"✅ {func_name}: 结果完全一致")
                    return True
                except AssertionError as e:
                    print(f"⚠️  {func_name}: 结果有差异")
                    print(f"   原版类型: {type(original_result)}, 形状: {getattr(original_result, 'shape', 'N/A')}")
                    print(f"   当前类型: {type(current_result)}, 形状: {getattr(current_result, 'shape', 'N/A')}")
                    if hasattr(original_result, 'head'):
                        print(f"   原版前几个值: {original_result.head()}")
                        print(f"   当前前几个值: {current_result.head()}")
                    return False
            else:
                print(f"❌ {func_name}: 返回类型不匹配 - 原版: {type(original_result)}, 当前: {type(current_result)}")
                return False
        elif isinstance(original_result, (int, float, np.number)):
            if isinstance(current_result, (int, float, np.number)):
                if np.isclose(original_result, current_result, rtol=1e-10, atol=1e-10):
                    print(f"✅ {func_name}: 结果完全一致 ({original_result})")
                    return True
                else:
                    print(f"⚠️  {func_name}: 数值有差异 - 原版: {original_result}, 当前: {current_result}")
                    return False
            else:
                print(f"❌ {func_name}: 返回类型不匹配 - 原版: {type(original_result)}, 当前: {type(current_result)}")
                return False
        else:
            # 其他类型的比较
            if original_result == current_result:
                print(f"✅ {func_name}: 结果完全一致")
                return True
            else:
                print(f"⚠️  {func_name}: 结果有差异")
                print(f"   原版: {original_result}")
                print(f"   当前: {current_result}")
                return False
                
    except Exception as e:
        print(f"❌ {func_name}: 比较过程出错 - {e}")
        return False

# 测试主要的统计函数
print("\n=== 统计函数对比 ===")
stats_tests = [
    ('sharpe', returns),
    ('sortino', returns),
    ('max_drawdown', returns),
    ('volatility', returns),
    ('cagr', returns),
    ('calmar', returns),
    ('omega', returns),
    ('skew', returns),
    ('kurtosis', returns),
    ('value_at_risk', returns),
    ('conditional_value_at_risk', returns),
    ('tail_ratio', returns),
    ('payoff_ratio', returns),
    ('profit_factor', returns),
    ('win_rate', returns),
    ('avg_win', returns),
    ('avg_loss', returns),
    ('best', returns),
    ('worst', returns),
    ('kelly_criterion', returns),
    ('risk_of_ruin', returns),
    ('ulcer_index', returns),
    ('serenity_index', returns),
    # 添加更多函数测试
    ('information_ratio', returns, benchmark),
    ('treynor_ratio', returns, benchmark),
    ('r_squared', returns, benchmark),
    ('greeks', returns, benchmark),
    ('gain_to_pain_ratio', returns),
    ('common_sense_ratio', returns),
    ('cpc_index', returns),
    ('outlier_win_ratio', returns),
    ('outlier_loss_ratio', returns),
    ('recovery_factor', returns),
    ('ulcer_performance_index', returns),
    ('probabilistic_sharpe_ratio', returns),
    ('expected_return', returns),
    ('geometric_mean', returns),
    ('ghpr', returns),
    ('autocorr_penalty', returns),
]

passed = 0
total = 0

for test in stats_tests:
    func_name = test[0]
    args = test[1:]
    total += 1
    if compare_function(func_name, *args):
        passed += 1

print(f"\n统计函数测试结果: {passed}/{total} 通过")

# 测试工具函数
print("\n=== 工具函数对比 ===")
utils_tests = [
    ('to_returns', (1 + returns).cumprod() * 100),
    ('to_prices', returns),
    ('aggregate_returns', returns, 'M'),
    ('make_portfolio', returns),
    ('to_log_returns', returns),
    ('rebase', (1 + returns).cumprod() * 100),
    ('to_excess_returns', returns, 0.02),
    ('exponential_stdev', returns),
]

utils_passed = 0
utils_total = 0

for test in utils_tests:
    func_name = test[0]
    args = test[1:]
    utils_total += 1
    if compare_function(func_name, *args):
        utils_passed += 1

print(f"\n工具函数测试结果: {utils_passed}/{utils_total} 通过")

# 总结
total_all = total + utils_total
passed_all = passed + utils_passed
print(f"\n" + "="*60)
print(f"总体测试结果: {passed_all}/{total_all} 通过 ({passed_all/total_all*100:.1f}%)")

if passed_all == total_all:
    print("🎉 所有测试都通过！当前实现与原版完全一致。")
elif passed_all / total_all >= 0.9:
    print("✅ 大部分测试通过，实现基本正确，有少量差异。")
else:
    print("⚠️  有较多差异，需要进一步检查和修复。")
