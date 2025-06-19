#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
测试边界条件和特殊情况的对比
"""

import pandas as pd
import numpy as np
import sys

# 导入原版和当前版本
import quantstats as qs_original
sys.path.insert(0, '/mnt/persist/workspace')
import quantstats as qs_current

print("=== 边界条件和特殊情况对比测试 ===\n")

def compare_edge_case(description, func_name, *args, **kwargs):
    """比较边界条件"""
    print(f"测试: {description}")
    
    try:
        # 获取函数
        original_func = getattr(qs_original.stats, func_name, None)
        if original_func is None:
            original_func = getattr(qs_original.utils, func_name, None)
        if original_func is None:
            original_func = getattr(qs_original, func_name, None)
            
        current_func = getattr(qs_current.stats, func_name, None)
        if current_func is None:
            current_func = getattr(qs_current.utils, func_name, None)
        if current_func is None:
            current_func = getattr(qs_current, func_name, None)
        
        if original_func is None or current_func is None:
            print(f"  ❌ 函数不存在")
            return False
        
        # 执行测试
        try:
            original_result = original_func(*args, **kwargs)
            current_result = current_func(*args, **kwargs)
            
            # 比较结果
            if isinstance(original_result, (pd.Series, pd.DataFrame)):
                if isinstance(current_result, (pd.Series, pd.DataFrame)):
                    try:
                        if isinstance(original_result, pd.Series):
                            pd.testing.assert_series_equal(original_result, current_result, rtol=1e-10, atol=1e-10)
                        else:
                            pd.testing.assert_frame_equal(original_result, current_result, rtol=1e-10, atol=1e-10)
                        print(f"  ✅ 结果一致")
                        return True
                    except AssertionError:
                        print(f"  ⚠️  结果有差异")
                        return False
                else:
                    print(f"  ❌ 返回类型不匹配")
                    return False
            elif isinstance(original_result, (int, float, np.number)):
                if isinstance(current_result, (int, float, np.number)):
                    if np.isnan(original_result) and np.isnan(current_result):
                        print(f"  ✅ 都返回NaN")
                        return True
                    elif np.isinf(original_result) and np.isinf(current_result):
                        print(f"  ✅ 都返回无穷大")
                        return True
                    elif np.isclose(original_result, current_result, rtol=1e-10, atol=1e-10):
                        print(f"  ✅ 结果一致 ({original_result})")
                        return True
                    else:
                        print(f"  ⚠️  数值有差异 - 原版: {original_result}, 当前: {current_result}")
                        return False
                else:
                    print(f"  ❌ 返回类型不匹配")
                    return False
            else:
                if original_result == current_result:
                    print(f"  ✅ 结果一致")
                    return True
                else:
                    print(f"  ⚠️  结果有差异")
                    return False
                    
        except Exception as e:
            print(f"  ❌ 执行出错: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试出错: {e}")
        return False

# 创建各种边界条件的测试数据
print("创建测试数据...")

# 1. 全零收益
zero_returns = pd.Series([0.0] * 100, index=pd.date_range('2020-01-01', periods=100))

# 2. 全正收益
positive_returns = pd.Series([0.01] * 100, index=pd.date_range('2020-01-01', periods=100))

# 3. 全负收益
negative_returns = pd.Series([-0.01] * 100, index=pd.date_range('2020-01-01', periods=100))

# 4. 极小的收益
tiny_returns = pd.Series([1e-10] * 100, index=pd.date_range('2020-01-01', periods=100))

# 5. 包含NaN的收益
nan_returns = pd.Series([0.01, 0.02, np.nan, 0.01, -0.01] * 20, index=pd.date_range('2020-01-01', periods=100))

# 6. 包含无穷大的收益
inf_returns = pd.Series([0.01, 0.02, np.inf, 0.01, -0.01] * 20, index=pd.date_range('2020-01-01', periods=100))

# 7. 极端波动的收益
extreme_returns = pd.Series([0.5, -0.5] * 50, index=pd.date_range('2020-01-01', periods=100))

# 8. 单个值
single_return = pd.Series([0.01], index=pd.date_range('2020-01-01', periods=1))

# 9. 空Series
empty_returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

print("\n开始边界条件测试...\n")

test_cases = [
    ("全零收益 - Sharpe比率", "sharpe", zero_returns),
    ("全零收益 - 波动率", "volatility", zero_returns),
    ("全零收益 - 最大回撤", "max_drawdown", zero_returns),
    
    ("全正收益 - Sharpe比率", "sharpe", positive_returns),
    ("全正收益 - Sortino比率", "sortino", positive_returns),
    ("全正收益 - 最大回撤", "max_drawdown", positive_returns),
    
    ("全负收益 - Sharpe比率", "sharpe", negative_returns),
    ("全负收益 - CAGR", "cagr", negative_returns),
    ("全负收益 - 胜率", "win_rate", negative_returns),
    
    ("极小收益 - 波动率", "volatility", tiny_returns),
    ("极小收益 - VaR", "value_at_risk", tiny_returns),
    
    ("包含NaN - Sharpe比率", "sharpe", nan_returns.dropna()),
    ("包含NaN - 波动率", "volatility", nan_returns.dropna()),
    
    ("极端波动 - 波动率", "volatility", extreme_returns),
    ("极端波动 - 偏度", "skew", extreme_returns),
    ("极端波动 - 峰度", "kurtosis", extreme_returns),
    
    ("单个值 - Sharpe比率", "sharpe", single_return),
    ("单个值 - 波动率", "volatility", single_return),
]

passed = 0
total = len(test_cases)

for description, func_name, data in test_cases:
    if compare_edge_case(description, func_name, data):
        passed += 1

print(f"\n边界条件测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

if passed == total:
    print("🎉 所有边界条件测试都通过！")
elif passed / total >= 0.8:
    print("✅ 大部分边界条件测试通过。")
else:
    print("⚠️  需要检查边界条件处理。")
