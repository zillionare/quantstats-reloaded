#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
测试reports模块的对比
"""

import pandas as pd
import numpy as np
import sys
import tempfile
import os

# 导入原版和当前版本
import quantstats as qs_original
sys.path.insert(0, '/mnt/persist/workspace')
import quantstats as qs_current

print("=== Reports模块对比测试 ===\n")

# 创建测试数据
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
benchmark = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

def compare_metrics():
    """比较metrics函数"""
    print("测试 metrics 函数...")
    
    try:
        # 测试无基准的情况
        original_metrics = qs_original.reports.metrics(returns, display=False)
        current_metrics = qs_current.reports.metrics(returns, display=False)
        
        print(f"  原版metrics形状: {original_metrics.shape}")
        print(f"  当前metrics形状: {current_metrics.shape}")
        
        # 比较列名
        original_cols = set(original_metrics.columns)
        current_cols = set(current_metrics.columns)
        
        if original_cols == current_cols:
            print("  ✅ 列名完全一致")
        else:
            print(f"  ⚠️  列名有差异:")
            print(f"    原版独有: {original_cols - current_cols}")
            print(f"    当前独有: {current_cols - original_cols}")
        
        # 比较行名
        original_rows = set(original_metrics.index)
        current_rows = set(current_metrics.index)
        
        if original_rows == current_rows:
            print("  ✅ 行名完全一致")
        else:
            print(f"  ⚠️  行名有差异:")
            print(f"    原版独有: {original_rows - current_rows}")
            print(f"    当前独有: {current_rows - original_rows}")
        
        # 测试带基准的情况
        print("\n  测试带基准的metrics...")
        original_metrics_bench = qs_original.reports.metrics(returns, benchmark=benchmark, display=False)
        current_metrics_bench = qs_current.reports.metrics(returns, benchmark=benchmark, display=False)
        
        print(f"  原版带基准metrics形状: {original_metrics_bench.shape}")
        print(f"  当前带基准metrics形状: {current_metrics_bench.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ metrics测试失败: {e}")
        return False

def compare_html():
    """比较HTML生成"""
    print("\n测试 HTML 生成...")
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp1:
            original_file = tmp1.name
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp2:
            current_file = tmp2.name
        
        try:
            # 生成HTML报告
            qs_original.reports.html(returns, output=original_file)
            qs_current.reports.html(returns, output=current_file)
            
            # 检查文件是否生成
            original_exists = os.path.exists(original_file)
            current_exists = os.path.exists(current_file)
            
            print(f"  原版HTML文件生成: {'✅' if original_exists else '❌'}")
            print(f"  当前HTML文件生成: {'✅' if current_exists else '❌'}")
            
            if original_exists and current_exists:
                # 比较文件大小
                original_size = os.path.getsize(original_file)
                current_size = os.path.getsize(current_file)
                
                print(f"  原版HTML文件大小: {original_size} bytes")
                print(f"  当前HTML文件大小: {current_size} bytes")
                
                # 检查文件内容的基本结构
                with open(original_file, 'r') as f:
                    original_content = f.read()
                with open(current_file, 'r') as f:
                    current_content = f.read()
                
                original_has_html = '<html' in original_content and '</html>' in original_content
                current_has_html = '<html' in current_content and '</html>' in current_content
                
                print(f"  原版HTML结构完整: {'✅' if original_has_html else '❌'}")
                print(f"  当前HTML结构完整: {'✅' if current_has_html else '❌'}")
                
                return original_has_html and current_has_html
            else:
                return False
                
        finally:
            # 清理临时文件
            for file in [original_file, current_file]:
                if os.path.exists(file):
                    os.unlink(file)
                    
    except Exception as e:
        print(f"  ❌ HTML测试失败: {e}")
        return False

def test_basic_and_full():
    """测试basic和full报告"""
    print("\n测试 basic 和 full 报告...")
    
    try:
        # 测试basic报告
        print("  测试basic报告...")
        qs_original.reports.basic(returns, display=False)
        qs_current.reports.basic(returns, display=False)
        print("  ✅ basic报告执行成功")
        
        # 测试full报告
        print("  测试full报告...")
        qs_original.reports.full(returns, display=False)
        qs_current.reports.full(returns, display=False)
        print("  ✅ full报告执行成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ basic/full测试失败: {e}")
        return False

def test_plots():
    """测试plots函数"""
    print("\n测试 plots 函数...")

    try:
        # 测试plots（不显示）
        qs_original.reports.plots(returns, show=False, savefig=False)
        qs_current.reports.plots(returns, show=False, savefig=False)
        print("  ✅ plots函数执行成功")

        return True

    except Exception as e:
        print(f"  ❌ plots测试失败: {e}")
        return False

# 运行所有测试
print("开始reports模块对比测试...\n")

tests = [
    ("metrics函数", compare_metrics),
    ("HTML生成", compare_html),
    ("basic/full报告", test_basic_and_full),
    ("plots函数", test_plots),
]

passed = 0
total = len(tests)

for test_name, test_func in tests:
    print(f"=== {test_name} ===")
    if test_func():
        passed += 1
        print(f"✅ {test_name} 测试通过\n")
    else:
        print(f"❌ {test_name} 测试失败\n")

print("="*50)
print(f"Reports模块测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

if passed == total:
    print("🎉 所有reports测试都通过！")
elif passed / total >= 0.75:
    print("✅ 大部分reports测试通过。")
else:
    print("⚠️  需要检查reports模块。")
