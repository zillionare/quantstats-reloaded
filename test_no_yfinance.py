#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import quantstats as qs
import pandas as pd
import numpy as np

print('QuantStats version:', qs.__version__)

print('\nTesting download_returns function:')
try:
    returns = qs.utils.download_returns('AAPL')
    print('Returns shape:', returns.shape)
except Exception as e:
    print('Error:', e)

print('\nTesting with manual data:')
# 创建一个示例数据集
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
prices = np.random.random(100) * 100 + 100  # 随机价格数据
returns = pd.Series(data=prices, index=dates).pct_change().dropna()

# 测试基本统计功能
print('Sharpe Ratio:', qs.stats.sharpe(returns))
print('CAGR:', qs.stats.cagr(returns))
print('Max Drawdown:', qs.stats.max_drawdown(returns))

print('\nAll tests completed!')