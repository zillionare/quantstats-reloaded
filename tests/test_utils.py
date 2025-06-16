#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantstats import utils

# 创建测试日期范围
@pytest.fixture
def dates():
    return pd.date_range(start='2020-01-01', end='2020-01-10')
        
def test_download_returns():
    # 测试下载单个股票的收益率
    returns = utils.download_returns('AAPL', period='1mo')
    
    # 验证返回的是pandas.Series
    assert isinstance(returns, pd.Series)
    
    # 验证数据不为空
    assert not returns.empty
    
    # 验证数据类型是浮点数
    assert returns.dtype == np.float64
        
def test_download_returns_period():
    # 测试使用日期范围下载收益率
    period = pd.date_range(start='2020-01-01', end='2020-01-31')
    returns = utils.download_returns('MSFT', period=period)
    
    # 验证返回的是pandas.Series
    assert isinstance(returns, pd.Series)
    
    # 验证数据不为空
    assert not returns.empty
        
def test_make_index():
    # 测试创建指数
    ticker_weights = {'AAPL': 0.5, 'MSFT': 0.5}
    index_returns = utils.make_index(ticker_weights, rebalance='1M', period='1mo')
    
    # 验证返回的是pandas.Series
    assert isinstance(index_returns, pd.Series)
    
    # 验证数据不为空
    assert not index_returns.empty
    
    # 验证数据类型是浮点数
    assert index_returns.dtype == np.float64