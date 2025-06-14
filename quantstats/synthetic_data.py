#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
QuantStats合成数据模块

此模块提供了创建合成金融数据的功能，用于在无法使用yfinance的情况下进行测试和演示。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_random_returns(start_date=None, end_date=None, periods=252, freq='B',
                         mean=0.0005, std=0.01, ticker='SYNTHETIC'):
    """
    创建随机收益率数据
    
    参数:
        start_date (str, datetime, optional): 开始日期，默认为一年前
        end_date (str, datetime, optional): 结束日期，默认为今天
        periods (int, optional): 如果未指定日期范围，则生成的数据点数量
        freq (str, optional): 日期频率，默认为'B'（工作日）
        mean (float, optional): 日收益率的平均值
        std (float, optional): 日收益率的标准差
        ticker (str, optional): 生成的Series的名称
        
    返回:
        pd.Series: 包含随机收益率的Series
    """
    if start_date is None and end_date is None:
        # 默认使用过去一年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    elif start_date is not None and end_date is not None:
        # 使用指定的日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    else:
        # 使用指定的周期数
        dates = pd.date_range(start='2020-01-01', periods=periods, freq=freq)
    
    # 生成随机收益率
    returns = np.random.normal(mean, std, len(dates))
    return pd.Series(returns, index=dates, name=ticker)


def create_random_prices(start_date=None, end_date=None, periods=252, freq='B',
                        mean=0.0005, std=0.01, initial_price=100.0, ticker='SYNTHETIC'):
    """
    创建随机价格数据
    
    参数:
        start_date (str, datetime, optional): 开始日期，默认为一年前
        end_date (str, datetime, optional): 结束日期，默认为今天
        periods (int, optional): 如果未指定日期范围，则生成的数据点数量
        freq (str, optional): 日期频率，默认为'B'（工作日）
        mean (float, optional): 日收益率的平均值
        std (float, optional): 日收益率的标准差
        initial_price (float, optional): 初始价格
        ticker (str, optional): 生成的Series的名称
        
    返回:
        pd.Series: 包含随机价格的Series
    """
    # 首先创建收益率
    returns = create_random_returns(start_date, end_date, periods, freq, mean, std, ticker)
    
    # 将收益率转换为价格
    prices = initial_price * (1 + returns).cumprod()
    return prices


def create_market_benchmark(returns=None, beta=1.0, alpha=0.0, noise=0.005, ticker='SPY'):
    """
    基于给定的收益率创建市场基准数据
    
    参数:
        returns (pd.Series, optional): 基础收益率，如果为None则创建随机数据
        beta (float, optional): 相对于基础收益率的beta系数
        alpha (float, optional): 相对于基础收益率的alpha值
        noise (float, optional): 添加到基准的噪声量
        ticker (str, optional): 生成的Series的名称
        
    返回:
        pd.Series: 包含基准收益率的Series
    """
    if returns is None:
        returns = create_random_returns(ticker='BASE')
    
    # 创建基准收益率: r_benchmark = alpha + beta * r_base + noise
    noise_component = np.random.normal(0, noise, len(returns))
    benchmark = alpha + beta * returns + noise_component
    benchmark.name = ticker
    
    return benchmark


def create_portfolio_returns(tickers=None, n_assets=5, periods=252, weights=None,
                           mean_range=(0.0003, 0.0007), std_range=(0.008, 0.015)):
    """
    创建投资组合收益率数据
    
    参数:
        tickers (list, optional): 资产的代码列表，如果为None则自动生成
        n_assets (int, optional): 如果未指定tickers，则生成的资产数量
        periods (int, optional): 生成的数据点数量
        weights (list, optional): 资产权重列表，如果为None则使用等权重
        mean_range (tuple, optional): 各资产平均收益率的范围
        std_range (tuple, optional): 各资产标准差的范围
        
    返回:
        tuple: (pd.DataFrame, pd.Series) - 资产收益率DataFrame和投资组合收益率Series
    """
    if tickers is None:
        tickers = [f'Asset_{i}' for i in range(n_assets)]
    else:
        n_assets = len(tickers)
    
    if weights is None:
        weights = np.ones(n_assets) / n_assets  # 等权重
    
    # 创建日期索引
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='B')
    
    # 为每个资产创建随机收益率
    assets_data = {}
    for i, ticker in enumerate(tickers):
        mean = np.random.uniform(mean_range[0], mean_range[1])
        std = np.random.uniform(std_range[0], std_range[1])
        returns = np.random.normal(mean, std, periods)
        assets_data[ticker] = returns
    
    # 创建资产收益率DataFrame
    assets_returns = pd.DataFrame(assets_data, index=dates)
    
    # 计算投资组合收益率
    portfolio_returns = (assets_returns * weights).sum(axis=1)
    portfolio_returns.name = 'Portfolio'
    
    return assets_returns, portfolio_returns


def load_csv_data(file_path, date_column='Date', price_column=None, returns_column=None,
                ticker=None, date_format=None):
    """
    从CSV文件加载金融数据
    
    参数:
        file_path (str): CSV文件路径
        date_column (str, optional): 日期列名
        price_column (str, optional): 价格列名，如果提供则计算收益率
        returns_column (str, optional): 收益率列名，如果未提供price_column则使用此列
        ticker (str, optional): 返回的Series的名称
        date_format (str, optional): 日期格式字符串，例如'%Y-%m-%d'
        
    返回:
        pd.Series: 包含收益率或价格的Series
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 处理日期列
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])
    
    # 设置日期索引
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    
    # 提取数据
    if price_column is not None:
        # 如果提供了价格列，计算收益率
        prices = df[price_column]
        if ticker:
            prices.name = ticker
        return prices
    elif returns_column is not None:
        # 如果提供了收益率列，直接使用
        returns = df[returns_column]
        if ticker:
            returns.name = ticker
        return returns
    else:
        # 如果只有一列数据（除了日期列），使用该列
        if len(df.columns) == 1:
            data = df[df.columns[0]]
            if ticker:
                data.name = ticker
            return data
        else:
            raise ValueError("必须指定price_column或returns_column，或者CSV文件只能包含一列数据（除了日期列）")