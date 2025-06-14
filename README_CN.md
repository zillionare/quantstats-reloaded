# QuantStats 中国大陆使用指南

## 简介

QuantStats是一个强大的Python库，用于执行投资组合分析，帮助量化分析师和投资组合经理通过提供深入的分析和风险指标来更好地理解其投资表现。

## 在中国大陆使用的特别说明

由于网络限制，在中国大陆无法正常使用`yfinance`库下载股票数据。为了解决这个问题，我们对QuantStats进行了以下修改：

1. 移除了对`yfinance`的强制依赖
2. 提供了替代方案，允许用户手动提供数据

## 如何使用

### 1. 手动提供数据

在中国大陆使用QuantStats时，您需要手动提供股票数据，而不是依赖自动下载功能。以下是几种方法：

#### 方法一：使用本地CSV文件

```python
import pandas as pd
import quantstats as qs

# 从本地CSV文件加载数据
stock_data = pd.read_csv('your_stock_data.csv', index_col='Date', parse_dates=True)

# 如果数据是价格数据，需要转换为收益率
if stock_data.min().min() > 1:  # 判断是否为价格数据
    stock_returns = stock_data.pct_change().dropna()
else:
    stock_returns = stock_data  # 已经是收益率数据

# 使用QuantStats分析
qs.reports.html(stock_returns, output='my_report.html')
```

#### 方法二：使用其他数据API

您可以使用其他可在中国大陆访问的金融数据API，如：

- [Tushare](https://tushare.pro/)
- [Baostock](http://baostock.com/)
- [Akshare](https://akshare.akfamily.xyz/)

```python
# 使用Tushare示例
import tushare as ts
import pandas as pd
import quantstats as qs

# 设置Tushare token
ts.set_token('your_token_here')
pro = ts.pro_api()

# 获取股票日线数据
df = pro.daily(ts_code='000001.SZ', start_date='20200101', end_date='20231231')

# 处理数据格式
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)
df.sort_index(inplace=True)  # 确保日期升序排列

# 计算日收益率
returns = df['close'].pct_change().dropna()

# 使用QuantStats分析
qs.reports.html(returns, output='my_report.html')
```

### 2. 使用基准数据

当需要使用基准数据进行比较时，不要使用字符串标识符（如'SPY'），而是直接提供基准数据的Series或DataFrame：

```python
# 错误方式（在中国大陆不可用）:
qs.reports.html(stock_returns, 'SPY', output='my_report.html')

# 正确方式:
benchmark_returns = pd.read_csv('benchmark_data.csv', index_col='Date', parse_dates=True)
qs.reports.html(stock_returns, benchmark_returns, output='my_report.html')
```

### 3. 创建投资组合指数

使用`make_index`函数时，必须提供`returns`参数：

```python
# 准备多个股票的收益率数据
stock1 = pd.read_csv('stock1.csv', index_col='Date', parse_dates=True)
stock2 = pd.read_csv('stock2.csv', index_col='Date', parse_dates=True)

# 合并为一个DataFrame
all_stocks = pd.DataFrame({
    'stock1': stock1['Close'].pct_change(),
    'stock2': stock2['Close'].pct_change()
}).dropna()

# 创建投资组合指数
portfolio_weights = {'stock1': 0.6, 'stock2': 0.4}
portfolio = qs.utils.make_index(portfolio_weights, returns=all_stocks)

# 分析投资组合
qs.reports.html(portfolio, output='portfolio_report.html')
```

## 注意事项

1. 所有需要下载数据的函数（如`download_returns`）都已被修改，会显示警告信息并返回空数据
2. 使用`_prepare_benchmark`函数时，必须直接提供基准数据，而不是使用字符串标识符
3. 使用`make_index`函数时，必须提供`returns`参数

## 数据格式要求

手动提供的数据应满足以下要求：

1. 使用pandas的Series或DataFrame格式
2. 索引应为日期格式（DatetimeIndex）
3. 数据可以是价格数据或收益率数据，QuantStats会自动检测并转换

## 贡献

如果您有任何改进建议或发现任何问题，请在GitHub上提交issue或pull request。