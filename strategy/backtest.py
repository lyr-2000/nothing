import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    # 将A股代码转换为yfinance格式
    if stock_code.endswith('.SS'):
        yf_code = stock_code.replace('.SS', '.SS')
    elif stock_code.endswith('.SZ'):
        yf_code = stock_code.replace('.SZ', '.SZ')
    else:
        yf_code = stock_code
        
    # 获取数据
    stock = yf.Ticker(yf_code)
    df = stock.history(start=start_date, end=end_date)
    return df

def plot_kline(df, title='K线图'):
    """绘制K线图"""
    # 创建子图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(title, '成交量'),
                       row_heights=[0.7, 0.3])

    # 添加K线
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='K线'),
                 row=1, col=1)

    # 添加成交量
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        marker_color=colors,
                        name='成交量'),
                 row=2, col=1)

    # 更新布局
    fig.update_layout(
        title=title,
        yaxis_title='价格',
        yaxis2_title='成交量',
        xaxis_rangeslider_visible=False,
        height=800
    )

    # 显示图表
    fig.show()

def backtest_strategy(stock_code, start_date, end_date):
    """回测策略"""
    # 获取数据
    df = get_stock_data(stock_code, start_date, end_date)
    
    # 计算技术指标
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 计算RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 绘制K线图
    plot_kline(df, title=f'{stock_code} K线图')
    
    # 返回数据
    return df

if __name__ == '__main__':
    # 设置回测参数
    stock_code = '600519.SS'  # 贵州茅台
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 回测一年
    
    # 运行回测
    df = backtest_strategy(stock_code, start_date, end_date)
    
    # 打印数据统计
    print("\n数据统计:")
    print(f"回测期间: {start_date.date()} 至 {end_date.date()}")
    print(f"初始价格: {df['Close'].iloc[0]:.2f}")
    print(f"最终价格: {df['Close'].iloc[-1]:.2f}")
    print(f"收益率: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%") 