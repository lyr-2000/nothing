# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 策略计算函数
def calculate_xi_chou(df):
    # 板块过滤条件
    df['S1'] = (~df['name'].str.startswith('S')).astype(int)
    df['S2'] = (~df['name'].str.contains(r'\*', regex=False)).astype(int)
    df['S4'] = (~df['is_科创板']).astype(int)
    df['S5'] = (~df['name'].str.contains('C')).astype(int)
    df['S6'] = (~df['is_创业板']).astype(int)
    df['S7'] = (~df['is_北证50']).astype(int)
    df['去除'] = df['S1'] & df['S2'] & df['S5'] & df['S4'] & df['S6'] & df['S7']

    # 指标计算
    N, M1, M2, P = 6, 3, 3, 9
    df['M60'] = df['close'].rolling(60).mean()
    df['M120'] = df['close'].rolling(120).mean()
    
    llv = df['low'].rolling(N).min()
    hhv = df['high'].rolling(N).max()
    df['RSV'] = (df['close'] - llv) / (hhv - llv).replace(0, np.nan) * 100
    df['RSV'] = df['RSV'].fillna(0)
    
    df['K'] = df['RSV'].ewm(span=M1, adjust=False).mean()
    df['D'] = df['K'].ewm(span=M2, adjust=False).mean()
    
    # 下影线条件
    df['X影线'] = np.minimum(df['close'], df['open']) - df['low']
    df['上影线'] = df['high'] - np.maximum(df['close'], df['open'])
    df['阳线'] = df['close'] > df['open']
    
    df['前一天条件'] = (
        (df['X影线'].shift(1) >= df['上影线'].shift(1) * 2) &
        (df['close'].shift(1)/df['close'].shift(2) <= 1.03) &
        (df['close'].shift(1) >= df['open'].shift(1)) &
        (df['close'].shift(1) >= df['close'].shift(2)) &
        (df['close'].shift(2) <= df['open'].shift(2)) &
        (df['low'].shift(1) < df['close'].shift(2))
    )
    
    # 下影线性价比
    success = df['前一天条件'].shift(1) & (df['close']/df['close'].shift(1) >= 1)
    failure = df['前一天条件'].shift(1) & (df['close']/df['close'].shift(1) < 1)
    df['下影线成功'] = success.rolling(120).sum()
    df['下影线失败'] = failure.rolling(120).sum()
    df['下影线性价比'] = df['下影线成功'] / df['下影线失败'].replace(0, np.nan)
    
    # 吸筹信号
    df['吸筹'] = (
        (df['去除']) &
        (df['close']/df['close'].shift(1) < 1.03) &
        (df['close'] > df['open']) &
        (df['close'] >= df['close'].shift(1)) &
        ((df['open'] - df['low']) > 2*(df['high'] - df['close'])) &
        (df['close'].shift(1) <= df['open'].shift(1)) &
        (df['low'] < df['close'].shift(1)) &
        (df['下影线性价比'] >= 2)
    )
    return df
def extract_digits_v3(code):
    return ''.join(filter(str.isdigit, code))
# 获取数据
def get_kline(stock_code="000001"):
    stock_code = extract_digits_v3(stock_code)
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
    df.rename(columns={
        '日期':'date', '开盘':'open', '收盘':'close',
        '最高':'high', '最低':'low', '成交量':'volume'
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

# 添加板块信息
def add_blocks(df, stock_code):
    df['name'] = stock_code  # 实际应用应从接口获取名称
    df['is_科创板'] = stock_code.startswith('688')
    df['is_创业板'] = stock_code.startswith('300')
    df['is_北证50'] = stock_code.startswith('8')
    return df

# Plotly可视化
def plot_strategy(df, stock_code):
    signals = df[df['吸筹']]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])
    
    # K线图
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线'),
                 row=1, col=1)
    
    # 吸筹信号
    fig.add_trace(go.Scatter(x=signals.index,
                            y=signals['high']*1.01,
                            mode='markers',
                            marker=dict(color='gold', size=10),
                            name='吸筹信号'),
                 row=1, col=1)
    
    # 成交量
    fig.add_trace(go.Bar(x=df.index,
                        y=df['volume'],
                        name='成交量',
                        marker_color='grey'),
                 row=2, col=1)
    
    # 布局设置
    fig.update_layout(
        title=f'{stock_code} 吸筹策略信号',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig

# 主程序
if __name__ == '__main__':
    stock_code = "603917"  # 贵州茅台
    df = get_kline(stock_code)
    df = add_blocks(df, stock_code)
    df = calculate_xi_chou(df)
    fig = plot_strategy(df.tail(120), stock_code)  # 展示最近120天
    fig.show()