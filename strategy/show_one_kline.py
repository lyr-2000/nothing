import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import sys
import os
import akshare as ak
# 添加策略目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyTT import *  # 导入MyTT模块的所有函数



from backtest_multi_stocks import calculate_indicators,get_stock_data


def plot_kline(df, title='K线图'):
    """绘制K线图"""
    # 创建子图
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=(title, 'KDJ', '吸筹指标(VAR9)', '成交量'),
                       row_heights=[0.5, 0.15, 0.15, 0.15])

    # 添加K线
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线'),
                 row=1, col=1)
    
    # 添加均线 - 检查均线是否存在再添加
    if 'MA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5'), row=1, col=1)
    if 'MA10' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='MA10'), row=1, col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'), row=1, col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60'), row=1, col=1)
    if 'MA120' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], name='MA120'), row=1, col=1)

    # 添加KDJ
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J'), row=2, col=1)
    
    # 添加VAR9吸筹指标
    fig.add_trace(go.Scatter(x=df.index, y=df['VAR9'], name='VAR9', line=dict(color='purple', width=2)), row=3, col=1)
    # 添加阈值线
    fig.add_trace(go.Scatter(x=df.index, y=[5] * len(df.index), name='VAR9阈值', line=dict(color='red', width=1, dash='dash')), row=3, col=1)
    
    # 标记吸筹信号
    signal_dates = df[df['吸筹'] == True].index
    if len(signal_dates) > 0:
        fig.add_trace(go.Scatter(
            x=signal_dates,
            y=df.loc[signal_dates, 'VAR9'],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color='red',
            ),
            name='吸筹信号'
        ), row=3, col=1)

    # 添加成交量
    colors = ['red' if row['open'] - row['close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                        marker_color=colors,
                        name='成交量'),
                 row=4, col=1)

    # 更新布局
    fig.update_layout(
        title=title,
        yaxis_title='价格',
        yaxis2_title='KDJ',
        yaxis3_title='VAR9',
        yaxis4_title='成交量',
        xaxis_rangeslider_visible=False,
        height=1000
    )
    
    # 更新x轴格式
    fig.update_xaxes(
        tickformat='%Y-%m-%d',
        tickangle=45
    )

    # 显示图表
    fig.show()
    
    return fig

def backtest_strategy(stock_code, start_date, end_date):
    """回测策略"""
    # 获取数据
    df = get_stock_data(stock_code, start_date, end_date)
    
    # 计算技术指标
    df = calculate_indicators(df)
    
    # 使用吸筹信号作为买入信号
    df['买入信号'] = df['吸筹']
    
    # 计算收益率
    df['收益率'] = df['close'].pct_change()
    
    # 计算策略收益率
    df['策略收益率'] = df['收益率'] * df['买入信号'].shift(1)
    df['累计收益率'] = (1 + df['策略收益率']).cumprod()
    
    # 绘制K线图
    plot_kline(df, title=f'{stock_code} K线图')
    
    # 返回数据
    return df

if __name__ == '__main__':
    # 设置回测参数
    stock_code = '600655'  #  
    # stock_code = '600332'  # 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 回测一年
    
    # 运行回测
    df = backtest_strategy(stock_code, start_date, end_date)
    
    # 打印数据统计
    print("\n数据统计:")
    print(f"回测期间: {start_date.date()} 至 {end_date.date()}")
    # print(df.head(10))
    val = df[df['t'] == '2024-02-05']
    # 显示所有列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 4)
    print(val[['吸筹指标','下影线性价比','下影加连阳','吸筹']])

    # print(f"吸筹信号次数: {df['吸筹'].sum()}")
    print(f"吸筹信号胜率: {(df[df['吸筹'] == True]['收益率'] > 0).mean() * 100:.2f}%")