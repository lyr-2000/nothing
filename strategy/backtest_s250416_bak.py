import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# 添加策略目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyTT import *  # 导入MyTT模块的所有函数

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

def calculate_indicators(df):
    """计算技术指标，根据通达信代码s250415.txt实现，使用MyTT库"""
    # 准备OPEN, CLOSE, HIGH, LOW, VOL数据
    OPEN = np.array(df['Open'])
    CLOSE = np.array(df['Close'])
    HIGH = np.array(df['High'])
    LOW = np.array(df['Low'])
    VOL = np.array(df['Volume'])
    
    # 计算均线 MA
    df['MA5'] = MA(CLOSE, 5)
    df['MA10'] = MA(CLOSE, 10)
    df['MA20'] = MA(CLOSE, 20)
    df['MA60'] = MA(CLOSE, 60)
    df['MA120'] = MA(CLOSE, 120)
    
    # 计算KDJ
    N = 6
    M1 = 3
    M2 = 3
    P = 9
    
    K, D, J = KDJ(CLOSE, HIGH, LOW, N, M1, M2)
    df['RSV'] = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    df['K'] = K
    df['D'] = D
    df['J'] = J
    
    # 计算买入卖出信号 CROSS
    df['K_上穿_D'] = CROSS(K, D)
    df['D_上穿_K'] = CROSS(D, K)
    K_small_20 = np.array(K < 20, dtype=bool)  # 类型转换
    D_large_80 = np.array(D > 80, dtype=bool)  # 类型转换
    df['MAIRU'] = np.array(df['K_上穿_D'] & K_small_20, dtype=bool)
    df['MAICHU'] = np.array(df['D_上穿_K'] & D_large_80, dtype=bool)
    
    # VAR2到VAR9计算
    df['VAR2'] = REF(LOW, 1)
    
    # VAR3:=SMA(ABS(LOW-VAR2),3,1)/SMA(MAX(LOW-VAR2,0),3,1)*100;
    abs_diff = ABS(LOW - df['VAR2'])
    max_diff = MAX(LOW - df['VAR2'], 0)
    df['VAR3'] = np.array(SMA(abs_diff, 3, 1) / SMA(max_diff, 3, 1) * 100)
    
    # VAR4:=EMA(IF(CLOSE*1.3,VAR3*10,VAR3/10),3);
    condition = np.array(CLOSE * 1.3 > 0, dtype=bool)  # 类型转换
    value = IF(condition, df['VAR3'] * 10, df['VAR3'] / 10)
    df['VAR4'] = EMA(value, 3)
    
    # VAR5:=LLV(LOW,13);
    df['VAR5'] = LLV(LOW, 13)
    
    # VAR6:=HHV(VAR4,13);
    df['VAR6'] = HHV(df['VAR4'], 13)
    
    # VAR7:=IF(MA(CLOSE,34),1,0);
    ma34 = MA(CLOSE, 34)
    condition = np.array(ma34 > 0, dtype=bool)  # 类型转换
    df['VAR7'] = IF(condition, 1, 0)
    
    # VAR8:=EMA(IF(LOW<=VAR5,(VAR4+VAR6*2)/2,0),3)/618*VAR7;
    LOW_arr = np.array(LOW)
    VAR5_arr = np.array(df['VAR5'])
    condition = np.array(LOW_arr <= VAR5_arr, dtype=bool)  # 类型转换
    VAR4_arr = np.array(df['VAR4'])
    VAR6_arr = np.array(df['VAR6'])
    value = IF(condition, (VAR4_arr + VAR6_arr * 2) / 2, 0)
    VAR7_arr = np.array(df['VAR7'])
    df['VAR8'] = np.array(EMA(value, 3) / 618 * VAR7_arr)
    
    # VAR9:=IF(VAR8>100,100,VAR8);
    VAR8_arr = np.array(df['VAR8'])
    condition = np.array(VAR8_arr > 100, dtype=bool)  # 类型转换
    df['VAR9'] = IF(condition, 100, VAR8_arr)
    
    # 下影线相关计算
    # 下影线:=(O-LOW)/REF(C,1)>(H-C)/REF(C,1)*2;
    OPEN_arr = np.array(OPEN)
    LOW_arr = np.array(LOW)
    CLOSE_arr = np.array(CLOSE)
    HIGH_arr = np.array(HIGH)
    REF_CLOSE = REF(CLOSE_arr, 1)
    left_part = (OPEN_arr - LOW_arr) / REF_CLOSE
    right_part = (HIGH_arr - CLOSE_arr) / REF_CLOSE * 2
    df['下影线'] = np.array(left_part > right_part, dtype=bool)
    
    # 放量:=VOL>REF(VOL,1);
    df['放量'] = np.array(VOL > REF(VOL, 1), dtype=bool)
    
    # X影线:=MIN(C,O)-L;
    df['X影线'] = MIN(CLOSE_arr, OPEN_arr) - LOW_arr
    
    # 上影线:=H-MAX(C,O);
    df['上影线'] = HIGH_arr - MAX(CLOSE_arr, OPEN_arr)
    
    # 阳线:=C>O;
    df['阳线'] = np.array(CLOSE_arr > OPEN_arr, dtype=bool)
    
    # 前一天条件
    X影线_arr = np.array(df['X影线'])
    上影线_arr = np.array(df['上影线'])
    cond1 = np.array(X影线_arr >= 上影线_arr * 2, dtype=bool)
    cond2 = np.array(CLOSE_arr / REF(CLOSE_arr, 1) <= 1.03, dtype=bool)
    cond3 = np.array(CLOSE_arr >= OPEN_arr, dtype=bool)
    cond4 = np.array(CLOSE_arr >= REF(CLOSE_arr, 1), dtype=bool)
    cond5 = np.array(REF(CLOSE_arr, 1) <= REF(OPEN_arr, 1), dtype=bool)
    cond6 = np.array(LOW_arr < REF(CLOSE_arr, 1), dtype=bool)
    df['前一天条件'] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    
    # 真阳线:=C>O AND C/REF(C,1)>=1;
    cond1 = np.array(CLOSE_arr > OPEN_arr, dtype=bool)
    cond2 = np.array(CLOSE_arr / REF(CLOSE_arr, 1) >= 1, dtype=bool)
    df['真阳线'] = cond1 & cond2
    
    # 下影线成功:=COUNT(REF(前一天条件,1) AND C/REF(C,1)>=1,120);
    前一天条件_arr = np.array(df['前一天条件'])
    cond1 = np.array(REF(前一天条件_arr, 1), dtype=bool)
    cond2 = np.array(CLOSE_arr / REF(CLOSE_arr, 1) >= 1, dtype=bool)
    df['下影线成功条件'] = cond1 & cond2
    df['下影线成功'] = COUNT(df['下影线成功条件'], 120)
    
    # 下影线失败:=COUNT(REF(前一天条件,1) AND C/REF(C,1)<=1,120);
    cond1 = np.array(REF(前一天条件_arr, 1), dtype=bool)
    cond2 = np.array(CLOSE_arr / REF(CLOSE_arr, 1) <= 1, dtype=bool)
    df['下影线失败条件'] = cond1 & cond2
    df['下影线失败'] = COUNT(df['下影线失败条件'], 120)
    
    # 下影线性价比:=下影线成功/下影线失败;
    下影线成功_arr = np.array(df['下影线成功'])
    下影线失败_arr = np.array(df['下影线失败'])
    df['下影线性价比'] = 下影线成功_arr / (下影线失败_arr + 1e-10)  # 避免除以0
    
    # 连续阳线计算
    # ZD:=C>REF(C,1);
    df['ZD'] = np.array(CLOSE_arr > REF(CLOSE_arr, 1), dtype=bool)
    
    # LX:=BARSLASTCOUNT(ZD);
    ZD_arr = np.array(df['ZD'])
    df['LX'] = BARSLASTCOUNT(ZD_arr)
    
    # TJ:=LX>=2;
    LX_arr = np.array(df['LX'])
    df['TJ'] = np.array(LX_arr >= 2, dtype=bool)
    
    # TJ1:=REF(TJ,1)=0 AND TJ;
    TJ_arr = np.array(df['TJ'])
    cond1 = np.array(REF(TJ_arr, 1) == 0, dtype=bool)
    cond2 = np.array(TJ_arr, dtype=bool)
    df['TJ1'] = cond1 & cond2
    
    # 连续阳线次:=COUNT(TJ1,120);
    TJ1_arr = np.array(df['TJ1'])
    df['连续阳线次'] = COUNT(TJ1_arr, 120)
    
    # 下影加连阳:=连续阳线次+下影线性价比;
    连续阳线次_arr = np.array(df['连续阳线次'])
    下影线性价比_arr = np.array(df['下影线性价比'])
    df['下影加连阳'] = 连续阳线次_arr + 下影线性价比_arr
    
    # 去除条件（这里暂时简化处理）
    # 去除:=S1 AND S2 AND S5 AND S4 AND S6 AND S7;
    df['去除'] = np.ones(len(CLOSE), dtype=bool)  # 全部设为True
    
    # 吸筹信号
    VAR9_arr = np.array(df['VAR9'])
    去除_arr = np.array(df['去除'])
    下影线_arr = np.array(df['下影线'])
    下影线性价比_arr = np.array(df['下影线性价比'])
    
    cond1 = np.array(VAR9_arr > 5, dtype=bool)
    cond2 = np.array(去除_arr, dtype=bool)
    cond3 = np.array(CLOSE_arr / REF(CLOSE_arr, 1) < 1.03, dtype=bool)
    cond4 = np.array(CLOSE_arr > OPEN_arr, dtype=bool)
    cond5 = np.array(CLOSE_arr >= REF(CLOSE_arr, 1), dtype=bool)
    cond6 = np.array(下影线_arr, dtype=bool)
    cond7 = np.array(REF(CLOSE_arr, 1) <= REF(OPEN_arr, 1), dtype=bool)
    cond8 = np.array(LOW_arr < REF(CLOSE_arr, 1), dtype=bool)
    cond9 = np.array(下影线性价比_arr > 2, dtype=bool)
    
    df['吸筹'] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9
               
    return df

def plot_kline(df, title='K线图'):
    """绘制K线图"""
    # 创建子图
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=(title, 'KDJ', '吸筹指标(VAR9)', '成交量'),
                       row_heights=[0.5, 0.15, 0.15, 0.15])

    # 添加K线
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
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
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
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
    df['收益率'] = df['Close'].pct_change()
    
    # 计算策略收益率
    df['策略收益率'] = df['收益率'] * df['买入信号'].shift(1)
    df['累计收益率'] = (1 + df['策略收益率']).cumprod()
    
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
    print(f"股票收益率: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"策略收益率: {((df['累计收益率'].iloc[-1]) - 1) * 100:.2f}%")
    print(f"吸筹信号次数: {df['吸筹'].sum()}")
    print(f"吸筹信号胜率: {(df[df['吸筹'] == True]['收益率'] > 0).mean() * 100:.2f}%")