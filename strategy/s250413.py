# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Mylib import *
from MyTT import *

# 策略计算函数
def calculate_xi_chou(df):
    # print(df.columns)
    # 板块过滤条件
    df['S1'] = (~df['name'].str.startswith('S')).astype(int)
    df['S2'] = (~df['name'].str.contains(r'\*', regex=False)).astype(int)
    df['S4'] = (~df['is_科创板']).astype(int)
    df['S5'] = (~df['name'].str.contains('C')).astype(int)
    df['S6'] = (~df['is_创业板']).astype(int)
    df['S7'] = (~df['is_北证50']).astype(int)
    df['去除'] = df['S1'] & df['S2'] & df['S5'] & df['S4'] & df['S6'] & df['S7']
    LOW = df.low.values
    CLOSE = df.close.values
    C = df.close.values
    O = df.open.values
    HIGH = df.high.values
    H  = HIGH
    L = LOW
    # 指标计算
    N, M1, M2, P = 6, 3, 3, 9
    M60=MA(C,60);
    M120=MA(C,120);
    VOL = df.volume.values
   
    RSV=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    K=SMA(RSV,M1,1);

    D=SMA(K,M2,1);
    
    JJ=P*(3*D-2*K);

    J=((3 * K) - (2 * D));
   
    MAICHU = (CROSS(D,K) )& ( D>80);
   
    VAR2 =REF(LOW,1);
    VAR3= SMA(ABS(LOW-VAR2),3,1)/SMA(MAX(LOW-VAR2,0),3,1)*100;

    VAR4 =EMA(IF(CLOSE*1.3,VAR3*10,VAR3/10),3);

    VAR5 =LLV(LOW,13);

    VAR6 =HHV(VAR4,13);

    VAR7 =IF(MA(CLOSE,34),1,0);

    VAR8 =EMA(IF(LOW<=VAR5,(VAR4+VAR6*2)/2,0),3)/618*VAR7;

    VAR9 =IF(VAR8>100,100,VAR8);
    下影线 =(O-LOW)/REF(C,1)>(H-C)/REF(C,1)*2;
    放量 =VOL>REF(VOL,1);
    
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
    df['吸筹指标'] = VAR9
    # 吸筹信号
    df['吸筹'] = (
        (df['去除']) & (df['吸筹指标'] > 5)&
        (  C/REF(C,1) < 1.03 ) &
        (  C > O ) &
        (  C>=REF(C,1) ) & (下影线) &
        (  REF(C,1)<=REF(O,1) ) &
        ( L<REF(C,1) ) &
        (df['下影线性价比'] >= 2)
    )
    return df


# Plotly可视化
def plot_strategy(df, stock_code,graph_height=1500,rows=3):
    signals = df[df['吸筹']]
    # print(df.columns)
    # df.index = df['date']
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    #    vertical_spacing=0.05,
                    #    specs=[  [{"type": "scatter"}],[{"type": "scatter"}],[{"type": "scatter"}],   [{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}],[{"type": "table"}]],
                    #    row_heights=[0.7, 0.3],
                       )
    
    # K线图
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线',
                                 # 关键参数：设置红涨绿跌
                                increasing={
                                    'line': {'color': 'red'},       # 上涨K线的边框色
                                    'fillcolor': 'red'              # 上涨K线的填充色
                                },
                                decreasing={
                                    'line': {'color': 'green'},     # 下跌K线的边框色
                                    'fillcolor': 'green'            # 下跌K线的填充色
                                }
                                
                                ),
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
    fig.add_trace(go.Scatter(x=df.index,
                        y=df['吸筹指标'],
                        name='吸筹指标',
                        marker_color='yellow'),
                 row=3, col=1)
    # 方法1：使用 add_hline 直接添加水平线（推荐）
    fig.add_hline(y=5, line_width=3, line_dash="dash", 
                line_color="red", annotation_text="关键阈值 5",
                annotation_position="bottom right",row=3,col=1)
    # 布局设置
    fig.update_layout(
        title=f'{stock_code} 吸筹策略信号',
        xaxis_rangeslider_visible=False,
        height=graph_height,
        legend_title="Legend",
        hovermode="x unified",
     
    )
    fig.update_xaxes(
        type='date',
        dtick='D',  # 设置x轴刻度间隔为每天
        tickformat='%Y-%m-%d',  # 自定义时间格式，这里为年月日
        rangeslider=dict(visible=False),
    )
    
    return fig




def plot_output(stock_code,stock_name,dir='./backtest_result'):
    # stock_code = "603917"  # 贵州茅台
    df_raw = get_kline(stock_code)
    df_raw = calculate_xi_chou(df=df_raw)
    df = df_raw.tail(365*3)
    a,b = backtest_strategy(df,trigger_signal=['吸筹'],)
    # fig.show()
    # fig.write_html(f"./demo_{stock_code}.html",encoding='utf-8')
    if  b and len(a)>0 :
        fig = plot_strategy( df, stock_code,rows=3+4,graph_height=800)  # 展示最近120天
        s = fig.to_html(include_plotlyjs='cdn')
        import os
        os.makedirs(dir,exist_ok=True)
        with open(f'./{dir}/回测结果_{stock_name}.html','w',encoding='utf-8') as w:
            w.write(f"<h2>股票名字: {stock_name}</h2>")
            w.write(dict_to_html(b))
            w.write(a.to_html())
            w.write(f'{s}')
    vobj = {
        'stat': b,
        'records': a.to_json(orient="records"),
    }
    return vobj

# 主程序
if __name__ == '__test1__':
    import time
    stock_code = "603917"  # 贵州茅台
    df_raw = get_kline(stock_code)
    df_raw = calculate_xi_chou(df=df_raw)
    df = df_raw.tail(365*3)
    fig = plot_strategy( df, stock_code,rows=3+4,graph_height=800)  # 展示最近120天
    a,b = backtest_strategy(df,trigger_signal=['吸筹'],)

    # fig.show()
    # fig.write_html(f"./demo_{stock_code}.html",encoding='utf-8')
    s = fig.to_html()
    if a and b:
        with open(f'./demo_{stock_code}.html','w',encoding='utf-8') as w:
            w.write(f"<h2>股票名字: {stock_code}</h2>")
            # w.write(f'<div style="height:500px;">{s}</div>')
            w.write(dict_to_html(b))
            w.write(a.to_html())
            w.write(f'{s}')
        

# ------------- todo ----------
# 常量设置
INTERVAL_MAP = {
    # "1小时": "60",
    # "4小时": "240",
    "日线": "daily"
}
import os
import time
import json
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "stock_list_cache.json")
CACHE_EXPIRY_HOURS = 6
from pathlib import Path
# 缓存相关函数
def ensure_cache_dir():
    Path(CACHE_DIR).mkdir(exist_ok=True)

def read_cache():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < CACHE_EXPIRY_HOURS * 3600:
                return cache_data['data']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None

def write_cache(data):
    try:
        ensure_cache_dir()
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'data': data
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入缓存失败: {str(e)}")

def fetch_stock_list():
    cached_data = read_cache()
    if cached_data is not None:
        return cached_data
    
    try:
        spot_df = ak.stock_zh_a_spot()
        stock_list = {f"{name} ({code})": code for code, name in zip(spot_df['代码'], spot_df['名称'])}
        write_cache(stock_list)
        return stock_list
    except Exception as e:
        print(f"获取股票列表失败: {str(e)}")
        # default_data = {"贵州茅台 (600519)": "600519"}
        # write_cache(default_data)
        return []



if __name__ == '__main__':
    stocklist = fetch_stock_list()
    # print(stocklist)
    for name in dict(stocklist):
        code = stocklist[name]
        try:
            plot_output(code,name)
        except Exception as e:
            print(e)
        