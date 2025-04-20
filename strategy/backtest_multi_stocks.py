import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import tqdm
import akshare as ak  # 使用akshare获取A股数据
from Mylib import extract_digits_v3
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import plotly.io as pio
from plotly.io import write_html  # 明确导入write_html函数
import time
import json
import logging
import argparse

"""
多股票回测系统 - 优化版
======================

该系统可以回测多支股票的交易策略，支持数据缓存和多线程处理，大幅提高回测速度。

使用方法：
--------
1. 基本用法：python backtest_multi_stocks.py
2. 指定日期：python backtest_multi_stocks.py --start 20230101 --end 20230228
3. 增加选股数：python backtest_multi_stocks.py --top 5
4. 限制股票数量：python backtest_multi_stocks.py --limit 200
5. 处理所有股票：python backtest_multi_stocks.py --all-stocks
6. 忽略缓存：python backtest_multi_stocks.py --no-cache
7. 强制重新计算：python backtest_multi_stocks.py --force
8. 指定K线图目录：python backtest_multi_stocks.py --chart-dir ./my_charts

参数说明：
--------
--start: 回测开始日期，格式为YYYYMMDD
--end: 回测结束日期，格式为YYYYMMDD
--top: 每日选择的股票数量
--limit: 限制处理的股票数量，可提高速度
--all-stocks: 处理所有股票（不限制数量）
--no-cache: 禁用缓存，每次都重新计算
--force: 强制重新计算，忽略现有缓存
--chart-dir: 指定K线图存储目录

功能特性：
--------
1. 使用预处理将所有股票的计算一次性完成，避免每日重复计算
2. 使用多线程并行处理股票数据
3. 实现多级缓存，包括原始数据、预处理结果和回测结果
4. 支持限制股票数量，可快速进行初步测试
5. 生成每只交易股票的K线图，并在回测报告中添加链接
6. 计算综合盈亏比和期望收益，提供更全面的策略评估

报告指标说明：
-----------
- 综合盈亏比：所有盈利交易的平均收益率 / 所有亏损交易的平均亏损率(绝对值)
- 期望收益：胜率 * 盈亏比 - (1 - 胜率)，正值表示长期有利可图
- K线图：可点击查看每只股票的K线走势和信号位置
"""

# 添加策略目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyTT import *  # 导入MyTT模块的所有函数

# 创建缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(stock_code, start_date, end_date):
    """生成缓存文件名"""
    stock_code = extract_digits_v3(stock_code)
    cache_key = f"{stock_code}_.json"
    # filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"  # 改为json扩展名
    return os.path.join(CACHE_DIR, cache_key)

# 获取数据
def get_kline(stock_code="000001", start_date: str = "19700101",
    end_date: str = "20500101",):
    stock_code = extract_digits_v3(stock_code)
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq", start_date = start_date,
    end_date =  end_date)
    if df.empty:
        raise Exception(f"获取 {stock_code} 数据失败 Empty")
    df.rename(columns={
        '日期':'date', '开盘':'open', '收盘':'close',
        '最高':'high', '最低':'low', '成交量':'volume'
    }, inplace=True)
    df['t'] = df['date']
    df['t'] = pd.to_datetime(df['t'])
    # df = _add_blocks(df,stock_code)
    return df.set_index('t')



# def _add_blocks(df, stock_code):
#     df['name'] = stock_code  # 实际应用应从接口获取名称
   
#     return df

# 定义JSON日期序列化函数
def date_converter(obj):
    """转换日期对象为字符串，用于JSON序列化"""
    from datetime import date
    if isinstance(obj, (datetime, pd.Timestamp, date)):
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

debug = False
def get_stock_data(stock_code, start_date, end_date,exclude_code=['002304','002127','002832']):
    """
    获取股票数据(带缓存)
    
    使用 akshare 获取 A 股 K 线数据,并进行缓存
    
    测试代码:
    >>> df = get_stock_data("000001", datetime(2024,1,1), datetime(2024,4,1))
    >>> print(df.head())
    """
    try:
        # 检查缓存
        cache_file = get_cache_filename(stock_code, start_date, end_date)
        
        # 尝试从缓存读取数据
        if os.path.exists(cache_file) and not debug and stock_code not in exclude_code:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换回DataFrame并恢复日期索引
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        print(f"从缓存加载 {stock_code} 数据")
                        return df
            except Exception as e:
                print(f"读取缓存文件 {cache_file} 失败，将重新下载: {str(e)}")
                # 如果缓存文件损坏，删除它
                try:
                    os.remove(cache_file)
                    print(f"已删除损坏的缓存文件 {cache_file}")
                except:
                    pass
        
        # 获取数据
        try:
            df = get_kline(stock_code)
            if df is None or df.empty:
                print(f"获取 {stock_code} K线数据为空")
                return None
                
            # 保存到缓存
            # 先将dataframe转为字典列表，然后序列化为JSON
            df_dict = df.reset_index().to_dict(orient='records')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(df_dict, f, ensure_ascii=False, default=date_converter)
                
            # time.sleep(0.03)
            return df
        except Exception as e:
            print(f"获取 {stock_code} K线数据异常: {str(e)}")
            return None

    except Exception as e:
        print(f"获取股票 {stock_code} 数据异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def download_all_stock_data(stock_codes, start_date, end_date):
    """预先下载所有股票数据"""
    print(f"开始预下载股票数据，共 {len(stock_codes)} 只股票")
    for i, code in enumerate(stock_codes):
        try:
            print(f"下载进度: {i+1}/{len(stock_codes)} - {code}")
            get_stock_data(code, start_date, end_date)
            # 添加短间隔，避免请求过于频繁
            # time.sleep(0.2)
        except Exception as e:
            print(f"下载 {code} 数据出错: {str(e)}")
            time.sleep(2)  # 如果出错，等待更长时间
    print("所有股票数据下载完成")


def calculate_indicators(df):
    try:
        return calculate_indicators_inner(df)
    except Exception as e:
        print(f"计算技术指标异常: {str(e)}")
        return df

def calculate_indicators_inner(df):
    """计算技术指标，根据通达信代码s250415.txt实现，使用MyTT库"""
    # 准备OPEN, CLOSE, HIGH, LOW, VOL数据
    OPEN = np.array(df['open'])
    CLOSE = np.array(df['close'])
    HIGH = np.array(df['high'])
    LOW = np.array(df['low'])
    VOL = np.array(df['volume'])
    
    # 计算均线 MA
    df['MA5'] = MA(CLOSE, 5)
    df['MA10'] = MA(CLOSE, 10)
    df['MA20'] = MA(CLOSE, 20)
    df['MA60'] = MA(CLOSE, 60)
    df['MA120'] = MA(CLOSE, 120)
    
    # 计算KDJ
    N = 9
    M1 = 3
    M2 = 3
    P = 9
    
    K, D, J = KDJ(CLOSE, HIGH, LOW, N, M1, M2)
    
    # 安全处理RSV计算
    try:
        llv_val = np.array(LLV(LOW, N))
        hhv_val = np.array(HHV(HIGH, N))
        diff_val = hhv_val - llv_val
        
        rsv = np.zeros(len(CLOSE))
        for i in range(len(CLOSE)):
            if diff_val[i] > 0:  # 避免除以0
                rsv[i] = (CLOSE[i] - llv_val[i]) / diff_val[i] * 100
        
        df['RSV'] = rsv
    except Exception as e:
        print(f"RSV计算异常: {str(e)}")
        df['RSV'] = 0  # 出错时设为0
    
    df['K'] = K
    df['D'] = D
    df['J'] = J
    
    # 计算买入卖出信号 CROSS
    df['K_上穿_D'] = CROSS(K, D)
    df['D_上穿_K'] = CROSS(D, K)
    
    # 安全处理K和D的比较
    try:
        K_small_20 = np.array([k < 20 for k in K], dtype=bool)
        D_large_80 = np.array([d > 80 for d in D], dtype=bool)
    except:
        K_small_20 = np.zeros(len(K), dtype=bool)
        D_large_80 = np.zeros(len(D), dtype=bool)
    
    df['MAIRU'] = np.array(df['K_上穿_D'] & K_small_20, dtype=bool)
    df['MAICHU'] = np.array(df['D_上穿_K'] & D_large_80, dtype=bool)
    
    # VAR2到VAR9计算
    df['VAR2'] = REF(LOW, 1)
    
    # VAR3:=SMA(ABS(LOW-VAR2),3,1)/SMA(MAX(LOW-VAR2,0),3,1)*100;
    # 安全处理VAR3计算
    try:
        abs_diff = np.array(ABS(LOW - df['VAR2']))
        max_diff = np.array(MAX(LOW - df['VAR2'], 0))
        
        sma1 = np.array(SMA(abs_diff, 3, 1))
        sma2 = np.array(SMA(max_diff, 3, 1))
        
        var3 = np.zeros(len(CLOSE))
        for i in range(len(CLOSE)):
            if sma2[i] > 0:  # 避免除以0
                var3[i] = sma1[i] / sma2[i] * 100
        
        df['VAR3'] = var3
    except Exception as e:
        print(f"VAR3计算异常: {str(e)}")
        df['VAR3'] = 0  # 出错时设为0
    
    # VAR4:=EMA(IF(CLOSE*1.3,VAR3*10,VAR3/10),3);
    condition = np.array([c * 1.3 > 0 for c in CLOSE], dtype=bool)
    value = IF(condition, df['VAR3'] * 10, df['VAR3'] / 10)
    df['VAR4'] = EMA(value, 3)
    
    # VAR5:=LLV(LOW,13);
    df['VAR5'] = LLV(LOW, 13)
    
    # VAR6:=HHV(VAR4,13);
    df['VAR6'] = HHV(df['VAR4'], 13)
    
    # VAR7:=IF(MA(CLOSE,34),1,0);
    # 安全处理MA条件判断
    try:
        ma34 = np.array(MA(CLOSE, 34))
        condition = np.array([m > 0 for m in ma34], dtype=bool)
    except:
        condition = np.zeros(len(CLOSE), dtype=bool)  # 异常时设为False而不是True
    
    df['VAR7'] = IF(condition, 1, 0)
    
    # VAR8:=EMA(IF(LOW<=VAR5,(VAR4+VAR6*2)/2,0),3)/618*VAR7;
    LOW_arr = np.array(LOW)
    VAR5_arr = np.array(df['VAR5'])
    condition = np.array([l <= v5 for l, v5 in zip(LOW_arr, VAR5_arr)], dtype=bool)
    VAR4_arr = np.array(df['VAR4'])
    VAR6_arr = np.array(df['VAR6'])
    value = IF(condition, (VAR4_arr + VAR6_arr * 2) / 2, 0)
    VAR7_arr = np.array(df['VAR7'])
    
    # 安全处理VAR8计算
    try:
        ema_val = np.array(EMA(value, 3))
        var8 = np.zeros(len(CLOSE))
        for i in range(len(CLOSE)):
            var8[i] = ema_val[i] / 618 * VAR7_arr[i]
        
        df['VAR8'] = var8
    except Exception as e:
        print(f"VAR8计算异常: {str(e)}")
        df['VAR8'] = 0  # 出错时设为0
    
    # VAR9:=IF(VAR8>100,100,VAR8);
    VAR8_arr = np.array(df['VAR8'])
    condition = np.array([v8 > 100 for v8 in VAR8_arr], dtype=bool)
    df['VAR9'] = IF(condition, 100, VAR8_arr)
    
    # 下影线相关计算
    # 下影线:=(O-LOW)/REF(C,1)>(H-C)/REF(C,1)*2;
    OPEN_arr = np.array(OPEN)
    LOW_arr = np.array(LOW)
    CLOSE_arr = np.array(CLOSE)
    HIGH_arr = np.array(HIGH)
    REF_CLOSE = REF(CLOSE_arr, 1)
    
    # 安全处理下影线计算
    try:
        left_part = np.zeros(len(CLOSE))
        right_part = np.zeros(len(CLOSE))
        
        ref_close_arr = np.array(REF_CLOSE)
        for i in range(len(CLOSE)):
            if ref_close_arr[i] > 0:  # 避免除以0
                left_part[i] = (OPEN_arr[i] - LOW_arr[i]) / ref_close_arr[i]
                right_part[i] = (HIGH_arr[i] - CLOSE_arr[i]) / ref_close_arr[i] * 2
        
        df['下影线'] = np.array(left_part > right_part, dtype=bool)
    except Exception as e:
        print(f"下影线计算异常: {str(e)}")
        df['下影线'] = False  # 出错时设为False
    
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
    
    # 安全处理比率计算
    cond2 = np.zeros(len(CLOSE), dtype=bool)
    ref_close = np.array(REF(CLOSE_arr, 1))
    for i in range(len(CLOSE)):
        if ref_close[i] > 0:  # 避免除以0
            cond2[i] = CLOSE_arr[i] / ref_close[i] <= 1.03
    
    cond3 = np.array(CLOSE_arr >= OPEN_arr, dtype=bool)
    cond4 = np.array(CLOSE_arr >= REF(CLOSE_arr, 1), dtype=bool)
    
    # 安全处理前一天收盘与开盘比较
    try:
        ref_close = np.array(REF(CLOSE_arr, 1))
        ref_open = np.array(REF(OPEN_arr, 1))
        cond5 = np.array([rc <= ro for rc, ro in zip(ref_close, ref_open)], dtype=bool)
    except:
        cond5 = np.zeros(len(CLOSE), dtype=bool)  # 异常时设为False而不是True
    
    cond6 = np.array(LOW_arr < REF(CLOSE_arr, 1), dtype=bool)
    df['前一天条件'] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    
    # 真阳线:=C>O AND C/REF(C,1)>=1;
    cond1 = np.array(CLOSE_arr > OPEN_arr, dtype=bool)
    
    # 安全处理比率计算
    cond2 = np.zeros(len(CLOSE), dtype=bool)
    ref_close = np.array(REF(CLOSE_arr, 1))
    for i in range(len(CLOSE)):
        if ref_close[i] > 0:  # 避免除以0
            cond2[i] = CLOSE_arr[i] / ref_close[i] >= 1
    
    df['真阳线'] = cond1 & cond2
    
    # 下影线成功:=COUNT(REF(前一天条件,1) AND C/REF(C,1)>=1,120);
    前一天条件_arr = np.array(df['前一天条件'])
    cond1 = np.array(REF(前一天条件_arr, 1), dtype=bool)
    
    # 安全处理比率计算
    cond2 = np.zeros(len(CLOSE), dtype=bool)
    ref_close = np.array(REF(CLOSE_arr, 1))
    for i in range(len(CLOSE)):
        if ref_close[i] > 0:  # 避免除以0
            cond2[i] = CLOSE_arr[i] / ref_close[i] >= 1
    
    df['下影线成功条件'] = cond1 & cond2
    df['下影线成功'] = COUNT(df['下影线成功条件'], 120)
    
    # 下影线失败:=COUNT(REF(前一天条件,1) AND C/REF(C,1)<=1,120);
    cond1 = np.array(REF(前一天条件_arr, 1), dtype=bool)
    
    # 安全处理比率计算
    cond2 = np.zeros(len(CLOSE), dtype=bool)
    ref_close = np.array(REF(CLOSE_arr, 1))
    for i in range(len(CLOSE)):
        if ref_close[i] > 0:  # 避免除以0
            cond2[i] = CLOSE_arr[i] / ref_close[i] <= 1
    
    df['下影线失败条件'] = cond1 & cond2
    df['下影线失败'] = COUNT(df['下影线失败条件'], 120)
    
    # 下影线性价比:=下影线成功/下影线失败;
    下影线成功_arr = np.array(df['下影线成功'])
    下影线失败_arr = np.array(df['下影线失败'])
    
    # 修改下影线性价比计算逻辑：当下影线失败为0时，将性价比设为0，而不是一个很大的值
    性价比 = np.zeros(len(下影线成功_arr))
    for i in range(len(下影线成功_arr)):
        if 下影线失败_arr[i] > 0:  # 只有当失败次数大于0时才计算比率
            性价比[i] = 下影线成功_arr[i] / 下影线失败_arr[i]
        else:
            性价比[i] = 0  # 失败次数为0时，设置性价比为0
    
    df['下影线性价比'] = 性价比  # 使用新的计算方式
    
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
    df['吸筹指标'] = df['VAR9']
    # 吸筹信号
    VAR9_arr = np.array(df['VAR9'])
    去除_arr = np.array(df['去除'])
    下影线_arr = np.array(df['下影线'])
    下影线性价比_arr = np.array(df['下影线性价比'])
    
    cond1 = np.array(VAR9_arr > 5, dtype=bool)
    cond2 = np.array(去除_arr, dtype=bool)
    
    # 安全处理比率计算
    cond3 = np.zeros(len(CLOSE), dtype=bool)
    ref_close = np.array(REF(CLOSE_arr, 1))
    for i in range(len(CLOSE)):
        if ref_close[i] > 0:  # 避免除以0
            cond3[i] = CLOSE_arr[i] / ref_close[i] < 1.03
    
    cond4 = np.array(CLOSE_arr > OPEN_arr, dtype=bool)
    cond5 = np.array(CLOSE_arr >= REF(CLOSE_arr, 1), dtype=bool)
    cond6 = np.array(下影线_arr, dtype=bool)
    
    # 安全处理前一天收盘与开盘比较
    try:
        ref_close = np.array(REF(CLOSE_arr, 1))
        ref_open = np.array(REF(OPEN_arr, 1))
        cond7 = np.array([rc <= ro for rc, ro in zip(ref_close, ref_open)], dtype=bool)
    except:
        cond7 = np.zeros(len(CLOSE), dtype=bool)  # 异常时设为False而不是True
    
    cond8 = np.array(LOW_arr < REF(CLOSE_arr, 1), dtype=bool)
    cond9 = np.array(下影线性价比_arr > 2, dtype=bool)
    
    df['吸筹'] = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9
    
    return df

def get_all_stocks():
    """获取所有A股代码，并剔除未交易的股票和ST股票"""
    # 检查缓存文件
    cache_file = os.path.join(CACHE_DIR, 'stock_list.json')

    try:
        # 使用akshare获取股票列表
        stock_list = ak.stock_zh_a_spot_em()
        
        # 添加股票属性信息
        stock_list['S1'] = ~stock_list['名称'].str.contains('S')  # 非S股
        stock_list['S2'] = ~stock_list['名称'].str.contains('\*')  # 非*股
        stock_list['S4'] = ~stock_list['名称'].str.contains('科创板')  # 非科创板
        stock_list['S5'] = ~stock_list['名称'].str.contains('C')  # 非C股
        stock_list['S6'] = ~stock_list['名称'].str.contains('创业板')  # 非创业板
        stock_list['S7'] = ~stock_list['名称'].str.contains('北证50')  # 非北证50
        
        # 导出股票列表到CSV文件
        export_columns = ['代码', '名称', 'S1', 'S2', 'S4', 'S5', 'S6', 'S7']
        stock_list[export_columns].to_csv('stock_list.csv', index=False, encoding='utf-8-sig')
        print("股票列表已导出到 stock_list.csv")
        
        # 剔除未交易的股票（通过成交量和最新价判断）
        active_stocks = stock_list[
            (stock_list['成交量'] > 0) & 
            (stock_list['最新价'] > 0)
        ]
        
        # 剔除ST股票（通过名称中是否包含"ST"判断）
        non_st_stocks = active_stocks[
            ~active_stocks['名称'].str.contains('ST|st|S\.T|退') 
        ]
        
        # 使用名称过滤不同类型的股票
        non_s_stocks = non_st_stocks[~non_st_stocks['名称'].str.contains('S')]  # 非S股
        non_star_stocks = non_s_stocks[~non_s_stocks['名称'].str.contains('\*')]  # 非*股
        non_kechuang = non_star_stocks[~non_star_stocks['名称'].str.contains('科创板')]  # 非科创板
        non_c_stocks = non_kechuang[~non_kechuang['名称'].str.contains('C')]  # 非C股
        non_chuangye = non_c_stocks[~non_c_stocks['名称'].str.contains('创业板')]  # 非创业板
        non_beizheng = non_chuangye[~non_chuangye['名称'].str.contains('北证50')]  # 非北证50
        
        print(f"总股票数: {len(stock_list)}, 活跃股票数: {len(active_stocks)}, 非ST股票数: {len(non_st_stocks)}, "
              f"非S股数: {len(non_s_stocks)}, 非*股数: {len(non_star_stocks)}, "
              f"非科创板股票数: {len(non_kechuang)}, 非C股数: {len(non_c_stocks)}, "
              f"非创业板股票数: {len(non_chuangye)}, 非北证50股票数: {len(non_beizheng)}")
        
        # 提取股票代码
        codes = non_beizheng['代码'].tolist()
        
        # 缓存股票名称映射
        try:
            name_mapping = dict(zip(non_beizheng['代码'], non_beizheng['名称']))
            name_cache_file = os.path.join(CACHE_DIR, 'stock_names.json')
            with open(name_cache_file, 'w', encoding='utf-8') as f:
                json.dump(name_mapping, f, ensure_ascii=False)
        except Exception as e:
            print(f"缓存股票名称映射异常: {str(e)}")

        # 保存到缓存
        cache_data = {
            'timestamp': time.time(),
            'codes': codes
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
            
        print(f"获取到 {len(codes)} 只符合条件的股票")
        return codes

    except Exception as e:
        print(f"获取股票列表异常: {str(e)}")
        # 如果发生异常且缓存文件存在,返回过期的缓存数据
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['codes']
        return []

def process_single_stock(stock_code, date):
    """处理单个股票的数据"""
    try:
        # 设置日期范围
        end_date = date + timedelta(days=1)  # 确保包含当天
        start_date = end_date - timedelta(days=365)  # 取一年数据

        # 获取数据
        df = get_stock_data(stock_code, start_date, end_date)
        if df is None or df.empty:
            return None

        # 计算指标
        df = calculate_indicators(df)
        if df is None:
            return None

        # 找到最接近指定日期的交易日数据
        target_date = date.date()
        closest_date = None
        min_delta = float('inf')

        for idx_date in df.index:
            idx_date_only = idx_date.date()
            delta = abs((idx_date_only - target_date).days)
            if delta < min_delta:
                closest_date = idx_date
                min_delta = delta

        # 如果找不到接近的日期或差距过大，返回None
        if closest_date is None or min_delta > 3:  # 允许3天的误差
            return None

        # 获取最接近日期的数据
        target_day = df.loc[closest_date]

        # 检查是否有吸筹信号
        if target_day['吸筹']:
            return {
                '股票代码': stock_code,
                '日期': closest_date,
                '下影线性价比': target_day['下影线性价比'],
                '收盘价': target_day['close'],  # 改为小写
                '吸筹指标': target_day['吸筹指标'],
                '下影加连阳': target_day['下影加连阳'],
                # '涨跌幅': target_day['涨跌幅'],
            }
        return None
    except Exception as e:
        print(f"处理股票 {stock_code} 数据异常: {str(e)}")
        return None

def select_stocks_from_preprocessed(date, all_stock_results, top_n=10):
    """从预处理的结果中选择指定日期的前N支股票
    
    优先按照"下影加连阳"降序排序，若值相同则按"下影线性价比"降序排序，最后按"吸筹指标"降序排序
    """
    # 获取当天的股票
    date_key = date.date()
    
    if date_key not in all_stock_results:
        return pd.DataFrame()
    
    # 获取当天的结果
    day_results = all_stock_results[date_key]
    
    # 按"下影加连阳"降序排序，若值相同则按"下影线性价比"降序排序
    if day_results:
        results_df = pd.DataFrame(day_results)
        
        # 确保预处理结果中包含所需的所有字段
        required_fields = ['下影加连阳', '下影线性价比', '吸筹指标']
        missing_fields = [field for field in required_fields if field not in results_df.columns]
        if missing_fields:
            print(f"警告: 预处理数据中缺少以下字段: {', '.join(missing_fields)}")
            print(f"可用字段: {list(results_df.columns)}")
            # 如果缺少下影加连阳字段，尝试使用下影线性价比代替
            if '下影加连阳' not in results_df.columns and '下影线性价比' in results_df.columns:
                results_df['下影加连阳'] = results_df['下影线性价比']
                print("使用'下影线性价比'作为'下影加连阳'的替代")
            # 如果缺少吸筹指标字段，设置默认值0
            if '吸筹指标' not in results_df.columns:
                results_df['吸筹指标'] = 0
                print("设置'吸筹指标'默认值为0")
        
        # 打印列信息用于调试
        print(f"DataFrame列: {results_df.columns.tolist()}")
        
        # 使用多级排序
        sort_columns = []
        # 依次添加排序字段（如果存在）
        for col in ['下影加连阳', '下影线性价比', '吸筹指标']:
            # if col in results_df.columns:
            sort_columns.append(col)
        
        print(f"排序字段: {sort_columns}")
        
        # 所有字段都按降序排序
        if sort_columns:
            results_df = results_df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
        else:
            print("警告: 没有可用的排序字段，返回原始数据")
        
        results_df=results_df[results_df['吸筹指标']>5]
        
        return results_df.head(top_n).copy()
    else:
        return pd.DataFrame()

def preprocess_all_stocks(all_stocks, start_date, end_date):
    """预先计算所有股票在整个回测期间的指标
    
    返回一个字典，包含每个日期的有效股票及其指标
    """
    # 缓存文件名
    cache_key = f"preprocess_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    # 检查缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                print(f"从缓存加载预处理数据: {cache_file}")
                data = json.load(f)
                # 转换日期字符串为日期对象
                result = {}
                for date_str, stocks in data.items():
                    result[datetime.strptime(date_str, '%Y-%m-%d').date()] = [
                        {**stock, '日期': pd.to_datetime(stock['日期'])} 
                        for stock in stocks
                    ]
                return result
        except Exception as e:
            print(f"读取预处理缓存错误: {str(e)}")
            # 删除损坏的缓存
            try:
                os.remove(cache_file)
                print(f"已删除损坏的缓存文件: {cache_file}")
            except:
                pass
    
    # 预先下载所有股票数据
    print("预先下载所有股票数据...")
    download_all_stock_data(all_stocks, start_date - timedelta(days=365), end_date + timedelta(days=7))
    
    # 为每个股票计算指标
    print("开始预处理所有股票数据...")
    all_results = {}  # 按日期组织的股票数据
    
    # 使用多线程加速处理
    with ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as executor:
        future_to_stock = {
            executor.submit(preprocess_stock, stock, start_date - timedelta(days=10), end_date + timedelta(days=10)): stock 
            for stock in all_stocks
        }
        
        for future in tqdm.tqdm(future_to_stock, desc="预处理股票数据"):
            stock = future_to_stock[future]
            try:
                stock_results = future.result()
                if stock_results:
                    # 将股票结果按日期分组
                    for result in stock_results:
                        date_key = result['日期'].date()
                        if date_key not in all_results:
                            all_results[date_key] = []
                        all_results[date_key].append(result)
            except Exception as e:
                print(f"处理股票 {stock} 失败: {str(e)}")
    
    # 缓存结果
    try:
        # 将日期转换为字符串以便序列化
        cache_data = {}
        for date_key, stocks in all_results.items():
            cache_data[date_key.strftime('%Y-%m-%d')] = stocks
            
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, default=date_converter)
        print(f"已缓存预处理数据至: {cache_file}")
    except Exception as e:
        print(f"缓存预处理数据异常: {str(e)}")
    
    return all_results

def preprocess_stock(stock_code, start_date, end_date):
    """处理单个股票在整个回测期间的数据，返回所有符合条件的日期点"""
    try:
        # 检查股票代码是否符合过滤条件
      
        # 获取数据
        df = get_stock_data(stock_code, start_date, end_date)
        if df is None or df.empty:
            return []

        # 计算指标
        df = calculate_indicators(df)
        if df is None:
            return []
        
        # 找出所有有吸筹信号的日期
        signal_days = df[df['吸筹'] == True]
        if signal_days.empty:
            return []
        
        # 提取结果
        results = []
        for idx, row in signal_days.iterrows():
            # 确保收盘价是一个有效的数值
            if pd.isna(row['close']):
                print(f"警告: {stock_code} 在 {idx} 的收盘价为空")
                continue
                
            # 将收盘价转换为浮点数以确保格式一致
            
            result_item = {
                '股票代码': stock_code,
                '日期': idx,
                '下影线性价比': row['下影线性价比'],
                '下影加连阳': row['下影加连阳'],
                '收盘价': row['close'],
                'close':row['close'],
                'open':row['open'],
                '吸筹指标': row['吸筹指标'] if '吸筹指标' in row else 0,
            }
            results.append(result_item)
        
        return results
    except Exception as e:
        print(f"预处理股票 {stock_code} 异常: {str(e)}")
        return []

def get_backtest_cache_filename(start_date, end_date, top_n):
    """生成回测结果缓存文件名"""
    cache_key = f"backtest_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{top_n}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    return cache_file

def backtest_strategy(start_date, end_date, top_n=2, use_cache=True, force_recalculate=False, stock_limit=None):
    """回测策略
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        top_n: 每日选取的股票数量
        use_cache: 是否使用缓存，默认为True
        force_recalculate: 是否强制重新计算（忽略缓存），默认为False
        stock_limit: 限制处理的股票数量，默认为None（处理所有股票）
    """
    try:
        # 检查缓存
        cache_file = get_backtest_cache_filename(start_date, end_date, top_n)
        
        # 如果使用缓存且不强制重新计算，尝试从缓存加载
        if use_cache and not force_recalculate and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                    # 转换交易记录
                    trades_df = pd.DataFrame(cache_data.get('trades', []))
                    if not trades_df.empty:
                        trades_df['日期'] = pd.to_datetime(trades_df['日期'])
                        if '买入日期' in trades_df.columns:
                            trades_df['买入日期'] = pd.to_datetime(trades_df['买入日期'])
                        if '卖出日期' in trades_df.columns:
                            trades_df['卖出日期'] = pd.to_datetime(trades_df['卖出日期'])
                    
                    # 转换月度统计
                    monthly_df = pd.DataFrame(cache_data.get('monthly', []))
                    # 简化月份处理逻辑，避免使用apply转换Period
                    if not monthly_df.empty:
                        # 直接传递list而不是使用apply
                        if '月份' in monthly_df.columns:
                            try:
                                # 将月份转换为字符串，不再使用Period对象
                                monthly_df['月份_str'] = monthly_df['月份']
                                # 删除原月份列，避免转换问题
                                monthly_df = monthly_df.drop(columns=['月份'])
                                # 重命名为原列名
                                monthly_df = monthly_df.rename(columns={'月份_str': '月份'})
                            except Exception as e:
                                print(f"月份字段处理异常: {str(e)}")
                                logging.error(f"月份字段处理异常: {str(e)}", exc_info=True)
                    
                    # 获取最终资金
                    final_capital = cache_data.get('final_capital', 10000)
                    
                    print(f"从缓存加载回测结果，回测周期: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
                    return trades_df, monthly_df, final_capital
            except Exception as e:
                print(f"读取回测缓存异常: {str(e)}")
                logging.error(f"读取回测缓存异常: {str(e)}", exc_info=True)
                # 如果缓存文件损坏，删除它并继续执行回测
                try:
                    os.remove(cache_file)
                    print(f"已删除损坏的缓存文件: {cache_file}")
                except:
                    pass
        
        # 获取所有股票代码
        all_stocks = get_all_stocks()
        if stock_limit:
            all_stocks = all_stocks[:stock_limit]
            print(f"限制处理股票数量: {len(all_stocks)}")
        
        # ===== 重要改进：预先计算所有股票的指标 =====
        stock_results = preprocess_all_stocks(all_stocks, start_date, end_date)
        
        # 初始化结果
        trades_list = []  # 记录每笔交易
        initial_capital = 10000  # 初始资金
        capital = initial_capital  # 当前资金
        position_size = capital / top_n  # 每只股票的投资金额
        
        # 按日期遍历
        current_date = start_date
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
                
            # 获取当天的交易信号 (使用预处理好的数据)
            try:
                day_stocks = select_stocks_from_preprocessed(current_date, stock_results, top_n)
                
                # 如果有交易信号
                if not day_stocks.empty:
                    for _, stock in day_stocks.iterrows():
                        try:
                            # 获取股票数据，多获取几天确保有足够数据
                            stock_data = get_stock_data(stock['股票代码'], current_date - timedelta(days=5), current_date + timedelta(days=5))
                            
                            if stock_data is None or stock_data.empty:
                                print(f"警告: {stock['股票代码']} 在 {current_date} 无法获取数据，跳过该交易")
                                continue
                            
                            # 找到当前日期和次日的数据
                            try:
                                # 找到最接近当前日期的交易日及其后一个交易日
                                current_date_str = current_date.strftime('%Y-%m-%d')
                                
                                # 将索引转换为日期字符串列表，便于比较
                                dates_str = [idx.strftime('%Y-%m-%d') for idx in stock_data.index]
                                
                                # 初始化为None，确保下面的检查能捕获未赋值情况
                                current_idx = None
                                next_idx = None
                                
                                if current_date_str in dates_str:
                                    # 如果当前日期存在于数据中
                                    current_idx = dates_str.index(current_date_str)
                                    if current_idx + 1 < len(dates_str):
                                        next_idx = current_idx + 1
                                    else:
                                        print(f"警告: {stock['股票代码']} 在 {current_date} 后没有下一个交易日，跳过该交易")
                                        continue
                                else:
                                    # 找到大于当前日期的第一个交易日
                                    for i, date_str in enumerate(dates_str):
                                        if date_str > current_date_str:
                                            current_idx = i
                                            if i + 1 < len(dates_str):
                                                next_idx = i + 1
                                            break
                                
                                # 确保两个索引都有值
                                if current_idx is None or next_idx is None or next_idx >= len(dates_str):
                                    print(f"警告: {stock['股票代码']} 在 {current_date} 之后无足够交易日，跳过该交易")
                                    continue
                                
                                # 获取对应的日期对象
                                buy_date = stock_data.index[current_idx]
                                # 第二天就卖出
                                sell_date = stock_data.index[next_idx]
                                
                                # 使用当日收盘价买入
                                buy_price = stock_data['close'].iloc[current_idx]
                                # 使用次日收盘价卖出
                                sell_price = stock_data['close'].iloc[next_idx]
                                
                                # 调试信息
                                print(f"交易: {stock['股票代码']} - 信号日:{current_date.date()}, 买入日:{buy_date.date()}, 卖出日:{sell_date.date()}")
                                print(f"     买入价:{buy_price:.2f}, 卖出价:{sell_price:.2f}, 收益率:{((sell_price/buy_price)-1)*100:.2f}%")
                                
                            except Exception as e:
                                print(f"警告: 处理 {stock['股票代码']} 日期索引异常: {str(e)}")
                                continue
                            
                            # 避免零价格或负价格
                            if buy_price <= 0 or sell_price <= 0:
                                print(f"警告: {stock['股票代码']} 价格异常 (买入:{buy_price}, 卖出:{sell_price})")
                                continue
                                
                            # 计算收益
                            shares = position_size / buy_price
                            profit = shares * (sell_price - buy_price)
                            profit_rate = (sell_price - buy_price) / buy_price
                            
                            # 更新资金
                            capital += profit
                            
                            # 记录交易
                            trades_list.append({
                                '日期': current_date,  # 原始信号日期
                                '买入日期': buy_date,   # 实际买入日期
                                '卖出日期': sell_date,  # 实际卖出日期
                                '股票代码': stock['股票代码'],
                                '下影线性价比': stock['下影线性价比'],
                                '买入价': buy_price,
                                '卖出价': sell_price,
                                '收益率': profit_rate,
                                '收益金额': profit
                            })
                        except Exception as e:
                            print(f"处理股票 {stock['股票代码']} 交易时发生错误: {str(e)}")
                            logging.error(f"交易处理异常 {stock['股票代码']}: {str(e)}", exc_info=True)
                            continue
            except Exception as e:
                print(f"处理日期 {current_date.strftime('%Y-%m-%d')} 时发生错误: {str(e)}")
                logging.error(f"日期处理异常 {current_date.strftime('%Y-%m-%d')}: {str(e)}", exc_info=True)
            
            current_date += timedelta(days=1)
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(trades_list)
        
        # 按月统计
        monthly_df = pd.DataFrame()
        if not trades_df.empty:
            try:
                trades_df['月份'] = trades_df['日期'].dt.to_period('M')
                monthly_stats = []
                
                for month, month_trades in trades_df.groupby('月份'):
                    try:
                        wins = len(month_trades[month_trades['收益率'] > 0])
                        losses = len(month_trades[month_trades['收益率'] <= 0])
                        total = len(month_trades)
                        
                        # 计算盈亏比 (平均盈利/平均亏损)
                        avg_profit = month_trades[month_trades['收益率'] > 0]['收益率'].mean() if wins > 0 else 0
                        avg_loss = abs(month_trades[month_trades['收益率'] <= 0]['收益率'].mean()) if losses > 0 else 1  # 避免除零
                        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
                        
                        monthly_stats.append({
                            '月份': str(month),  # 将Period转换为字符串
                            '交易次数': total,
                            '盈利次数': wins,
                            '胜率': wins/total if total > 0 else 0,
                            '盈亏比': profit_loss_ratio,
                            '月收益率': month_trades['收益率'].mean() if not month_trades.empty else 0,
                            '月收益金额': month_trades['收益金额'].sum()
                        })
                    except Exception as e:
                        print(f"计算月度统计 {month} 时发生错误: {str(e)}")
                        logging.error(f"月度统计异常 {month}: {str(e)}", exc_info=True)
                        continue
                
                if monthly_stats:
                    monthly_df = pd.DataFrame(monthly_stats)
            except Exception as e:
                print(f"生成月度统计时发生错误: {str(e)}")
                logging.error(f"月度统计生成异常: {str(e)}", exc_info=True)
            
        # 缓存回测结果
        if use_cache:
            try:
                # 准备缓存数据
                cache_data = {
                    'trades': trades_df.to_dict(orient='records') if not trades_df.empty else [],
                    'monthly': monthly_df.to_dict(orient='records') if not monthly_df.empty else [],
                    'final_capital': capital,
                    'timestamp': time.time()
                }
                
                # 写入缓存文件
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, default=date_converter)
                print(f"已缓存回测结果至: {cache_file}")
            except Exception as e:
                print(f"缓存回测结果异常: {str(e)}")
                logging.error(f"缓存回测结果异常: {str(e)}", exc_info=True)
        
        return trades_df, monthly_df, capital
    except Exception as e:
        print(f"回测策略执行异常: {str(e)}")
        logging.error(f"回测策略异常: {str(e)}", exc_info=True)
        # 返回空结果，避免后续处理崩溃
        return pd.DataFrame(), pd.DataFrame(), 10000  # 返回初始资金

def plot_stock_kline(stock_code, start_date=None, end_date=None, chart_dir='./charts'):
    """为单只股票生成K线图，突出显示吸筹信号，保存为HTML文件"""
    try:
        # 创建图表目录
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)
            
        # 获取足够长的历史数据
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*3)
        if end_date is None:
            end_date = datetime.now()
            
        # 往前多取一年的数据，确保有足够的数据计算指标
        df = get_stock_data(stock_code, start_date - timedelta(days=365), end_date)
        if df is None or df.empty:
            print(f"获取 {stock_code} 数据失败")
            return None
            
        # 计算指标
        df = calculate_indicators(df)
        if df is None:
            print(f"计算 {stock_code} 指标失败")
            return None
        
        # 裁剪到需要显示的时间范围
        df = df.loc[df.index >= pd.to_datetime(start_date)]
        
        # 提取信号日期
        signals = df[df['吸筹']]
        
        # 创建图表 - 增加一行用于显示VAR9指标
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.15, 0.15, 0.2])
        
        # K线图
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='K线',
                                    increasing={'line': {'color': 'red'}, 'fillcolor': 'red'},
                                    decreasing={'line': {'color': 'green'}, 'fillcolor': 'green'}),
                     row=1, col=1)
        
        # 添加MA5和MA10
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], 
                                name='MA5', line=dict(color='blue', width=1)),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA10'],
                                name='MA10', line=dict(color='orange', width=1)),
                     row=1, col=1)
        
        # 吸筹信号 - 在K线上标记
        if not signals.empty:
            fig.add_trace(go.Scatter(x=signals.index,
                                    y=signals['high'] * 1.01,
                                    mode='markers',
                                    marker=dict(color='gold', size=10, symbol='triangle-down'),
                                    name='吸筹信号'),
                         row=1, col=1)
        
        # 成交量
        fig.add_trace(go.Bar(x=df.index,
                            y=df['volume'],
                            name='成交量',
                            marker_color='grey'),
                     row=2, col=1)
        
        # 下影线性价比
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['下影线性价比'],
                                name='下影线性价比',
                                line=dict(color='purple', width=1)),
                     row=3, col=1)
        
        # 添加下影线性价比的阈值线 (2)
        fig.add_shape(
            type='line',
            x0=df.index[0], x1=df.index[-1],
            y0=2, y1=2,
            line=dict(color='red', width=1, dash='dash'),
            row=3, col=1
        )
        
        fig.add_annotation(
            x=df.index[-1], y=2,
            text="阈值 2",
            showarrow=False,
            font=dict(color='red', family='Arial, SimHei'),
            align='right',
            xanchor='right',
            row=3, col=1
        )
        
        # 添加VAR9指标
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['VAR9'],
                                name='VAR9吸筹指标',
                                line=dict(color='blue', width=2)),
                     row=4, col=1)
        
        # 标记VAR9 > 5的点
        var9_signals = df[df['VAR9'] > 5]
        if not var9_signals.empty:
            fig.add_trace(go.Scatter(x=var9_signals.index,
                                    y=var9_signals['VAR9'],
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='circle'),
                                    name='VAR9 > 5'),
                         row=4, col=1)
        
        # 添加VAR9的阈值线 (5)
        fig.add_shape(
            type='line',
            x0=df.index[0], x1=df.index[-1],
            y0=5, y1=5,
            line=dict(color='red', width=1, dash='dash'),
            row=4, col=1
        )
        
        fig.add_annotation(
            x=df.index[-1], y=5,
            text="阈值 5",
            showarrow=False,
            font=dict(color='red', family='Arial, SimHei'),
            align='right',
            xanchor='right',
            row=4, col=1
        )
        
        # 获取股票名称
        try:
            stock_name = ak.stock_individual_info_em(symbol=extract_digits_v3(stock_code))['value'][0]
            stock_name = str(stock_name)  # 确保是字符串
        except Exception as e:
            print(f"获取股票名称异常: {str(e)}")
            stock_name = stock_code
            
        # 布局设置
        fig.update_layout(
            title={
                'text': f'{stock_code} {stock_name} 吸筹策略信号分析',
                'font': {'family': 'Arial, SimHei', 'size': 24}  # 使用中文兼容字体
            },
            xaxis_rangeslider_visible=False,
            height=900,  # 增加高度以适应新增的子图
            legend_title="图例",
            hovermode="x unified",
            font=dict(family='Arial, SimHei')  # 全局设置中文兼容字体
        )
        
        fig.update_xaxes(
            type='date',
            tickformat='%Y-%m-%d',
            rangeslider=dict(visible=False),
        )
        
        # 设置Y轴标题
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="下影线性价比", row=3, col=1)
        fig.update_yaxes(title_text="VAR9吸筹指标", row=4, col=1)
        
        # 保存为HTML文件
        filename = f"{chart_dir}/{stock_code}_kline.html"
        try:
            with open(f"{filename}", "w", encoding="utf8") as file:
                write_html(fig, file)
            print(f"已生成K线图: {filename}")
        except Exception as e:
            print(f"保存K线图文件异常: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return filename
    except Exception as e:
        print(f"绘制 {stock_code} K线图异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_stock_charts(trades_df, start_date=None, end_date=None, chart_dir='./charts'):
    """为交易列表中的所有股票生成K线图"""
    if trades_df is None or trades_df.empty:
        return {}
    
    # 如果未指定日期范围，则使用交易数据的日期范围
    if start_date is None and not trades_df.empty:
        start_date = trades_df['日期'].min() - timedelta(days=10)
    if end_date is None and not trades_df.empty:
        end_date = trades_df['日期'].max() + timedelta(days=10)
    
    # 确保图表目录存在
    os.makedirs(chart_dir, exist_ok=True)
    
    # 收集所有唯一的股票代码
    stock_codes = trades_df['股票代码'].unique()
    
    # 为每只股票生成K线图
    chart_files = {}
    for code in stock_codes:
        chart_file = plot_stock_kline(code, start_date, end_date, chart_dir)
        if chart_file:
            # 只保存相对路径部分，不包含目录前缀
            chart_files[code] = os.path.basename(chart_file)
    
    return chart_files

def calculate_total_profit_loss_ratio(trades_df):
    """计算综合盈亏比：所有盈利交易的平均收益率 / 所有亏损交易的平均亏损率(绝对值)"""
    if trades_df is None or trades_df.empty:
        return 0
    
    # 盈利交易
    profit_trades = trades_df[trades_df['收益率'] > 0]
    # 亏损交易
    loss_trades = trades_df[trades_df['收益率'] <= 0]
    
    # 计算平均盈利和平均亏损
    avg_profit = profit_trades['收益率'].mean() if not profit_trades.empty else 0
    avg_loss = abs(loss_trades['收益率'].mean()) if not loss_trades.empty else 1  # 避免除零，默认为1
    
    # 计算盈亏比
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
    
    return profit_loss_ratio

def generate_report(trades_df, monthly_df, final_capital, initial_capital=10000, chart_dir='./charts'):
    """生成HTML报告"""
    try:
        # 确保图表目录存在
        os.makedirs(chart_dir, exist_ok=True)
        
        # 生成K线图
        chart_files = generate_stock_charts(trades_df, chart_dir=chart_dir)
        
        # 计算综合盈亏比
        total_profit_loss_ratio = calculate_total_profit_loss_ratio(trades_df)
        
        # 获取图表目录的相对路径
        chart_rel_path = os.path.relpath(chart_dir, '.')
        
        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>多股票交易策略回测报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, 'Microsoft YaHei', SimHei, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart-container {{ width: 100%; height: 400px; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-left: 5px solid #ffeeba; margin: 10px 0; }}
                .kline-link {{ color: blue; text-decoration: underline; cursor: pointer; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>多股票吸筹交易策略回测报告</h1>
            
            <h2>回测总结</h2>
            <table>
                <tr>
                    <td>初始资金</td>
                    <td>{initial_capital:.2f}</td>
                </tr>
                <tr>
                    <td>最终资金</td>
                    <td>{final_capital:.2f}</td>
                </tr>
                <tr>
                    <td>总收益</td>
                    <td class="{'positive' if final_capital > initial_capital else 'negative'}">
                        {final_capital - initial_capital:.2f} ({(final_capital/initial_capital - 1)*100:.2f}%)
                    </td>
                </tr>
        """

        if trades_df is not None and not trades_df.empty:
            try:
                wins = len(trades_df[trades_df['收益率'] > 0])
                total = len(trades_df)
                win_rate = wins/total if total > 0 else 0
                
                html_content += f"""
                    <tr>
                        <td>总交易次数</td>
                        <td>{total}</td>
                    </tr>
                    <tr>
                        <td>盈利次数</td>
                        <td>{wins}</td>
                    </tr>
                    <tr>
                        <td>亏损次数</td>
                        <td>{total - wins}</td>
                    </tr>
                    <tr>
                        <td>总胜率</td>
                        <td>{win_rate*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>综合盈亏比</td>
                        <td>{total_profit_loss_ratio:.2f}</td>
                    </tr>
                    <tr>
                        <td>期望收益</td>
                        <td class="{'positive' if win_rate * total_profit_loss_ratio - (1 - win_rate) > 0 else 'negative'}">
                            {(win_rate * total_profit_loss_ratio - (1 - win_rate))*100:.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>平均收益率</td>
                        <td class="{'positive' if trades_df['收益率'].mean() > 0 else 'negative'}">
                            {trades_df['收益率'].mean()*100:.2f}%
                        </td>
                    </tr>
                """
            except Exception as e:
                logging.error(f"生成交易统计异常: {str(e)}", exc_info=True)
                html_content += f"""
                    <tr>
                        <td colspan="2" class="warning">生成交易统计时出错: {str(e)}</td>
                    </tr>
                """

        html_content += """
            </table>
        """

        # 月度表现
        if monthly_df is not None and not monthly_df.empty:
            try:
                html_content += """
                <h2>月度表现</h2>
                <table>
                    <tr>
                        <th>月份</th>
                        <th>交易次数</th>
                        <th>胜率 (%)</th>
                        <th>盈亏比</th>
                        <th>月收益率 (%)</th>
                        <th>月收益金额</th>
                    </tr>
                """

                for _, row in monthly_df.iterrows():
                    try:
                        profit_loss_ratio = row.get('盈亏比', 0)
                        html_content += f"""
                        <tr>
                            <td>{row['月份']}</td>
                            <td>{row['交易次数']}</td>
                            <td>{row['胜率']*100:.2f}%</td>
                            <td>{profit_loss_ratio:.2f}</td>
                            <td class="{'positive' if row['月收益率'] > 0 else 'negative'}">{row['月收益率']*100:.2f}%</td>
                            <td class="{'positive' if row['月收益金额'] > 0 else 'negative'}">{row['月收益金额']:.2f}</td>
                        </tr>
                        """
                    except Exception as e:
                        logging.error(f"生成月度行记录异常: {str(e)}", exc_info=True)
                        html_content += f"""
                        <tr>
                            <td colspan="6" class="warning">处理月度记录时出错</td>
                        </tr>
                        """
                        continue

                html_content += """
                </table>
                """
            except Exception as e:
                logging.error(f"生成月度表现异常: {str(e)}", exc_info=True)
                html_content += f"""
                <div class="warning">生成月度表现时出错: {str(e)}</div>
                """

        # 所有交易记录
        if trades_df is not None and not trades_df.empty:
            try:
                html_content += """
                <h2>所有交易记录</h2>
                <table>
                    <tr>
                        <th>信号日期</th>
                        <th>买入日期</th>
                        <th>卖出日期</th>
                        <th>股票代码</th>
                        <th>股票名称</th>
                        <th>K线图</th>
                        <th>下影线性价比</th>
                        <th>下影加连阳</th>
                        <th>吸筹指标</th>
                        <th>买入价</th>
                        <th>卖出价</th>
                        <th>收益率 (%)</th>
                        <th>收益金额</th>
                    </tr>
                """

                # 确保所有必要的列都存在
                required_columns = ['日期', '股票代码', '买入价', '卖出价', '收益率', '收益金额']
                missing_columns = [col for col in required_columns if col not in trades_df.columns]
                if missing_columns:
                    print(f"警告: 交易数据中缺少以下列: {missing_columns}")
                    html_content += f"""
                    <tr>
                        <td colspan="13" class="warning">交易数据不完整，缺少以下列: {', '.join(missing_columns)}</td>
                    </tr>
                    """
                else:
                    for _, row in trades_df.iterrows():
                        try:
                            stock_code = row['股票代码']
                            chart_filename = chart_files.get(stock_code, None)
                            
                            if chart_filename:
                                chart_url = f"{chart_rel_path}/{chart_filename}"
                                kline_link = f"""<a href="{chart_url}" target="_blank" class="kline-link">查看K线</a>"""
                            else:
                                kline_link = "无"
                            
                            # 获取股票名称
                            try:
                                stock_name = ak.stock_individual_info_em(symbol=extract_digits_v3(stock_code))['value'][0]
                                stock_name = str(stock_name)  # 确保是字符串
                            except Exception as e:
                                print(f"获取股票 {stock_code} 名称异常: {str(e)}")
                                stock_name = stock_code
                            
                            # 处理日期显示
                            signal_date = row['日期'].strftime('%Y-%m-%d')
                            
                            # 处理买入日期和卖出日期
                            buy_date = signal_date  # 默认使用信号日期
                            sell_date = "未知"  # 默认值
                            
                            if '买入日期' in row and pd.notna(row['买入日期']):
                                buy_date = row['买入日期'].strftime('%Y-%m-%d')
                            
                            if '卖出日期' in row and pd.notna(row['卖出日期']):
                                sell_date = row['卖出日期'].strftime('%Y-%m-%d')
                            
                            # 确保数值字段存在
                            下影线性价比 = row.get('下影线性价比', 0)
                            下影加连阳 = row.get('下影加连阳', 0)
                            吸筹指标 = row.get('吸筹指标', 0)
                            
                            html_content += f"""
                            <tr>
                                <td>{signal_date}</td>
                                <td>{buy_date}</td>
                                <td>{sell_date}</td>
                                <td>{stock_code}</td>
                                <td>{stock_name}</td>
                                <td>{kline_link}</td>
                                <td>{下影线性价比:.2f}</td>
                                <td>{下影加连阳:.2f}</td>
                                <td>{吸筹指标:.2f}</td>
                                <td>{row['买入价']:.2f}</td>
                                <td>{row['卖出价']:.2f}</td>
                                <td class="{'positive' if row['收益率'] > 0 else 'negative'}">{row['收益率']*100:.2f}%</td>
                                <td class="{'positive' if row['收益金额'] > 0 else 'negative'}">{row['收益金额']:.2f}</td>
                            </tr>
                            """
                        except Exception as e:
                            print(f"处理交易记录异常: {str(e)}")
                            html_content += f"""
                            <tr>
                                <td colspan="13" class="warning">处理交易记录时出错: {str(e)}</td>
                            </tr>
                            """
                            continue

                html_content += """
                </table>
                """
            except Exception as e:
                print(f"生成交易记录表异常: {str(e)}")
                html_content += f"""
                <div class="warning">生成交易记录时出错: {str(e)}</div>
                """

        html_content += f"""
        <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-left: 5px solid #ddd;">
            <p>K线图路径说明: 所有K线图保存在 <strong>{chart_dir}</strong> 目录下</p>
            <p>如果K线图链接无法正常打开，请检查:</p>
            <ol>
                <li>目录 <strong>{chart_dir}</strong> 是否存在</li>
                <li>K线图文件是否正常生成</li>
                <li>可以尝试访问相对路径 <strong>{chart_rel_path}/{list(chart_files.keys())[0] + '_kline.html' if chart_files else '股票代码_kline.html'}</strong></li>
            </ol>
        </div>
        </body>
        </html>
        """

        # 保存HTML报告
        try:
            with open('回测报告.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("回测报告已生成：回测报告.html")
        except Exception as e:
            logging.error(f"保存报告文件异常: {str(e)}", exc_info=True)
            print(f"保存报告文件时出错: {str(e)}")

        return html_content
    except Exception as e:
        logging.error(f"生成报告异常: {str(e)}", exc_info=True)
        print(f"生成报告时出错: {str(e)}")
        # 返回简单的错误报告
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>回测失败报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #d9534f; }}
                .error-box {{ background-color: #f2dede; border: 1px solid #ebccd1; 
                             padding: 15px; border-radius: 4px; color: #a94442; }}
                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow: auto; }}
            </style>
        </head>
        <body>
            <h1>回测过程出错</h1>
            <div class="error-box">
                <h3>错误信息</h3>
                <p>{str(e)}</p>
                
                <h3>回测参数</h3>
                <ul>
                    <li>开始日期: {start_date.strftime('%Y-%m-%d')}</li>
                    <li>结束日期: {end_date.strftime('%Y-%m-%d')}</li>
                    <li>选股数量: {top_n}</li>
                    <li>使用缓存: {'否' if not use_cache else '是'}</li>
                    <li>强制重算: {'是' if force_recalculate else '否'}</li>
                    <li>股票限制: {stock_limit if stock_limit is not None else '全部'}</li>
                    <li>K线图目录: {chart_dir}</li>
                </ul>
            </div>
            
            <h3>建议</h3>
            <ul>
                <li>检查日志文件 'backtest.log' 获取详细错误信息</li>
                <li>尝试减少回测时间范围</li>
                <li>减少处理的股票数量，使用 --limit 参数</li>
                <li>检查是否存在网络连接问题</li>
                <li>确保 akshare 库正常工作</li>
                <li>可以尝试使用 --force 参数强制重新计算</li>
            </ul>
        </body>
        </html>
        """
        with open('回测错误报告.html', 'w', encoding='utf-8') as f:
            f.write(error_html)
        return error_html

# 设置日志
logging.basicConfig(
    filename='backtest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    # 设置默认的回测参数（在try块外定义，确保始终可用）
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 1)  # 测试1个月
    top_n = 2  # 每天选择前2支股票
    use_cache = True  # 是否使用缓存
    force_recalculate = False  # 是否强制重新计算
    stock_limit = 8000  # 默认限制处理前 8000只股票，可提高速度
    chart_dir = './charts'  # K线图存储目录
    
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='股票回测系统')
        parser.add_argument('--start', type=str, help='回测开始日期 (YYYYMMDD)')
        parser.add_argument('--end', type=str, help='回测结束日期 (YYYYMMDD)')
        parser.add_argument('--top', type=int, help='每日选择股票数量')
        parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
        parser.add_argument('--force', action='store_true', help='强制重新计算')
        parser.add_argument('--limit', type=int, help='限制处理的股票数量')
        parser.add_argument('--all-stocks', action='store_true', help='处理所有股票（不限制数量）')
        parser.add_argument('--chart-dir', type=str, help='K线图存储目录')
        
        args = parser.parse_args()
        
        # 如果提供了命令行参数，则使用命令行参数
        if args.start:
            start_date = datetime.strptime(args.start, '%Y%m%d')
        if args.end:
            end_date = datetime.strptime(args.end, '%Y%m%d')
        if args.top:
            top_n = args.top
        if args.no_cache:
            use_cache = False
        if args.force:
            force_recalculate = True
        if args.limit:
            stock_limit = args.limit
        if args.all_stocks:
            stock_limit = None
        if args.chart_dir:
            chart_dir = args.chart_dir
            
        # 打印回测配置
        print("="*40)
        print(f"回测配置:")
        print(f"- 开始日期: {start_date.strftime('%Y-%m-%d')}")
        print(f"- 结束日期: {end_date.strftime('%Y-%m-%d')}")
        print(f"- 选股数量: {top_n}")
        print(f"- 使用缓存: {'否' if not use_cache else '是'}")
        print(f"- 强制重算: {'是' if force_recalculate else '否'}")
        print(f"- 股票限制: {stock_limit if stock_limit is not None else '全部'}")
        print(f"- K线图目录: {chart_dir}")
        print("="*40)
        
        logging.info(f"开始回测，时间范围：{start_date.date()} 至 {end_date.date()}")
        print(f"开始回测，时间范围：{start_date.date()} 至 {end_date.date()}")
        
        # 运行回测
        trades_df, monthly_df, final_capital = backtest_strategy(
            start_date, 
            end_date, 
            top_n,
            use_cache=use_cache,
            force_recalculate=force_recalculate,
            stock_limit=stock_limit
        )
        
        # 生成HTML报告
        generate_report(trades_df, monthly_df, final_capital, chart_dir=chart_dir)
        
        logging.info(f"回测完成，最终资金：{final_capital:.2f}，收益率：{(final_capital/10000-1)*100:.2f}%")
        print(f"回测完成，回测报告已生成。最终资金：{final_capital:.2f}，收益率：{(final_capital/10000-1)*100:.2f}%")
    except KeyboardInterrupt:
        logging.warning("用户中断回测过程")
        print("\n回测被用户中断")
    except MemoryError:
        logging.error("内存不足，无法完成回测", exc_info=True)
        print("\n内存不足，请尝试减少回测周期或股票数量")
    except FileNotFoundError as e:
        logging.error(f"找不到文件: {str(e)}", exc_info=True)
        print(f"\n找不到必要文件: {str(e)}")
    except PermissionError as e:
        logging.error(f"权限错误: {str(e)}", exc_info=True)
        print(f"\n文件权限错误: {str(e)}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.error(f"数据解析错误: {str(e)}", exc_info=True)
        print(f"\n数据解析错误: {str(e)}")
    except Exception as e:
        logging.error(f"回测过程出错: {str(e)}", exc_info=True)
        print(f"\n回测过程出错: {str(e)}")
        
        # 尝试生成错误报告
        try:
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>回测失败报告</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #d9534f; }}
                    .error-box {{ background-color: #f2dede; border: 1px solid #ebccd1; 
                               padding: 15px; border-radius: 4px; color: #a94442; }}
                    pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow: auto; }}
                </style>
            </head>
            <body>
                <h1>回测过程出错</h1>
                <div class="error-box">
                    <h3>错误信息</h3>
                    <p>{str(e)}</p>
                    
                    <h3>回测参数</h3>
                    <ul>
                        <li>开始日期: {start_date.strftime('%Y-%m-%d')}</li>
                        <li>结束日期: {end_date.strftime('%Y-%m-%d')}</li>
                        <li>选股数量: {top_n}</li>
                        <li>使用缓存: {'否' if not use_cache else '是'}</li>
                        <li>强制重算: {'是' if force_recalculate else '否'}</li>
                        <li>股票限制: {stock_limit if stock_limit is not None else '全部'}</li>
                        <li>K线图目录: {chart_dir}</li>
                    </ul>
                </div>
                
                <h3>建议</h3>
                <ul>
                    <li>检查日志文件 'backtest.log' 获取详细错误信息</li>
                    <li>尝试减少回测时间范围</li>
                    <li>减少处理的股票数量，使用 --limit 参数</li>
                    <li>检查是否存在网络连接问题</li>
                    <li>确保 akshare 库正常工作</li>
                    <li>可以尝试使用 --force 参数强制重新计算</li>
                </ul>
            </body>
            </html>
            """
            with open('回测错误报告.html', 'w', encoding='utf-8') as f:
                f.write(error_html)
            print("已生成错误报告: 回测错误报告.html")
        except Exception:
            pass
    finally:
        # 确保清理资源
        print("回测程序已结束")
        logging.info("回测程序结束")