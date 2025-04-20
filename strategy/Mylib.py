# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    df = _add_blocks(df,stock_code)
    return df.set_index('date')


def _add_blocks(df, stock_code):
    df['name'] = stock_code  # 实际应用应从接口获取名称
    df['is_科创板'] = stock_code.startswith('688')
    df['is_创业板'] = stock_code.startswith('300')
    df['is_北证50'] = stock_code.startswith('8')
    return df

    
import pandas as pd
import numpy as np

def backtest_strategy_beta(df,printa=True):
    """
    策略回测函数
    输入要求：
    df必须包含以下列：
    - open: 开盘价
    - high: 最高价
    - close: 收盘价
    - 吸筹: 买入信号（布尔型）
    """
    
    # 生成有效信号：过滤最后一天无法交易的信号
    valid_signals = df['吸筹'] & df['open'].shift(-1).notna()
    
    # 计算交易价格
    buy_prices = df['open'].shift(-1).loc[valid_signals]    # 第二天开盘价买入
    sell_close = df['close'].shift(-1).loc[valid_signals]  # 第二天收盘价卖出
    sell_high = df['high'].shift(-1).loc[valid_signals]    # 第二天最高价卖出
    
    # 计算收益率
    returns_close = (sell_close - buy_prices) / buy_prices
    returns_high = (sell_high - buy_prices) / buy_prices
    
    # 统计指标
    stats = {
        '总交易次数': len(returns_close),
        '收盘卖出总收益率': returns_close.sum() * 100,  # 百分比显示
        '最高价卖出总收益率': returns_high.sum() * 100,
        '平均每次收益（收盘）': returns_close.mean() * 100,
        '平均每次收益（最高）': returns_high.mean() * 100,
        '胜率（收盘）': f"{returns_close.gt(0).mean() * 100:.2f}%",
        '胜率（最高）': f"{returns_high.gt(0).mean() * 100:.2f}%",
        '最大单次收益（收盘）': returns_close.max() * 100,
        '最大单次收益（最高）': returns_high.max() * 100,
        '最大单次亏损（收盘）': returns_close.min() * 100,
        '最大单次亏损（最高）': returns_high.min() * 100
    }
    
    # 生成交易明细
    trades = pd.DataFrame({
        '买入时间': buy_prices.index,
        '买入价格': buy_prices.values,
        '收盘卖出价': sell_close.values,
        '最高卖出价': sell_high.values,
        '收盘收益率%': returns_close.round(4) * 100,
        '最高收益率%': returns_high.round(4) * 100
    })
    if printa:
        # 美化统计结果打印
        print("\n\033[1;36m============= 策略回测结果 =============\033[0m")
        
        # 统计结果表格
        from tabulate import tabulate
        stats_table = [
            ["总交易次数", f"{stats['总交易次数']} 笔"],
            ["累计收益率（收盘）", f"{stats['收盘卖出总收益率']:.2f}%"],
            ["累计收益率（最高）", f"\033[1;32m{stats['最高价卖出总收益率']:.2f}%\033[0m"],
            ["平均收益率（收盘）", f"{stats['平均每次收益（收盘）']:.2f}%"],
            ["平均收益率（最高）", f"\033[1;32m{stats['平均每次收益（最高）']:.2f}%\033[0m"],
            ["胜率（收盘）", stats['胜率（收盘）']],
            ["胜率（最高）", f"\033[1;32m{stats['胜率（最高）']}\033[0m"],
            ["最大盈利（收盘）", f"{stats['最大单次收益（收盘）']:.2f}%"],
            ["最大盈利（最高）", f"\033[1;32m{stats['最大单次收益（最高）']:.2f}%\033[0m"],
            ["最大亏损（收盘）", f"\033[1;31m{stats['最大单次亏损（收盘）']:.2f}%\033[0m"],
            ["最大亏损（最高）", f"\033[1;31m{stats['最大单次亏损（最高）']:.2f}%\033[0m"]
        ]
        print(tabulate(stats_table, tablefmt="fancy_grid"))
        
        # 交易记录表格（显示前30笔）
        print("\n\033[1;36m============= 近期交易明细 =============\033[0m")
        print(trades.head(30).to_string(
            index=False,  # 不显示索引
            float_format=lambda x: f"{x:.2f}%",  # 统一格式化
            justify='center',  # 居中对齐
            columns=['买入时间', '买入价格','收盘卖出价','最高卖出价', '收盘收益率%', '最高收益率%'],
            formatters={
                '买入时间': lambda x: str(x)[:10],  # 缩短日期显示
                '买入价格': lambda x: f"    {x:.2f} ",
                '收盘卖出价': lambda x: f"    {x:.2f} ",
                '最高卖出价': lambda x: f"    {x:.2f}     ",
                '收盘收益率%': lambda x: f"\033[31m{x:.2f}%\033[0m" if x <0 else f"\033[32m{x:.2f}%\033[0m",
                '最高收益率%': lambda x: f"\033[31m{x:.2f}%\033[0m" if x <0 else f"\033[32m{x:.2f}%\033[0m"
            }
        ))
    
    return trades, stats



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def dict_to_html(b):
    pass
    # 将字典转换为 HTML 表格
    html_b = "<table>\n"
    html_b += "<tr><th>Key</th><th>Value</th></tr>\n"
    for key, value in b.items():
        html_b += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    html_b += "</table>"
    return html_b


def backtest_strategy(df,trigger_signal=['buy','sell'], printa=True,):
    """
    增强版策略回测函数
    输入要求：
    df必须包含以下列：
    - open: 开盘价
    - high: 最高价
    - close: 收盘价
    - 吸筹: 买入信号（布尔型）
    """
    buysigcol = str(trigger_signal[0])
    # 生成有效信号：过滤最后一天无法交易的信号
    valid_signals = df[buysigcol] & df['open'].shift(-1).notna()
    
    # 计算交易价格
    buy_prices = df['open'].shift(-1).loc[valid_signals]    # 第二天开盘价买入
    sell_close = df['close'].shift(-1).loc[valid_signals]  # 第二天收盘价卖出
    
    # 计算单次交易收益率
    returns_close = (sell_close - buy_prices) / buy_prices
    
     
    
    # 构建统计指标
   
    
    # 生成交易明细
    trades = pd.DataFrame({
        '买入时间': buy_prices.index,
        '买入价格': buy_prices.values.round(2),
        '卖出价格': sell_close.values.round(2),
        '持仓天数': 1,
        '收益率%': (returns_close * 100).round(2)
    })
    if len(trades) == 0:
        return None,None
    win_trades = trades["收益率%"] > 0  # 标记盈利交易
    loss_trades = trades["收益率%"] <= 0  # 标记亏损交易

    win_rate = (win_trades.sum() / len(trades) * 100).round(2)
    # avg_profit = trades.loc[win_trades, "收益率%"].mean().round(2)  # 平均盈利
    # avg_loss = trades.loc[loss_trades, "收益率%"].abs().mean().round(2)  # 平均亏损
    # 计算平均盈利和平均亏损（避免对float调用.round()）
    avg_profit = trades.loc[win_trades, "收益率%"].mean()
    avg_profit = round(avg_profit, 2) if not pd.isna(avg_profit) else 0.0  # 处理空值并四舍五入

    avg_loss = trades.loc[loss_trades, "收益率%"].abs().mean()
    avg_loss = round(avg_loss, 2) if not pd.isna(avg_loss) else 0.0  # 处理空值并四舍五入

    # 计算盈亏比（避免除零错误）
    if avg_loss == 0:
        profit_ratio = float("inf")
    else:
        profit_ratio = round(avg_profit / avg_loss, 2)

    # # 处理无亏损交易的情况（避免除零错误）
    # if avg_loss == 0:
    #     profit_ratio = float("inf")
    # else:
    #     profit_ratio = (avg_profit / avg_loss).round(2)
    stats = {
        '胜率': win_rate,
        '盈亏比': profit_ratio,
    }
   
    if printa:
        # 打印统计指标
        print("\n\033[1;36m============= 风险收益指标 =============\033[0m")
        stats_df = pd.DataFrame([stats]).T.reset_index()
        stats_df.columns = ['指标', '数值']
        print(stats_df.to_string(index=False, justify='center', 
                               formatters={'数值': lambda x: f"\033[1;34m{x}\033[0m"}))
        # 打印交易明细
        print("\n\033[1;36m============= 最近10笔交易 =============\033[0m")
        trades_display = trades.tail(10).copy()
        trades_display['买入时间'] = trades_display['买入时间'].dt.strftime('%Y-%m-%d')
        trades_display['收益率%'] = trades_display['收益率%'].apply(
            lambda x: f"\033[31m{x}%\033[0m" if x < 0 else f"\033[32m{x}%\033[0m"
        )
        print(trades_display.to_string(
            index=False,
            formatters={
                '买入价格': '{:.2f}'.format,
                '卖出价格': '{:.2f}'.format
            },
            justify='center'
        ))
    
    return trades, stats

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 生成测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31')
    np.random.seed(42)
    df = pd.DataFrame({
        'open': np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
        'high': np.cumprod(1 + np.random.normal(0.0015, 0.025, len(dates))),
        'close': np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
        'buy': np.random.choice([True, False], len(dates), p=[0.1, 0.9])
    }, index=dates)
    
    # 运行回测
    trades, stats = backtest_strategy(df)
    # fig.show()
    print('-------')
    print(trades)
    print(stats)