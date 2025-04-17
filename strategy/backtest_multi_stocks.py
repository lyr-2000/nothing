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
import time
import json
import logging

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
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="hfq", start_date = start_date,
    end_date =  end_date)
    if df.empty:
        raise Exception(f"获取 {stock_code} 数据失败 Empty")
    df.rename(columns={
        '日期':'date', '开盘':'open', '收盘':'close',
        '最高':'high', '最低':'low', '成交量':'volume'
    }, inplace=True)
    df['t'] = df['date']
    df['t'] = pd.to_datetime(df['t'])
    df = _add_blocks(df,stock_code)
    return df.set_index('t')



def _add_blocks(df, stock_code):
    df['name'] = stock_code  # 实际应用应从接口获取名称
    df['is_科创板'] = stock_code.startswith('688')
    df['is_创业板'] = stock_code.startswith('300')
    df['is_北证50'] = stock_code.startswith('8')
    return df

# 定义JSON日期序列化函数
def date_converter(obj):
    """转换日期对象为字符串，用于JSON序列化"""
    from datetime import date
    if isinstance(obj, (datetime, pd.Timestamp, date)):
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def get_stock_data(stock_code, start_date, end_date):
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
        if os.path.exists(cache_file):
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
            time.sleep(0.2)
        except Exception as e:
            print(f"下载 {code} 数据出错: {str(e)}")
            time.sleep(2)  # 如果出错，等待更长时间
    print("所有股票数据下载完成")

def calculate_indicators(df):
    """计算技术指标，根据通达信代码实现"""
    try:
        # 准备基本数据 - 注意列名是小写的
        OPEN = df['open'].values
        CLOSE = df['close'].values
        HIGH = df['high'].values
        LOW = df['low'].values

        # 计算前一日收盘价
        PREV_CLOSE = np.roll(CLOSE, 1)
        PREV_CLOSE[0] = np.nan

        # 计算前一日开盘价
        PREV_OPEN = np.roll(OPEN, 1)
        PREV_OPEN[0] = np.nan

        # 下影线: (开盘-最低)/前收>(最高-收盘)/前收*2
        lower_shadow = (OPEN - LOW)
        upper_shadow = (HIGH - CLOSE)

        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            lower_shadow_ratio = np.where(PREV_CLOSE > 0, lower_shadow / PREV_CLOSE, 0)
            upper_shadow_ratio = np.where(PREV_CLOSE > 0, upper_shadow / PREV_CLOSE * 2, 0)

        is_lower_shadow = lower_shadow_ratio > upper_shadow_ratio

        # X影线: MIN(收盘,开盘)-最低
        x_shadow = np.minimum(CLOSE, OPEN) - LOW

        # 上影线: 最高-MAX(收盘,开盘)
        upper_shadow_abs = HIGH - np.maximum(CLOSE, OPEN)

        # 前一天条件
        cond1 = x_shadow >= upper_shadow_abs * 2  # X影线大于上影线的2倍

        # 收盘/前收 <= 1.03
        with np.errstate(divide='ignore', invalid='ignore'):
            price_ratio = np.where(PREV_CLOSE > 0, CLOSE / PREV_CLOSE, 0)
        cond2 = price_ratio <= 1.03

        cond3 = CLOSE >= OPEN  # 收盘价大于等于开盘价
        cond4 = CLOSE >= PREV_CLOSE  # 收盘价大于等于前一天收盘价
        cond5 = PREV_CLOSE <= PREV_OPEN  # 前一天收盘价小于等于前一天开盘价
        cond6 = LOW < PREV_CLOSE  # 最低价小于前一天收盘价

        # 结合所有条件
        all_conditions = cond1 & cond2 & cond3 & cond4 & cond5 & cond6

        # 将条件数组转换为DataFrame列
        df['前一天条件'] = all_conditions

        # 使用移动窗口统计计算成功和失败次数，窗口大小120
        df['前一天条件_前一日'] = df['前一天条件'].shift(1)

        # 下影线成功条件: 前一天条件的前一日 AND 收盘/前收>=1
        df['下影线成功条件'] = (df['前一天条件_前一日']) & (price_ratio >= 1)

        # 下影线失败条件: 前一天条件的前一日 AND 收盘/前收<1
        df['下影线失败条件'] = (df['前一天条件_前一日']) & (price_ratio < 1)

        # 计算过去120天内的成功和失败次数
        df['下影线成功'] = df['下影线成功条件'].rolling(window=120).sum()
        df['下影线失败'] = df['下影线失败条件'].rolling(window=120).sum()

        # 下影线性价比: 下影线成功/下影线失败
        # 避免除零
        df['下影线性价比'] = df['下影线成功'] / (df['下影线失败'] + 1e-10)

        # 判断是否有吸筹信号
        df['下影线'] = is_lower_shadow
        df['吸筹'] = df['下影线'] & (df['下影线性价比'] > 2) & cond2 & cond3 & cond4 & cond5 & cond6

        return df
    except Exception as e:
        print(f"计算指标异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_all_stocks():
    """获取所有A股代码"""
    # 检查缓存文件
    cache_file = os.path.join(CACHE_DIR, 'stock_list.json')

    # 如果缓存文件存在且未过期(24小时),直接返回缓存数据
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < 24 * 3600:
                print("从缓存加载股票列表")
                return cache_data['codes']

    try:
        # 使用akshare获取股票列表
        stock_list = ak.stock_zh_a_spot_em()
        codes = stock_list['代码'].tolist()

        # 保存到缓存
        cache_data = {
            'timestamp': time.time(),
            'codes': codes
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)

        return codes

    except Exception as e:
        print(f"获取股票列表异常: {str(e)}")
        # 如果发生异常且缓存文件存在,返回过期的缓存数据
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['codes'][:20]
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
            }
        return None
    except Exception as e:
        print(f"处理股票 {stock_code} 数据异常: {str(e)}")
        return None

def select_stocks_for_date(date, all_stocks, top_n=2):
    """选择指定日期的前N支股票"""
    # 处理所有股票
    results = []
    for stock in tqdm.tqdm(all_stocks, desc=f"处理 {date.strftime('%Y-%m-%d')} 的股票"):
        try:
            result = process_single_stock(stock, date)
            if result:
                results.append(result)
        except Exception as e:
            print(f"处理 {stock} 异常: {str(e)}")

    # 按下影线性价比降序排序
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='下影线性价比', ascending=False)
        # 返回前N名
        return results_df.head(top_n)
    else:
        return pd.DataFrame()

def backtest_strategy(start_date, end_date, top_n=2):
    """回测策略"""
    # 获取所有股票代码
    all_stocks = get_all_stocks()
    
    # 预先下载所有股票数据
    download_all_stock_data(all_stocks[:], start_date - timedelta(days=365), end_date + timedelta(days=7))
    
    # 临时返回空结果，用于测试数据下载功能
    print("数据下载完成，回测将在完整版中运行")
    return pd.DataFrame(), pd.DataFrame(), 10000

def generate_report(trades_df, monthly_df, final_capital, initial_capital=10000):
    """生成HTML报告"""
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>多股票交易策略回测报告</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .chart-container {{ width: 100%; height: 400px; margin: 20px 0; }}
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

    if not trades_df.empty:
        html_content += f"""
            <tr>
                <td>总交易次数</td>
                <td>{len(trades_df)}</td>
            </tr>
            <tr>
                <td>盈利次数</td>
                <td>{len(trades_df[trades_df['收益率'] > 0])}</td>
            </tr>
            <tr>
                <td>亏损次数</td>
                <td>{len(trades_df[trades_df['收益率'] <= 0])}</td>
            </tr>
            <tr>
                <td>总胜率</td>
                <td>{len(trades_df[trades_df['收益率'] > 0])/len(trades_df)*100:.2f}%</td>
            </tr>
            <tr>
                <td>平均收益率</td>
                <td class="{'positive' if trades_df['收益率'].mean() > 0 else 'negative'}">
                    {trades_df['收益率'].mean()*100:.2f}%
                </td>
            </tr>
        """

    html_content += """
        </table>
    """

    # 月度表现
    if not monthly_df.empty:
        html_content += """
        <h2>月度表现</h2>
        <table>
            <tr>
                <th>月份</th>
                <th>交易次数</th>
                <th>胜率 (%)</th>
                <th>盈亏比</th>
                <th>月收益率 (%)</th>
            </tr>
        """

        for _, row in monthly_df.iterrows():
            html_content += f"""
            <tr>
                <td>{row['月份']}</td>
                <td>{row['交易次数']}</td>
                <td>{row['胜率']*100:.2f}%</td>
                <td>{row['盈亏比']:.2f}</td>
                <td class="{'positive' if row['月收益率'] > 0 else 'negative'}">{row['月收益率']*100:.2f}%</td>
            </tr>
            """

        html_content += """
        </table>
        """

    # 所有交易记录
    if not trades_df.empty:
        html_content += """
        <h2>所有交易记录</h2>
        <table>
            <tr>
                <th>日期</th>
                <th>股票代码</th>
                <th>下影线性价比</th>
                <th>买入价</th>
                <th>卖出价</th>
                <th>收益率 (%)</th>
                <th>收益金额</th>
            </tr>
        """

        for _, row in trades_df.iterrows():
            html_content += f"""
            <tr>
                <td>{row['日期'].strftime('%Y-%m-%d')}</td>
                <td>{row['股票代码']}</td>
                <td>{row['下影线性价比']:.2f}</td>
                <td>{row['买入价']:.2f}</td>
                <td>{row['卖出价']:.2f}</td>
                <td class="{'positive' if row['收益率'] > 0 else 'negative'}">{row['收益率']*100:.2f}%</td>
                <td class="{'positive' if row['收益金额'] > 0 else 'negative'}">{row['收益金额']:.2f}</td>
            </tr>
            """

        html_content += """
        </table>
        """

    html_content += """
    </body>
    </html>
    """

    # 保存HTML报告
    with open('回测报告.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_content

# 设置日志
logging.basicConfig(
    filename='backtest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    try:
        # 设置回测参数
        # 为了避免数据量过大，选择较短的回测时间段
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 1)  # 测试1个月
        top_n = 2  # 每天选择前2支股票
        
        logging.info(f"开始回测，时间范围：{start_date.date()} 至 {end_date.date()}")
        print(f"开始回测，时间范围：{start_date.date()} 至 {end_date.date()}")
        
        # 运行回测
        trades_df, monthly_df, final_capital = backtest_strategy(start_date, end_date, top_n)
        
        # 生成HTML报告
        generate_report(trades_df, monthly_df, final_capital)
        
        logging.info(f"回测完成，最终资金：{final_capital:.2f}，收益率：{(final_capital/10000-1)*100:.2f}%")
        print(f"回测完成，回测报告已生成。最终资金：{final_capital:.2f}，收益率：{(final_capital/10000-1)*100:.2f}%")
    except Exception as e:
        logging.error(f"回测过程出错: {str(e)}", exc_info=True)
        print(f"回测过程出错: {str(e)}")