import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import tqdm
import akshare as ak  # 使用akshare获取A股数据
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import plotly.io as pio
import time
import pickle
import hashlib
import random

# 添加策略目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyTT import *  # 导入MyTT模块的所有函数

# 创建缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(stock_code, start_date, end_date):
    """生成缓存文件名"""
    cache_key = f"{stock_code}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".pkl"
    return os.path.join(CACHE_DIR, filename)

def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据(带缓存)"""
    try:
        # 检查缓存
        cache_file = get_cache_filename(stock_code, start_date, end_date)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
                print(f"从缓存加载 {stock_code} 数据")
                return df
        
        # 将股票代码转换为akshare格式
        if isinstance(stock_code, str):
            if stock_code.endswith('.SS') or stock_code.endswith('.SH'):
                ak_code = stock_code.split('.')[0]
            elif stock_code.endswith('.SZ'):
                ak_code = stock_code.split('.')[0]
            elif stock_code.startswith('6'):
                ak_code = stock_code
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                ak_code = stock_code
            else:
                ak_code = stock_code
            
            # 添加随机延迟，避免频繁请求
            time.sleep(random.uniform(0.1, 0.5))
            
            # 使用akshare获取A股日线数据
            try:
                # 转换日期格式
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')
                
                # 获取数据
                df = ak.stock_zh_a_hist(symbol=ak_code, period="daily", 
                                        start_date=start_date_str, end_date=end_date_str, 
                                        adjust="qfq")  # 前复权数据
                
                # 检查数据是否为空
                if df.empty:
                    print(f"股票 {stock_code} 数据为空")
                    return None
                
                # 重命名列以匹配原代码
                df = df.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume'
                })
                
                # 设置日期为索引
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # 检查数据是否足够
                if len(df) < 120:  # 至少需要120天数据计算指标
                    print(f"股票 {stock_code} 数据不足120天")
                    return None
                
                # 保存到缓存
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                    
                return df
                
            except Exception as e:
                print(f"使用akshare获取股票 {stock_code} 数据异常: {str(e)}")
                return None
                
        return None
    except Exception as e:
        print(f"获取股票 {stock_code} 数据异常: {str(e)}")
        return None

def download_all_stock_data(stock_codes, start_date, end_date):
    """预先下载所有股票数据"""
    print(f"开始预下载股票数据，共 {len(stock_codes)} 只股票")
    for i, code in enumerate(stock_codes):
        try:
            print(f"下载进度: {i+1}/{len(stock_codes)} - {code}")
            get_stock_data(code, start_date, end_date)
            # 添加短间隔，避免请求过于频繁
            time.sleep(0.5)
        except Exception as e:
            print(f"下载 {code} 数据出错: {str(e)}")
            time.sleep(2)  # 如果出错，等待更长时间
    print("所有股票数据下载完成")

def calculate_indicators(df):
    """计算技术指标，根据通达信代码实现"""
    try:
        # 准备基本数据
        OPEN = df['Open'].values
        CLOSE = df['Close'].values
        HIGH = df['High'].values
        LOW = df['Low'].values
        
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
    try:
        # 使用akshare获取股票列表
        stock_list = ak.stock_zh_a_spot_em()
        # 提取股票代码
        codes = stock_list['代码'].tolist()
        # 仅取前20只股票进行测试，加快回测速度
        return codes[:20]
    except Exception as e:
        # 如果akshare获取失败，使用备用列表
        print(f"获取股票列表失败: {str(e)}，使用备用列表")
        # 沪深300成分股 (仅取部分进行测试)
        hs300 = ['600000', '600015', '600016', '600018', '600019', '600025', '600028', '600029', '600030']
        return hs300

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
                '收盘价': target_day['Close'],
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
    download_all_stock_data(all_stocks, start_date - timedelta(days=365), end_date + timedelta(days=7))
    
    # 生成交易日期序列（每个工作日）
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # B表示工作日
    
    # 初始资金10000
    initial_capital = 10000
    current_capital = initial_capital
    
    # 记录交易结果
    all_trades = []
    
    # 开始回测
    for i in range(len(all_dates)-1):
        current_date = all_dates[i]
        next_date = all_dates[i+1]
        
        print(f"回测日期: {current_date.strftime('%Y-%m-%d')}")
        
        # 当日收盘前选择前N支符合条件的股票
        selected_stocks = select_stocks_for_date(current_date, all_stocks, top_n)
        
        if selected_stocks.empty:
            print(f"  没有满足条件的股票")
            continue
            
        print(f"  选出 {len(selected_stocks)} 只股票: {', '.join(selected_stocks['股票代码'].tolist())}")
            
        # 计算每只股票分配资金
        allocation_per_stock = current_capital / len(selected_stocks)
        
        # 模拟当日收盘买入，次日收盘卖出
        for _, stock in selected_stocks.iterrows():
            stock_code = stock['股票代码']
            buy_price = stock['收盘价']
            
            # 获取次日数据
            next_day_data = get_stock_data(stock_code, next_date, next_date + timedelta(days=2))
            if next_day_data is None or next_day_data.empty:
                print(f"  无法获取 {stock_code} 次日数据")
                continue
            
            # 找到最接近次日的数据
            target_date = next_date.date()
            closest_date = None
            min_delta = float('inf')
            
            for idx_date in next_day_data.index:
                idx_date_only = idx_date.date()
                delta = abs((idx_date_only - target_date).days)
                if delta < min_delta:
                    closest_date = idx_date
                    min_delta = delta
            
            if closest_date is None or min_delta > 3:  # 允许3天的误差
                print(f"  无法获取 {stock_code} 次日接近的交易数据")
                continue
                
            # 次日收盘卖出价格
            sell_price = next_day_data.loc[closest_date]['Close']
            
            # 计算收益率
            profit_rate = (sell_price / buy_price) - 1
            profit = allocation_per_stock * profit_rate
            
            # 更新资金
            current_capital += profit
            
            # 记录交易
            all_trades.append({
                '日期': current_date,
                '股票代码': stock_code,
                '下影线性价比': stock['下影线性价比'],
                '买入价': buy_price,
                '卖出价': sell_price,
                '收益率': profit_rate,
                '收益金额': profit
            })
            
            print(f"  交易 {stock_code}: 买入价 {buy_price:.2f}, 卖出价 {sell_price:.2f}, 收益率 {profit_rate*100:.2f}%")
    
    # 将交易记录转换为DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # 计算月度表现
    if not trades_df.empty:
        trades_df['月份'] = trades_df['日期'].dt.strftime('%Y-%m')
        monthly_stats = []
        
        for month, month_trades in trades_df.groupby('月份'):
            month_profit_rates = month_trades['收益率'].tolist()
            month_profit = month_trades['收益金额'].sum()
            win_count = sum(1 for r in month_profit_rates if r > 0)
            total_count = len(month_profit_rates)
            
            # 计算盈亏比
            if win_count > 0 and total_count - win_count > 0:
                avg_win = sum(r for r in month_profit_rates if r > 0) / win_count
                avg_loss = abs(sum(r for r in month_profit_rates if r <= 0) / (total_count - win_count)) if total_count - win_count > 0 else 0
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            else:
                profit_loss_ratio = 0 if win_count == 0 else float('inf')
                
            monthly_stats.append({
                '月份': month,
                '交易次数': total_count,
                '盈利次数': win_count,
                '亏损次数': total_count - win_count,
                '胜率': win_count / total_count if total_count > 0 else 0,
                '盈亏比': profit_loss_ratio,
                '月收益': month_profit,
                '月收益率': month_profit / (current_capital - month_profit) if month_profit != current_capital else 1,
            })
        
        monthly_df = pd.DataFrame(monthly_stats)
    else:
        monthly_df = pd.DataFrame()
    
    return trades_df, monthly_df, current_capital

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

if __name__ == '__main__':
    # 设置回测参数
    # 为了避免数据量过大，选择较短的回测时间段
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 2, 1)  # 测试1个月
    top_n = 2  # 每天选择前2支股票
    
    print(f"开始回测，时间范围：{start_date.date()} 至 {end_date.date()}")
    
    # 运行回测
    trades_df, monthly_df, final_capital = backtest_strategy(start_date, end_date, top_n)
    
    # 生成HTML报告
    generate_report(trades_df, monthly_df, final_capital)
    
    print(f"回测完成，回测报告已生成。最终资金：{final_capital:.2f}，收益率：{(final_capital/10000-1)*100:.2f}%") 