#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试2025年3月5日的指标数据和排序结果
"""

import os
import sys
import pandas as pd
from datetime import datetime
import json
import akshare as ak

# 添加当前目录到路径，确保能够导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入回测系统相关函数
from backtest_multi_stocks import (
    preprocess_all_stocks, 
    select_stocks_from_preprocessed,
    get_all_stocks,
    CACHE_DIR
)

# 设置Pandas显示选项，确保DataFrame打印时对齐和美观
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)        # 设置显示宽度
pd.set_option('display.float_format', '{:.4f}'.format)  # 浮点数格式
pd.set_option('display.colheader_justify', 'center')    # 列标题居中
pd.set_option('display.precision', 4)       # 小数精度


def test_daily_indicators(test_date_str="2025-03-05", top_n=10, stock_limit=None, use_active_only=True):
    """
    测试指定日期的指标数据和排序结果
    
    参数:
        test_date_str: 要测试的日期，格式为YYYY-MM-DD
        top_n: 要显示的股票数量
        stock_limit: 限制处理的股票数量
        use_active_only: 是否只使用当前活跃交易的股票
    """
    # 解析日期
    test_date = datetime.strptime(test_date_str, "%Y-%m-%d")
    
    print(f"测试日期: {test_date_str}")
    
    # 设置日期范围，只需覆盖测试日期
    start_date = test_date
    end_date = test_date
    
    # 获取股票列表 - 根据参数决定是获取所有股票还是只获取活跃股票
 
    all_stocks = get_all_stocks()
        
    if stock_limit:
        all_stocks = all_stocks[:stock_limit]
        print(f"限制处理股票数量: {len(all_stocks)}")
    
    # 加载预处理结果
    print("加载指标数据...")
    stock_results = preprocess_all_stocks(all_stocks, start_date, end_date)
    
    # 获取当日筛选结果
    print("获取当日筛选结果...")
    filtered_df = select_stocks_from_preprocessed(test_date, stock_results, top_n)
    
    # 打印结果
    if filtered_df.empty:
        print(f"警告: {test_date_str} 没有符合条件的股票")
    else:
        # 确保结果中有需要的列
        required_columns = ['股票代码', '下影线性价比']
        if all(col in filtered_df.columns for col in required_columns):
            # 尝试加载股票名称映射
            stock_names = {}
            try:
                # 尝试从缓存中加载股票名称
                name_cache_file = os.path.join(CACHE_DIR, 'stock_names.json')
                if os.path.exists(name_cache_file):
                    with open(name_cache_file, 'r', encoding='utf-8') as f:
                        stock_names = json.load(f)
            except Exception as e:
                print(f"加载股票名称缓存异常: {str(e)}")
            
            # 添加股票名称列
            filtered_df['股票名称'] = filtered_df['股票代码'].apply(
                lambda x: stock_names.get(x, "未知")
            )
            
            # 尝试获取最新股票名称
            try:
                # 获取最新的A股股票列表，包含名称
                latest_stocks = ak.stock_zh_a_spot_em()[['代码', '名称']]
                # 创建代码到名称的映射
                latest_names = dict(zip(latest_stocks['代码'], latest_stocks['名称']))
                
                # 使用最新名称更新股票名称列
                filtered_df['股票名称'] = filtered_df['股票代码'].apply(
                    lambda x: latest_names.get(x, stock_names.get(x, "未知"))
                )
            except Exception as e:
                print(f"获取最新股票名称异常: {str(e)}")
            
            # 重新排列列顺序
            columns_order = ['股票代码', '股票名称']
            if '下影加连阳' in filtered_df.columns:
                columns_order.append('下影加连阳')
            columns_order.append('下影线性价比')
            if '吸筹指标' in filtered_df.columns:
                columns_order.append('吸筹指标')
            if 'close' in filtered_df.columns:
                columns_order.append('close')
            if 'open' in filtered_df.columns:
                columns_order.append('open')
            # if '涨跌幅' in filtered_df.columns:
            #     columns_order.append('涨跌幅')
            
            # 保留所有其他列
            for col in filtered_df.columns:
                if col not in columns_order:
                    columns_order.append(col)
                    
            # 确保所有列都在DataFrame中
            available_columns = [col for col in columns_order if col in filtered_df.columns]
            
            # 打印结果
            print(f"\n{test_date_str} 筛选结果 (按下影加连阳、下影线性价比和吸筹指标降序排序):")
            print("="*100)
            
            # 为了美观打印，对DataFrame进行格式化处理
            display_df = filtered_df[available_columns].copy()
            
            # 格式化浮点数列
          
            # 尝试使用tabulate打印表格(如果已安装)
            try:
                from tabulate import tabulate
                # 将DataFrame转换为适合tabulate的格式
                print(tabulate(display_df.values.tolist(), headers=list(display_df.columns), tablefmt='pretty', showindex=False))
            except ImportError:
                # 如果没有tabulate，使用增强的pandas格式
                pd.set_option('display.expand_frame_repr', False)
                pd.set_option('display.colheader_justify', 'center')
                pd.set_option('display.unicode.ambiguous_as_wide', True)
                pd.set_option('display.unicode.east_asian_width', True)
                
                # 显示格式化后的DataFrame
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                    # 调整列宽度
                    for col in display_df.columns:
                        max_length = max(display_df[col].astype(str).map(len).max(), len(col)) + 2
                        pd.set_option(f'display.max_{col}_width', max_length)
                    
                    print(display_df.to_string(index=False))
            
            print("="*100)
            print(f"共找到 {len(filtered_df)} 支符合条件的股票")
    
    return filtered_df

if __name__ == "__main__":
    # 从命令行参数获取日期，默认为2025-03-05
    import argparse
    parser = argparse.ArgumentParser(description='测试指定日期的指标数据')
    parser.add_argument('--date', type=str, default='2024-02-05', help='要测试的日期，格式为YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=20000, help='要显示的股票数量')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的股票数量')
    parser.add_argument('--all', action='store_true', help='使用所有股票，包括未交易的')
    
    args = parser.parse_args()
    
    # 运行测试
    test_daily_indicators(args.date, args.top, args.limit, not args.all) 