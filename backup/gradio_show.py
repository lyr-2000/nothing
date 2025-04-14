import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from datetime import datetime, timedelta
import time
import os 
import json 
from pathlib import Path
import sys
sys.path.append('./strategy')

# 常量设置
INTERVAL_MAP = {
    # "1小时": "60",
    # "4小时": "240",
    "日线": "daily"
}
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "stock_list_cache.json")
CACHE_EXPIRY_HOURS = 6

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
        default_data = {"贵州茅台 (600519)": "600519"}
        write_cache(default_data)
        return default_data

stocklist = fetch_stock_list()


from strategy.s250413 import *
from strategy.Mylib import *


# 主处理函数
def analyze_stock(stock_info, interval, ):
    try:
        # 获取数据
        # stock_code = "600519"  # 贵州茅台
        # b = stock_info.split('(')[1]
        # stock_code = str(b.split(')')[0])
        stock_info = stock_info.upper()
        df = get_kline(stock_info)
       
      
        df = calculate_xi_chou(df)
        fig = plot_strategy(df.tail(120), stock_info)  # 展示最近120天
        
        return (
            fig,
           None,
        )
        
    except Exception as e:
        raise gr.Error(f"分析失败: {str(e)}")

# 创建界面
def create_gradio_ui():
    with gr.Blocks(title="智能吸筹分析系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 智能吸筹策略分析系统")
        
        with gr.Row():
            with gr.Column(scale=1):
                stock_input = gr.Dropdown(
                    label="选择股票",
                    choices=list(stocklist.keys()),
                    value="贵州茅台 (600519)"
                )
                interval_input = gr.Dropdown(
                    label="K线周期",
                    choices=list(INTERVAL_MAP.keys()),
                    value="日线"
                )
                # start_date = gr.Textbox(
                #     label="开始日期 (YYYY-MM-DD)",
                #     value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                # )
                # end_date = gr.Textbox(
                #     label="结束日期 (YYYY-MM-DD)",
                #     value=datetime.now().strftime("%Y-%m-%d")
                # )
                analyze_btn = gr.Button("开始分析", variant="primary")
            
            with gr.Column(scale=2):
                plot_output = gr.Plot()
                stats_output = gr.DataFrame(
                    headers=["数值"],
                    label="分析统计",
                    datatype=["number", "str"]
                )
        
        analyze_btn.click(
            analyze_stock,
            inputs=[stock_input, interval_input,],
            outputs=[plot_output, stats_output]
        )
    
    return app

# 启动应用
if __name__ == "__main__":
    create_gradio_ui().launch(server_port=7860)