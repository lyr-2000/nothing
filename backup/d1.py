#=======================================================
#项目名称：优化版自动化交易策略
#项目说明：基于简化版策略的优化版本
#         设置3%止损，5%回撤止盈，30天强制平仓
#         加入RSI指标选择强势股，成交量过滤，长影线过滤，控制资金使用比例
#         下入两点三十后买入
#         过滤涨停板股票，避免极端市场环境交易
#==========================================================
import pandas as pd
import numpy as np
import talib as ta

# 全局变量配置
g = {
    "index": "399101.XBHS",
    "buy_count": 20,             # 最大持仓数量
    "position_ratio": 0.05,      # 单只股票仓位占总资产比例
    "max_fund_usage": 0.8,       # 最大资金使用比例
    "holdings": {},
    "available_cash": 0,
    "frozen_cash": 0,
    "trade_log": [],
    "stock_list": [],
    "market_status": "normal"    # 市场状态：normal, extreme
}

# ======================
# 初始化配置模块
# ======================
def initialize(context):
    """策略初始化配置"""
    set_benchmark("000300.XSHG")
    set_commission(commission_ratio=0.0001, min_commission=5.0)
    set_slippage(slippage=0.002)
  
    g.update({
        "buy_count": 20,             # 最大持仓数量
        "position_ratio": 0.05,      # 单只股票仓位占总资产比例
        "max_fund_usage": 0.8,       # 最大资金使用比例
    })
    
    context.universe2 = [
        '002191.SZ', '600804.SS', '603879.SS', '600190.SS', '300137.SZ', '000040.SZ', 
        '002388.SZ', '000518.SZ', '002098.SZ', '000893.SZ', '002486.SZ', '300366.SZ', 
        '600811.SS', '000070.SZ', '300044.SZ', '603825.SS', '600603.SS', '300651.SZ', 
        '002397.SZ', '300503.SZ', '002581.SZ', '600589.SS', '600624.SS', '603429.SS', 
        '300301.SZ', '600545.SS', '002656.SZ', '002313.SZ', '603880.SS', '002072.SZ', 
        '600079.SS', '002800.SZ', '600165.SS', '603843.SS', '002712.SZ', '300096.SZ', 
        '002808.SZ', '300280.SZ', '000010.SZ', '603377.SS', '603985.SS', '600753.SS', 
        '002092.SZ', '301218.SZ', '300030.SZ', '300630.SZ', '600360.SS', '002354.SZ', 
        '600387.SS', '300052.SZ', '300020.SZ', '000546.SZ', '300518.SZ', '688338.SS', 
        '300972.SZ', '300863.SZ', '000690.SZ', '002120.SZ', '830974.SZ', '603261.SS', 
        '603222.SS', '300472.SZ', '600725.SS', '300260.SZ', '002021.SZ', '600777.SS', 
        '002515.SZ', '600719.SS', '002383.SZ', '600962.SS', '600100.SS', '833266.SZ', 
        '002542.SZ', '300598.SZ', '603138.SS', '600108.SS', '000735.SZ', '600463.SS', 
        '000889.SZ', '600310.SS', '002719.SZ', '002460.SZ', '002700.SZ', '835185.SZ', 
        '300108.SZ', '600864.SS', '600696.SS', '000009.SZ', '002412.SZ', '000778.SZ', 
        '002206.SZ', '688126.SS', '603909.SS', '002668.SZ', '002371.SZ', '835368.SZ', 
        '002409.SZ', '600999.SS', '300604.SZ', '688012.SS', '002005.SZ', '600343.SS', 
        '600641.SS', '688072.SS', '600757.SS', '600703.SS', '300201.SZ', '688981.SS', 
        '600584.SS', '003040.SZ', '002104.SZ', '600182.SS', '300078.SZ', '603665.SS', 
        '002309.SZ', '002168.SZ', '002569.SZ', '002833.SZ', '600674.SS', '002574.SZ', 
        '000700.SZ', '300310.SZ', '300555.SZ', '600256.SS', '603803.SS', '300478.SZ', 
        '300600.SZ', '002321.SZ', '688656.SS', '600763.SS', '600287.SS', '600835.SS', 
        '300370.SZ', '300077.SZ', '002640.SZ', '002652.SZ', '603007.SS', '688660.SS', 
        '002366.SZ', '601999.SS', '002692.SZ', '300013.SZ', '002255.SZ', '688565.SS', 
        '600748.SS', '002642.SZ', '603669.SS', '300290.SZ', '002742.SZ', '300225.SZ', 
        '688701.SS', '600599.SS', '600872.SS', '000068.SZ', '000155.SZ', '300671.SZ', 
        '300585.SZ', '000576.SZ', '603032.SS', '000980.SZ', '000686.SZ', '002785.SZ', 
        '603588.SS', '300338.SZ', '605577.SS', '600812.SS', '000937.SZ', '300530.SZ', 
        '002564.SZ', '600221.SS', '688296.SS', '300324.SZ', '000776.SZ', '688500.SS', 
        '601198.SS', '002634.SZ', '002166.SZ', '600739.SS', '600251.SS', '600530.SS', 
        '600388.SS', '600436.SS', '300427.SZ', '002584.SZ', '600734.SS', '002748.SZ', 
        '688619.SS', '300376.SZ', '002431.SZ', '002565.SZ', '002178.SZ', '300716.SZ', 
        '002196.SZ', '300091.SZ', '601900.SS', '002555.SZ', '002310.SZ', '600958.SS', 
        '600745.SS', '300036.SZ', '002037.SZ', '002602.SZ', '300368.SZ', '601989.SS', 
        '002586.SZ', '000711.SZ', '605081.SS', '600185.SS', '688163.SS', '002413.SZ', 
        '600679.SS', '300266.SZ', '600376.SS', '600136.SS', '002193.SZ', '300899.SZ', 
        '000615.SZ', '002159.SZ', '601619.SS', '300902.SZ', '601607.SS', '000688.SZ', 
        '603326.SS', '300715.SZ', '603903.SS', '600039.SS', '601816.SS', '000566.SZ', 
        '688676.SS', '601868.SS', '000796.SZ', '300331.SZ', '300532.SZ', '600070.SS', 
        '000506.SZ', '002810.SZ', '688679.SS', '603559.SS', '000718.SZ', '600520.SS', 
        '601012.SS', '000752.SZ', '688237.SS', '600233.SS', '603023.SS', '300300.SZ', 
        '300398.SZ', '000510.SZ', '600771.SS', '603399.SS', '600884.SS', '002793.SZ', 
        '603613.SS', '002816.SZ', '300477.SZ', '002355.SZ', '300419.SZ', '002822.SZ', 
        '600831.SS', '002107.SZ', '000978.SZ', '002316.SZ', '300614.SZ', '600358.SS', 
        '002132.SZ', '600470.SS', '000868.SZ', '300200.SZ', '603778.SS', '600511.SS', 
        '603022.SS', '600246.SS', '002729.SZ', '002568.SZ', '300102.SZ', '300840.SZ', 
        '301039.SZ', '300649.SZ', '603001.SS', '002721.SZ', '603716.SS', '002554.SZ', 
        '002629.SZ', '430090.SZ', '603421.SS', '300841.SZ', '301049.SZ', '002145.SZ', 
        '601555.SS', '002689.SZ', '300165.SZ', '600030.SS', '300097.SZ', '600267.SS', 
        '002973.SZ', '600706.SS', '002131.SZ', '603019.SS', '603458.SS', '603161.SS', 
        '600715.SS', '002201.SZ', '688287.SS', '002197.SZ', '002425.SZ', '688063.SS', 
        '300242.SZ', '600711.SS', '603797.SS', '300875.SZ', '300730.SZ', '300070.SZ', 
        '000851.SZ', '603883.SS', '600606.SS', '688511.SS', '600925.SS', '300941.SZ', 
        '002273.SZ', '002360.SZ', '603995.SS', '603517.SS', '300311.SZ', '603353.SS', 
        '688553.SS', '300233.SZ', '000863.SZ', '603869.SS', '688419.SS', '000698.SZ', 
        '300128.SZ', '600926.SS', '600187.SS', '603633.SS', '601696.SS', '603687.SS', 
        '301355.SZ', '600979.SS', '300462.SZ', '300147.SZ', '002141.SZ', '600110.SS', 
        '300343.SZ', '688036.SS', '300175.SZ', '603815.SS', '600576.SS', '603959.SS', 
        '603529.SS', '000952.SZ', '002943.SZ', '300998.SZ', '834021.SZ', '000793.SZ', 
        '601995.SS', '688076.SS', '300527.SZ', '603018.SS', '002984.SZ', '600193.SS', 
        '002424.SZ', '300032.SZ', '300686.SZ', '000821.SZ', '603235.SS', '688130.SS', 
        '002789.SZ', '603003.SS', '300502.SZ', '600067.SS', '002055.SZ', '300125.SZ', 
        '300010.SZ', '839680.SZ', '600080.SS', '600462.SS', '688335.SS', '300173.SZ', 
        '300878.SZ', '000900.SZ', '600543.SS', '000903.SZ', '688053.SS', '002872.SZ', 
        '300208.SZ', '002520.SZ', '300437.SZ', '603067.SS', '300109.SZ', '002528.SZ', 
        '600200.SS', '000552.SZ', '300117.SZ', '603608.SS', '002485.SZ', '300205.SZ', 
        '000609.SZ', '600365.SS', '603363.SS', '000909.SZ', '002592.SZ', '300159.SZ', 
        '300268.SZ', '002259.SZ', '300313.SZ', '600234.SS', '000669.SZ', '000584.SZ', 
        '688282.SS', '000622.SZ', '002124.SZ', '002490.SZ', '000989.SZ', '002251.SZ', 
        '000525.SZ', '000488.SZ', '600608.SS', '300209.SZ', '600375.SS', '000809.SZ', 
        '002289.SZ', '600671.SS', '002336.SZ', '002217.SZ', '002200.SZ', '300536.SZ', 
        '002647.SZ', '002024.SZ', '603557.SS', '600289.SS', '002750.SZ', '603388.SS', 
        '600381.SS', '002052.SZ', '002199.SZ', '603828.SS', '002951.SZ', '300506.SZ', 
        '300163.SZ', '000908.SZ', '600568.SS', '002650.SZ', '000656.SZ', '600303.SS', 
        '300167.SZ', '300965.SZ', '300029.SZ', '000683.SZ', '001395.SZ', '301566.SZ', 
        '688716.SS', '301509.SZ', '301371.SZ', '833455.SZ', '839792.SZ', '838810.SZ', 
        '688459.SS', '688409.SS', '301227.SZ', '301349.SZ', '603255.SS', '301103.SZ', 
        '301130.SZ', '688267.SS', '301100.SZ', '688739.SS', '836077.SZ', '688305.SS', 
        '688276.SS', '301007.SZ', '831726.SZ', '300985.SZ', '601279.SS', '836826.SZ', 
        '003029.SZ', '300923.SZ', '688529.SS', '300900.SZ', '688378.SS', '688233.SS', 
        '688037.SS', '688011.SS', '002952.SZ', '300758.SZ', '603396.SS', '002900.SZ', 
        '603360.SS', '300597.SZ', '300573.SZ', '300510.SZ', '603866.SS', '300489.SZ', 
        '300473.SZ', '603315.SS', '603318.SS', '603567.SS', '002737.SZ', '002731.SZ', 
        '300405.SZ', '300396.SZ', '603099.SS', '603609.SS', '002698.SZ', '300293.SZ', 
        '601929.SS', '002622.SZ', '002606.SZ', '300210.SZ', '002566.SZ', '601011.SS', 
        '601880.SS', '002501.SZ', '002487.SZ', '002437.SZ', '300082.SZ', '601188.SS', 
        '601518.SS', '601106.SS', '002338.SZ', '300040.SZ', '300024.SZ', '002231.SZ', 
        '002232.SZ', '002204.SZ', '002123.SZ', '002069.SZ', '000875.SZ', '600371.SS', 
        '600593.SS', '600598.SS', '600346.SS', '600396.SS', '600356.SS', '600399.SS', 
        '600038.SS', '600333.SS', '600241.SS', '000985.SZ', '600231.SS', '600215.SS', 
        '000922.SZ', '600202.SS', '600167.SS', '000901.SZ', '000928.SZ', '600179.SS', 
        '600178.SS', '600189.SS', '000881.SZ', '600125.SS', '600148.SS', '000898.SZ', 
        '000761.SZ', '000818.SZ', '600095.SS', '000751.SZ', '000800.SZ', '000766.SZ', 
        '000715.SZ', '000692.SZ', '000059.SZ', '000679.SZ', '000661.SZ', '000638.SZ', 
        '600758.SS', '000623.SZ', '600742.SS', '000420.SZ', '000410.SZ', '600726.SS', 
        '600718.SS', '000597.SZ', '600867.SS', '600853.SS', '600829.SS', '000545.SZ', 
        '000530.SZ', '600694.SS', '000030.SZ', '600705.SS', '600795.SS', '600697.SS', 
        '600609.SS', '600666.SS', '600653.SS', '600664.SS', '600881.SS', '300561.SZ',
        '002898.SZ', '603931.SS', '603131.SS', '603200.SS', '603371.SS', '002580.SZ'
    ]  # 风险股票池
    
    if not is_trade():
        set_backtest()

# ======================
# 回测环境配置
# ======================
def set_backtest():
    """设置回测专用参数"""
    set_limit_mode("UNLIMITED")

# ======================
# 日志格式统一
# ======================
def log_info(msg):
    """普通信息日志"""
    print("[INFO] %s" % msg)

def log_warning(msg):
    """警告信息日志"""
    print("[WARNING] %s" % msg)

def log_error(msg):
    """错误信息日志"""
    print("[ERROR] %s" % msg)

# ======================
# 功能：股票筛选模块
# ======================       
def filter_risk_stocks(all_stocks, trade_date):
    """
    过滤ST/停牌/退市/退市整理期股票
    """
    # 获取股票的状态ST、停牌、退市
    st_status = get_stock_status(all_stocks, 'ST')
    halt_status = get_stock_status(all_stocks, 'HALT')
    delisting_status = get_stock_status(all_stocks, 'DELISTING')
    
    # 将三种状态的股票剔除当日的股票池
    for stock in all_stocks.copy():
        if st_status[stock] or halt_status[stock] or delisting_status[stock]:
            all_stocks.remove(stock)

# ======================
# 过滤涨停/跌停股票
# ======================
def filter_limit_up_down(stock_list, trade_date):
    """过滤涨停/跌停的股票"""
    filtered_stocks = []
    
    for stock in stock_list:
        try:
            # 获取前一日收盘价
            hist_data = get_history(
                security=stock,
                end_date=trade_date,
                frequency='1d',
                field=['close', 'high', 'low'],
                count=2
            )
            
            if len(hist_data) < 2:
                continue
                
            prev_close = hist_data['close'].values[0]
            today_high = hist_data['high'].values[1]
            today_low = hist_data['low'].values[1]
            
            # 简单计算涨停价和跌停价（假设10%涨跌幅限制）
            limit_up_price = round(prev_close * 1.1, 2)
            limit_down_price = round(prev_close * 0.9, 2)
            
            # 判断是否涨停或跌停
            is_limit_up = abs(today_high - limit_up_price) < 0.01
            is_limit_down = abs(today_low - limit_down_price) < 0.01
            
            if not is_limit_up and not is_limit_down:
                filtered_stocks.append(stock)
                
        except Exception as e:
            log_error("检查涨跌停 %s 失败: %s" % (stock, str(e)))
            
    return filtered_stocks
    
# ======================
# 筛选具有长下影线形态的股票
# ======================
def filter_long_lower_shadow(stock_list, trade_date, shadow_ratio=2.0):
    filtered_stocks = []
    
    for stock in stock_list:
        try:
            # 获取K线数据
            kdata = get_price(
                security=stock,
                end_date=trade_date,
                frequency='1d',
                fields=['open', 'high', 'low', 'close'],
                count=1
            )
            
            open_price = kdata['open'].values[0]
            high_price = kdata['high'].values[0]
            low_price = kdata['low'].values[0]
            close_price = kdata['close'].values[0]
            
            # 计算实体长度和下影线长度
            body_length = abs(close_price - open_price)
            
            # 确定实体底部价格
            body_bottom = min(open_price, close_price)
            
            # 计算下影线长度
            lower_shadow = body_bottom - low_price
            
            # 判断是否为长下影线(下影线长度 > 实体长度的shadow_ratio倍)
            if body_length > 0 and lower_shadow > body_length * shadow_ratio:
                # 确保收盘价接近最高价
                if (high_price - close_price) / (high_price - low_price) < 0.3:
                    filtered_stocks.append(stock)
        
        except Exception as e:
            log_error("长下影线检测 %s 失败: %s" % (stock, str(e)))
    
    return filtered_stocks

# ======================
# 流动性过滤（成交量）
# ======================
def filter_by_volume(stock_list, trade_date, min_vol_ratio=1.0):
    """过滤流动性不足的股票，要求近期成交量大于历史平均"""
    filtered_stocks = []
    
    for stock in stock_list:
        try:
            # 获取历史成交量
            hist_data = get_price(
                security=stock,
                end_date=trade_date,
                frequency='1d',
                fields=['volume'],
                count=30  # 获取30天数据
            )
            
            if len(hist_data) < 20:  # 至少需要20天数据
                continue
                
            volumes = hist_data['volume'].values
            
            # 计算近5日平均成交量和20日平均成交量
            recent_avg_vol = np.mean(volumes[-5:])
            long_avg_vol = np.mean(volumes)
            
            # 近期成交量大于历史平均的min_vol_ratio倍
            if recent_avg_vol > long_avg_vol * min_vol_ratio:
                filtered_stocks.append(stock)
                
        except Exception as e:
            log_error("成交量过滤 %s 失败: %s" % (stock, str(e)))
            
    return filtered_stocks

# ======================
# 相对强度(RSI)选股
# ======================
def filter_by_rsi(stock_list, trade_date, rsi_period=14, lower_bound=30, upper_bound=70):
    """使用RSI指标选择强势股票"""
    filtered_stocks = []
    
    for stock in stock_list:
        try:
            # 获取历史收盘价
            hist_data = get_price(
                security=stock,
                end_date=trade_date,
                frequency='1d',
                fields=['close'],
                count=30  # RSI计算需要一定历史数据
            )
            
            if len(hist_data) < rsi_period + 5:  # 确保有足够数据计算RSI
                continue
                
            closes = hist_data['close'].values
            
            # 使用talib计算RSI
            rsi = ta.RSI(closes, timeperiod=rsi_period)
            current_rsi = rsi[-1]
            
            # 筛选RSI在特定范围内的股票（避免超买超卖区域）
            if lower_bound <= current_rsi <= upper_bound:
                filtered_stocks.append(stock)
                
        except Exception as e:
            log_error("RSI筛选 %s 失败: %s" % (stock, str(e)))
            
    return filtered_stocks

# ======================
# 市场环境判断
# ======================
def check_market_condition(trade_date):
    """判断市场环境是否极端"""
    try:
        # 获取指数数据（以沪深300为例）
        index_code = '000300.XSHG'
        hist_data = get_price(
            security=index_code,
            end_date=trade_date,
            frequency='1d',
            fields=['close'],
            count=20  # 获取20天数据
        )
        
        if len(hist_data) < 20:
            return "normal"  # 数据不足时默认为正常
            
        closes = hist_data['close'].values
        
        # 计算20日涨跌幅
        change_pct = (closes[-1] / closes[0] - 1) * 100
        
        # 判断极端情况：大涨大跌超过10%
        if abs(change_pct) > 10:
            log_warning("检测到极端市场环境，涨跌幅: %.2f%%" % change_pct)
            return "extreme"
        
        return "normal"
        
    except Exception as e:
        log_error("市场环境检查失败: %s" % str(e))
        return "normal"  # 出错时默认正常

# ======================
# 盘前处理主函数 - 优化版
# ======================
def before_trading_start(context, data):
    """盘前数据处理（9:15-9:30）"""
    try:
        g["target_stocks"] = []  # 每日初始化
        trade_date = context.previous_date.strftime("%Y-%m-%d")
        
        # 判断市场环境
        g["market_status"] = check_market_condition(trade_date)
        if g["market_status"] == "extreme":
            log_warning("市场环境极端，今日暂停交易")
            return
            
        # 获取全部A股
        all_stocks = get_Ashares()
        
        # 过滤手动添加的风险票
        all_stocks = [s for s in all_stocks if s not in context.universe2]
        log_info("原始股票池数量：%d" % len(all_stocks))
        
        # 过滤风险股票 ST 停牌 退市整理期间
        filter_risk_stocks(all_stocks, trade_date)
        safe_stocks = all_stocks
        log_info("过滤风险股后股票数量：%d" % len(safe_stocks))   
        
        # 筛选出60 00开头的票适合新手账号运行
        g["stock_list"] = [s for s in safe_stocks if s.startswith(('60', '00'))]
        log_info("筛选主板股票后数量：%d" % len(g["stock_list"]))
        
        # 过滤涨停跌停股
        #g["stock_list"] = filter_limit_up_down(g["stock_list"], trade_date)
        #log_info("过滤涨跌停后股票数量：%d" % len(g["stock_list"]))
        
        # 成交量过滤
        g["stock_list"] = filter_by_volume(g["stock_list"], trade_date, min_vol_ratio=1.2)
        log_info("成交量过滤后股票数量：%d" % len(g["stock_list"]))
        
        g["stock_list"] = filter_long_lower_shadow(g["stock_list"], trade_date, shadow_ratio=2.0)
        log_info("长下影线筛选后股票数量：%d" % len(g["stock_list"]))
        
        # RSI指标筛选
        g["target_stocks"] = filter_by_rsi(g["stock_list"], trade_date, rsi_period=14, lower_bound=40, upper_bound=65)
        log_info("RSI筛选后股票数量：%d" % len(g["target_stocks"]))
        
        # 如果筛选后股票数量不足，使用成交量过滤后的股票池
        if len(g["target_stocks"]) < 20:
            g["target_stocks"] = g["stock_list"][:min(len(g["stock_list"]), 100)]
            log_info("筛选股票不足，使用备选股票池")
        
        log_info("今日适合开仓股票数：%d" % len(g["target_stocks"]))
        log_info("适合操作股票清单(前10只)：%s" % g["target_stocks"][:10])
          
    except Exception as e:
        log_error("盘前处理异常: %s" % str(e))
        g["target_stocks"] = []  # 确保当日无操作

# ======================
# 交易核心模块 - 设置每日14:30后交易
# ======================
def handle_data(context, data):
    """交易日执行"""
    log_info("检查交易条件...")
    
    # 获取当前时间
    current_time = context.current_dt.time()
    trading_start_time = pd.Timestamp('14:30:00').time()
    
    # 如果当前时间早于14:30，跳过交易
    if current_time < trading_start_time:
        log_info("当前时间 %s 早于交易时间 14:30，暂不交易" % current_time.strftime('%H:%M:%S'))
        return
    
    # 如果市场环境极端，跳过交易
    if g["market_status"] == "extreme":
        log_info("市场环境极端，跳过交易")
        return
    
    log_info("开始执行下午交易...")
    
    # 执行交易逻辑
    update_holdings(context)
    execute_sell(context, data)
    execute_buy(context, data)

# ======================
# 持仓管理模块
# ======================
def update_holdings(context):
    """更新持仓状态"""
    log_info("更新持仓状态...")
    for stock, position in context.portfolio.positions.items():
        # 字段正确性验证
        required_fields = {
            'sid': position.sid,  # 持仓股票名称
            'enable_amount': position.enable_amount,  # 可用数量
            'amount': position.amount,  # 总的持仓数量
            'last_sale_price': position.last_sale_price,  # 最新价格
            'cost_basis': position.cost_basis  # 持仓成本价格
        }
        
        # 装入g["holdings"]函数。存入持仓所有信息。
        if stock not in g["holdings"]:
            g["holdings"][stock] = {
                'cost_price': required_fields['cost_basis'],  # 持仓初始成本
                'highest_price': required_fields['last_sale_price'],  # 历史最高价格
                'hold_days': 0,  # 持股天数
                'position_detail': required_fields  # 持仓详情
            }
        else:
            g["holdings"][stock]['highest_price'] = max(
                g["holdings"][stock]['highest_price'],
                required_fields['last_sale_price']
            )
            g["holdings"][stock]['hold_days'] += 1  # 持仓天数加一天

# ======================
# 卖出模块 - 优化参数
# ======================
def execute_sell(context, data):
    """动态止盈止损策略(利润回撤5%或亏损3%)"""
    log_info("检查卖出条件...")
    for stock in list(context.portfolio.positions.keys()):
        try:
            position = context.portfolio.positions[stock]
            
            # 获取持仓关键数据
            cost_price = position.cost_basis          # 持仓成本价
            current_price = position.last_sale_price  # 最新成交价
            highest_price = g["holdings"][stock]['highest_price']  # 持仓期间最高价
            hold_days = g["holdings"][stock]['hold_days']
            
            # 计算收益率和回撤
            return_rate = (current_price - cost_price) / cost_price
            drawdown_from_peak = (highest_price - current_price) / highest_price
            
            # 调试信息
            debug_msg = (
                "%s 成本:%.2f 现价:%.2f 最高:%.2f 天数:%d "
                "收益率:%.1f%% 回撤:%.1f%%" % 
                (stock, cost_price, current_price, highest_price, 
                 hold_days, return_rate*100, drawdown_from_peak*100)
            )
            log_info(debug_msg)
            
            # 触发条件
            sell_reason = []
            if return_rate <= -0.03:  # 亏损3%止损
                sell_reason.append("达到3%止损线")
            if return_rate > 0 and drawdown_from_peak >= 0.05:  # 利润回撤5%止盈
                sell_reason.append("利润回撤5%")
                
            # 执行卖出
            if sell_reason and position.enable_amount > 0:
                order_target(stock, 0)
                log_info("卖出 %s 原因: %s" % (stock, "+".join(sell_reason)))
                del g["holdings"][stock]
                
            # 强制平仓规则
            if hold_days >= 30:  # 持仓超过30天强制卖出
                order_target(stock, 0)
                log_info("卖出 %s 原因: 持仓超30天" % stock)
                del g["holdings"][stock]
                
        except KeyError as e:
            log_error("持仓数据缺失 %s: %s" % (stock, str(e)))
        except Exception as e:
            log_error("卖出异常 %s: %s" % (stock, str(e)))

# ======================
# 买入模块 - 优化资金管理
# ======================
def execute_buy(context, data):
    """买入操作（限制单票5%仓位，总持仓≤20只，最多使用80%资金）"""
    log_info("检查买入条件...")
    # 获取当前总资产（现金+持仓价值）
    total_value = context.portfolio.portfolio_value
    
    # 计算当前有效持仓数量
    current_hold = sum(1 for p in context.portfolio.positions.values() if p.enable_amount > 0)
    
    # 持仓已达上限则停止买入
    if current_hold >= g["buy_count"]:
        log_info("持仓已达%d只上限，停止买入" % g["buy_count"])
        return
    
    # 计算当前已用资金比例
    current_positions_value = total_value - context.portfolio.cash
    used_fund_ratio = current_positions_value / total_value
    
    # 如果已用资金超过最大使用比例，停止买入
    if used_fund_ratio >= g["max_fund_usage"]:
        log_info("已用资金比例%.2f%%超过最大限制%.2f%%，停止买入" % 
                (used_fund_ratio*100, g["max_fund_usage"]*100))
        return
    
    # 生成可买列表（排除已持仓）
    buy_list = [s for s in g["target_stocks"] if s not in context.portfolio.positions]
    
    # 计算实际可买数量（考虑剩余仓位）
    max_buy_num = min(g["buy_count"] - current_hold, len(buy_list))
    if max_buy_num <= 0:
        return
    
    # 资金分配逻辑
    for stock in buy_list[:max_buy_num]:
        try:
            # === 价格获取 ===
            current_price = data[stock].last
            if np.isnan(current_price) or current_price <= 0:
                log_info("%s 价格无效: %s" % (stock, current_price))
                continue
            
            # === 计算最大可买金额 ===
            # 单票最大金额 = 总资产 * 单只仓位比例
            max_amount_by_value = total_value * g["position_ratio"]
            # 实际可用资金 = min(账户现金, 最大可买金额)
            available_cash = min(context.portfolio.cash, max_amount_by_value)
            
            # === 计算可买数量 ===
            # 计算考虑手续费后的最大数量（佣金按最低5元计算）
            commission = max(current_price * 100 * 0.0003, 5)
            max_shares = int((available_cash - commission) // (current_price * 100)) * 100
            
            if max_shares >= 100:
                # 发送市价单
                order_id = order(
                    security=stock,
                    amount=max_shares
                )
                log_info("委托成功 %s 数量:%d 价格:%.2f 金额:%.2f" % (stock, max_shares, current_price, current_price * max_shares))
                # 更新当前持仓计数
                current_hold += 1
                if current_hold >= g["buy_count"]:
                    log_info("持仓已达%d只，终止本轮买入" % g["buy_count"])
                    break
            else:
                log_info("资金不足 %s 需%.2f 可用:%.2f" % (
                    stock, 
                    current_price * 100, 
                    available_cash
                ))     
        except Exception as e:
            log_error("下单异常 %s: %s" % (stock, str(e)))

# ======================
# 盘后处理模块
# ======================
def after_trading_end(context, data):
    """每日收盘后执行，打印账户概况"""
    total_value = context.portfolio.portfolio_value
    
    # 账户总览
    log_info("=" * 50)
    log_info("【每日账户报告】日期：%s" % context.current_dt.strftime('%Y-%m-%d'))
    log_info("总市值：%.2f元" % total_value)
    cash_ratio = context.portfolio.cash / total_value * 100
    log_info("可用资金：%.2f元 (%.2f%%)" % (context.portfolio.cash, cash_ratio))
    log_info("=" * 50)
    
    # 持仓检查
    if not g.get("holdings"):
        log_info("当前无持仓股票")
        return
    
    # 构建表头
    log_info("%-10s %-8s %-10s %-10s %-10s" % 
            ("股票代码", "持仓天数", "成本价", "当前价", "盈亏率"))
    log_info("-" * 50)
    
    # 遍历持仓数据
    for stock in g["holdings"]:
        try:
            holding = g["holdings"][stock]
            cost_price = holding['cost_price']
            hold_days = holding['hold_days']
            
            # 获取收盘价（更稳定）
            current_price = data[stock].close
            
            # 计算盈亏率
            profit_ratio = 0.0
            if cost_price != 0:
                profit_ratio = ((current_price - cost_price) / cost_price) * 100
            
            # 格式化输出
            log_line = "%-10s %-8d %-10.2f %-10.2f %+7.2f%%" % (
                stock, hold_days, cost_price, current_price, profit_ratio
            )
            log_info(log_line)
            
        except KeyError as e:
            log_error("股票%s数据字段缺失：%s" % (stock, str(e)))
        except Exception as e:
            log_error("处理%s时发生未知错误：%s" % (stock, str(e)))