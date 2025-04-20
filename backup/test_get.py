# 300996  | 普联软件

import akshare as ak

# ----------- 
spot_df = ak.stock_xgsglb_em(symbol="创业板")
print(spot_df)
spot_df['交易市场'] = '创业板'

# 查找股票300996的信息
print(spot_df[spot_df['股票代码'] == '300996'][['股票代码','交易市场','股票简称']])

spot_df['fullName'] = spot_df['交易市场'] + ':' + spot_df['股票简称']
import json
import os

# 确保cache目录存在
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 创建股票代码到fullName的映射字典
code_to_name = dict(zip(spot_df['股票代码'], spot_df['fullName']))

# 写入到cache/meta.json
cache_file = os.path.join(cache_dir, '创业板.json')
with open(cache_file, 'w', encoding='utf-8') as f:
    json.dump(code_to_name, f, ensure_ascii=False, indent=2)

print(f"股票代码与名称映射已保存到: {cache_file}")

spot_df = None

#  ---------------- 

spot_df = ak.stock_xgsglb_em(symbol="科创板")
print(spot_df)
spot_df['交易市场'] = '科创板'

spot_df['fullName'] = spot_df['交易市场'] + ':' + spot_df['股票简称']
import json
import os

# 确保cache目录存在
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 创建股票代码到fullName的映射字典
code_to_name = dict(zip(spot_df['股票代码'], spot_df['fullName']))

# 写入到cache/meta.json
cache_file = os.path.join(cache_dir, '科创板.json')
with open(cache_file, 'w', encoding='utf-8') as f:
    json.dump(code_to_name, f, ensure_ascii=False, indent=2)

print(f"股票代码与名称映射已保存到: {cache_file}")


