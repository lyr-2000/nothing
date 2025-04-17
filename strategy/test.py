print("Hello World")
try:
    import pandas as pd
    import numpy as np
    import akshare as ak
    from datetime import datetime
    from Mylib import extract_digits_v3
    print("所有必需的包已成功导入")
except Exception as e:
    print(f"导入错误: {str(e)}") 