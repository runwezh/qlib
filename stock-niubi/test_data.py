import qlib
from qlib.data import D

# 初始化 Qlib
qlib.init(provider_uri='/Users/zhaohua/.qlib/qlib_data/cn_data', region='cn')

# 测试单只股票
stock_id = ['sh600475']
fields = ['$close']
df = D.features(stock_id, fields, start_time='2021-01-01', end_time='2022-03-21')
print(f"\n测试股票 {stock_id} 的数据：")
print(df.head())

# 打印股票列表
print("\n可用的股票列表：")
print(D.instruments('csi500'))