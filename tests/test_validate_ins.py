import qlib
from qlib.constant import REG_CN

# 初始化 Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
print("Qlib 安装成功！")