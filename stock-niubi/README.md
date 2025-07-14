# 多因子选股系统

> 本项目（stock-niubi）为 Qlib 项目的一个子目录，依赖 Qlib 主项目的数据和环境。所有运行命令均在 Qlib 根目录下执行，不作为独立项目安装依赖。

## 项目结构

```
stock-niubi/
├── factor_analysis.py     # 因子分析主程序
├── stock_selector.py      # 选股程序
├── factor_analysis_results/  # 因子分析结果目录
└── stock_selection_results/  # 选股结果目录
```

## 功能特点

1. 多因子模型
   - 经典因子（Fama-French、SDF）
   - 动量因子（日内动量、动量反转）
   - 价量因子（量价背离、资金流向）
   - 场景因子（市场环境识别）

2. 目标收益
   - 短期（10天）：目标收益10%
   - 中期（60天）：目标收益30%

3. 风险控制
   - 最大回撤控制：15%
   - 波动率控制
   - 动态权重调整

## 环境准备
请确保在 Qlib 根目录下安装所有依赖，并激活相应的 Python 环境（如 micromamba 或 pip 虚拟环境）。

### 方法一：使用 pip（推荐新手使用）
```bash
pip install qlib pandas numpy scipy fire loguru tqdm ruamel.yaml
```

### 方法二：使用 micromamba（推荐高级用户使用）
```bash
# 创建新环境
micromamba create -n stock-niubi python=3.10 -c conda-forge -c defaults

# 激活环境
micromamba activate stock-niubi

# 安装基础依赖
micromamba install -c conda-forge -c defaults pandas numpy scipy

# 安装 qlib 和其他依赖（使用 pip）
pip install qlib fire loguru tqdm ruamel.yaml
```

## 数据下载
在 Qlib 根目录下执行以下命令下载数据：

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 初始化 Qlib
在 Qlib 根目录下执行以下命令初始化 Qlib：

```bash
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')"
```

## 运行因子分析
在 Qlib 根目录下执行以下命令运行因子分析：

```bash
python stock-niubi/factor_analysis.py
```

## 运行选股程序
在 Qlib 根目录下执行以下命令运行选股程序：

```bash
python stock-niubi/stock_selector.py
```

## 输出格式
### 因子分析报告
- 输出文件：`stock-niubi/factor_analysis_results/latest_config.json`
- 包含因子权重和场景权重

### 选股结果
- 输出文件：`stock-niubi/stock_selection_results/selection_YYYYMMDD_HHMMSS.json`
- 包含短期和中期选股结果

## 注意事项

1. 数据更新
   - 建议每日更新数据
   - 使用最新数据运行选股程序

2. 风险控制
   - 关注最大回撤指标
   - 定期检查因子表现
   - 及时调整配置参数

3. 性能优化
   - 可以调整回测周期
   - 可以修改股票池范围
   - 可以自定义因子权重

## 常见问题

### 环境问题
1. 检查 micromamba 是否正确安装
2. 确认环境变量设置
3. 验证包版本兼容性

### 数据问题
1. 确保数据已下载
2. 检查数据路径正确性
3. 验证数据完整性

### 运行问题
1. 检查配置文件格式
2. 确认输入参数正确
3. 查看日志输出

### 结果问题
1. 验证因子有效性
2. 检查选股逻辑
3. 分析收益表现

4. 环境问题
   - 确保 micromamba 正确安装
   - 检查环境变量设置
   - 验证包版本兼容性 