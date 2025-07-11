# Qlib Examples 细致学习路线

---

## 1. 基础认知与环境验证

### 1.1 阅读主项目文档
- 先通读 Qlib 官方文档的"快速开始"、"数据准备"、"工作流"等章节，理解整体架构。

### 1.2 验证环境
- 跑通 `examples/workflow_by_code.py`，确保数据、依赖、环境都没问题。
- 理解代码中每一步（数据加载、特征工程、模型训练、回测、评估）的作用。

---

## 2. 因子与模型 benchmark 深入

### 2.1 经典机器学习模型
- 进入 `examples/benchmarks/LightGBM`，运行 LightGBM 示例，理解配置文件和输出结果。
- 依次尝试 `XGBoost`、`CatBoost`、`Linear`、`MLP` 等子目录，比较不同模型的表现和配置差异。

### 2.2 深度学习模型
- 进入 `ALSTM`、`LSTM`、`GRU`、`TFT`、`TabNet` 等目录，运行并分析深度学习模型的训练与回测流程。
- 关注模型输入特征、网络结构、超参数设置。

### 2.3 结果对比与总结
- 整理各模型的回测指标（如IC、IR、收益率、回撤等），形成对比表格。
- 思考不同模型适用场景和优缺点。

---

## 3. 数据处理与扩展能力

### 3.1 数据缓存与内存管理
- 阅读并运行 `examples/data_demo/data_cache_demo.py`、`data_mem_resuse_demo.py`，理解 Qlib 的数据缓存机制。

### 3.2 高频数据处理
- 进入 `examples/highfreq/`，学习高频数据的加载、处理、特征构建方法。

### 3.3 盘口数据
- 阅读 `orderbook_data/` 下的 `create_dataset.py`、`example.py`，了解如何处理 Level2 盘口数据。

---

## 4. 组合管理与风险控制

### 4.1 指数增强与风险管理
- 进入 `portfolio/`，分析 `config_enhanced_indexing.yaml` 和 `prepare_riskdata.py`，学习如何做指数增强和风险数据准备。

### 4.2 组合回测
- 结合主项目的 `qlib/backtest/` 代码，理解回测引擎的实现原理。

---

## 5. 模型解释与可视化

### 5.1 特征重要性分析
- 阅读 `model_interpreter/feature.py`，学习如何解释模型结果、分析特征贡献。

---

## 6. 强化学习与高级策略

### 6.1 强化学习基础
- 进入 `rl/`、`rl_order_execution/`，阅读 README 和示例脚本，理解 RL 在量化中的应用场景。

### 6.2 订单执行与策略优化
- 跑通 `rl_order_execution/scripts/` 下的示例，理解 RL 在订单执行中的优势。

---

## 7. 进阶与创新

### 7.1 自定义因子与模型
- 尝试在 `benchmarks/` 目录下新建自己的模型配置，或修改现有因子，观察回测结果变化。

### 7.2 结合自己的数据
- 用 `scripts/data_collector/` 采集或转换自己的行情数据，复现 examples 流程。

### 7.3 代码阅读与二次开发
- 深入阅读 `qlib/` 主目录下的核心模块（如 `data/`、`model/`、`workflow/`），为后续自定义开发打基础。

---

## 8. 建议的学习节奏

- 每周聚焦一个主题（如本周只做 benchmark，对比模型，下周专攻数据处理等）。
- 每个主题都要"读-跑-改-总结"：先读文档和代码，跑通示例，再尝试修改，最后做小结。
- 记录遇到的问题和解决方法，形成自己的 Qlib 学习笔记。

---

如你有特别感兴趣的方向（如高频、深度学习、组合优化、RL等），可以进一步细化某一块的学习路线！ 