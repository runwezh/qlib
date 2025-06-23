# 多因子选股系统核心算法说明

## 1. 因子模型体系

### 1.1 经典因子模型

#### Fama-French三因子模型
```python
# 核心计算逻辑
def calculate_factors(self):
    # 市值因子 (SMB)
    smb = self._calculate_smb()
    
    # 账面市值比因子 (HML)
    hml = self._calculate_hml()
    
    # 市场因子
    market = self._calculate_market_factor()
    
    return pd.DataFrame({
        "smb": smb,
        "hml": hml,
        "market": market
    })
```

#### SDF模型
```python
# 核心计算逻辑
def calculate_factors(self):
    # 宏观因子
    macro = self._calculate_macro_factors()
    
    # 微观因子
    micro = self._calculate_micro_factors()
    
    # 北向资金因子
    north_money = self._calculate_north_money()
    
    return pd.DataFrame({
        "macro": macro,
        "micro": micro,
        "north_money": north_money
    })
```

### 1.2 动量因子模型

#### 日内动量因子
```python
# 核心计算逻辑
def calculate_factors(self):
    # 日内动量
    intraday = self._calculate_intraday_momentum()
    
    # 隔夜动量
    overnight = self._calculate_overnight_momentum()
    
    return pd.DataFrame({
        "intraday": intraday,
        "overnight": overnight
    })
```

#### 动量反转因子
```python
# 核心计算逻辑
def calculate_factors(self):
    # 短期动量
    short_term = self._calculate_short_term_momentum()
    
    # 长期反转
    long_term = self._calculate_long_term_reversal()
    
    # VIX调整
    vix_adjusted = self._adjust_by_vix(short_term, long_term)
    
    return pd.DataFrame({
        "short_term": short_term,
        "long_term": long_term,
        "vix_adjusted": vix_adjusted
    })
```

### 1.3 价量关系模型

#### 量价背离因子
```python
# 核心计算逻辑
def calculate_factors(self):
    # ATR计算
    atr = self._calculate_atr()
    
    # 背离信号
    divergence = self._calculate_divergence()
    
    return pd.DataFrame({
        "atr": atr,
        "divergence": divergence
    })
```

#### 资金流向因子
```python
# 核心计算逻辑
def calculate_factors(self):
    # 大单资金流向
    large_order = self._calculate_large_order_flow()
    
    # 北向资金流向
    north_money = self._calculate_north_money_flow()
    
    # 融资资金流向
    margin = self._calculate_margin_flow()
    
    return pd.DataFrame({
        "large_order": large_order,
        "north_money": north_money,
        "margin": margin
    })
```

## 2. 因子组合优化

### 2.1 风险平价方法
```python
def optimize_weights(self):
    # 计算协方差矩阵
    cov_matrix = returns.cov()
    
    # 计算风险平价权重
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / inv_vol.sum()
    
    return weights
```

### 2.2 场景自适应
```python
def adapt_to_scenario(self):
    # 识别市场场景
    scenario = self._identify_market_scenario()
    
    # 根据场景调整权重
    if scenario == MarketScenario.BULL:
        weights = self._bull_market_weights()
    elif scenario == MarketScenario.BEAR:
        weights = self._bear_market_weights()
    elif scenario == MarketScenario.SIDEWAYS:
        weights = self._sideways_market_weights()
    else:  # VOLATILE
        weights = self._volatile_market_weights()
    
    return weights
```

## 3. 选股策略

### 3.1 综合得分计算
```python
def calculate_composite_score(self):
    # 获取因子暴露
    factor_exposures = self._get_factor_exposures()
    
    # 获取场景暴露
    scenario_exposures = self._get_scenario_exposures()
    
    # 计算综合得分
    scores = pd.Series(0, index=factor_exposures.index)
    
    # 合并因子得分
    for factor, weight in self.factor_weights.items():
        scores += weight * factor_exposures[factor]
    
    # 合并场景得分
    for scenario, weight in self.scenario_weights.items():
        scores += weight * scenario_exposures[scenario]
    
    return scores
```

### 3.2 股票筛选
```python
def select_stocks(self):
    # 计算综合得分
    scores = self.calculate_composite_score()
    
    # 选择得分最高的股票
    selected_stocks = scores.nlargest(self.max_stocks)
    
    # 生成选股结果
    result = {
        "short_term": selected_stocks.index.tolist()[:10],
        "mid_term": selected_stocks.index.tolist(),
        "scores": selected_stocks.to_dict()
    }
    
    return result
```

## 4. 风险控制

### 4.1 回撤控制
```python
def control_drawdown(self):
    # 计算回撤
    drawdown = self._calculate_drawdown()
    
    # 检查是否超过阈值
    if drawdown <= -self.max_drawdown_threshold:
        # 调整仓位
        self._adjust_position()
        
        # 更新风险控制参数
        self._update_risk_parameters()
```

### 4.2 波动率控制
```python
def control_volatility(self):
    # 计算波动率
    volatility = self._calculate_volatility()
    
    # 检查是否超过阈值
    if volatility > self.volatility_threshold:
        # 调整因子权重
        self._adjust_factor_weights()
        
        # 更新风险控制参数
        self._update_risk_parameters()
```

## 5. 性能优化

### 5.1 数据预处理
```python
def preprocess_data(self):
    # 数据清洗
    data = self._clean_data()
    
    # 异常值处理
    data = self._handle_outliers(data)
    
    # 缺失值处理
    data = self._handle_missing_values(data)
    
    return data
```

### 5.2 计算优化
```python
def optimize_calculation(self):
    # 使用向量化运算
    results = np.vectorize(self._calculate_factor)(data)
    
    # 使用并行计算
    with Pool() as pool:
        results = pool.map(self._calculate_factor, data_chunks)
    
    return results
``` 