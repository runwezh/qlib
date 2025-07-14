"""
复合场景模型实现
用于动态适应不同市场环境
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

class MarketScenario:
    """市场场景定义"""
    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    VOLATILE = "volatile"  # 剧烈波动

class CompositeScenarioModel:
    """复合场景模型实现"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        scenario_window: int = 20,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.05
    ):
        """
        初始化复合场景模型
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        scenario_window : int
            场景判断窗口
        volatility_threshold : float
            波动率阈值
        trend_threshold : float
            趋势阈值
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.scenario_window = scenario_window
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
    def identify_market_scenario(self) -> pd.Series:
        """识别市场场景"""
        # 获取指数数据
        try:
            # 尝试直接获取指数数据
            index_data = D.features([self.instruments], ["$close"], 
                                  start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            index_data = D.features(D.instruments(self.instruments), ["$close"], 
                                  start_time=self.start_time, end_time=self.end_time)
            # 计算指数收益率
            index_data = index_data.groupby(level=0)["$close"].mean()
            index_data = pd.DataFrame({"$close": index_data})
        
        # 计算收益率和波动率
        returns = index_data["$close"].pct_change()
        volatility = returns.rolling(window=self.scenario_window).std()
        
        # 计算趋势
        trend = returns.rolling(window=self.scenario_window).mean()
        
        # 识别市场场景
        scenario = pd.Series(index=returns.index)
        
        # 剧烈波动市
        scenario[volatility > self.volatility_threshold] = MarketScenario.VOLATILE
        
        # 牛市
        scenario[(volatility <= self.volatility_threshold) & 
                (trend > self.trend_threshold)] = MarketScenario.BULL
        
        # 熊市
        scenario[(volatility <= self.volatility_threshold) & 
                (trend < -self.trend_threshold)] = MarketScenario.BEAR
        
        # 震荡市
        scenario[(volatility <= self.volatility_threshold) & 
                (abs(trend) <= self.trend_threshold)] = MarketScenario.SIDEWAYS
        
        return scenario
    
    def calculate_scenario_factors(self) -> pd.DataFrame:
        """计算场景因子"""
        # 获取基础数据
        data = D.features(self.instruments, 
                         ["$close", "$volume", "$amount", "$turnover_rate"], 
                         start_time=self.start_time, end_time=self.end_time)
        
        # 计算技术指标
        data["ma5"] = data["$close"].rolling(window=5).mean()
        data["ma20"] = data["$close"].rolling(window=20).mean()
        data["vol_ma5"] = data["$volume"].rolling(window=5).mean()
        data["vol_ma20"] = data["$volume"].rolling(window=20).mean()
        
        # 计算因子
        factors = pd.DataFrame(index=data.index)
        
        # 趋势因子
        factors["trend"] = (data["ma5"] - data["ma20"]) / data["ma20"]
        
        # 成交量因子
        factors["volume"] = (data["vol_ma5"] - data["vol_ma20"]) / data["vol_ma20"]
        
        # 换手率因子
        factors["turnover"] = data["$turnover_rate"].rolling(window=5).mean()
        
        return factors
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 识别市场场景
        scenario = self.identify_market_scenario()
        
        # 计算场景因子
        factors = self.calculate_scenario_factors()
        
        # 合并数据
        factor_data = pd.concat([factors, scenario.to_frame("scenario")], axis=1)
        
        # 生成投资组合
        portfolio = self._generate_portfolio(factor_data)
        
        return portfolio
    
    def _generate_portfolio(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """生成投资组合"""
        # 初始化投资组合
        portfolio_returns = pd.Series(0, index=factor_data.index.get_level_values(0).unique())
        
        # 根据不同场景生成投资组合
        for scenario in MarketScenario.__dict__.values():
            if isinstance(scenario, str):
                # 获取当前场景的数据
                scenario_data = factor_data[factor_data["scenario"] == scenario]
                
                if len(scenario_data) > 0:
                    # 根据场景特点选择因子
                    if scenario == MarketScenario.BULL:
                        # 牛市：关注趋势和成交量
                        score = scenario_data["trend"] + scenario_data["volume"]
                    elif scenario == MarketScenario.BEAR:
                        # 熊市：关注换手率和趋势
                        score = -scenario_data["trend"] + scenario_data["turnover"]
                    elif scenario == MarketScenario.SIDEWAYS:
                        # 震荡市：关注换手率
                        score = scenario_data["turnover"]
                    else:  # VOLATILE
                        # 剧烈波动：综合多个因子
                        score = (scenario_data["trend"].abs() + 
                                scenario_data["volume"].abs() + 
                                scenario_data["turnover"])
                    
                    # 选择得分最高的股票
                    selected_stocks = score.nlargest(20)
                    
                    # 等权重配置
                    if len(selected_stocks) > 0:
                        portfolio_returns[selected_stocks.index] = 1 / len(selected_stocks)
        
        return pd.DataFrame({"returns": portfolio_returns}) 