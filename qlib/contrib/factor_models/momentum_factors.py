"""
动量因子模型
包括日内动量和动量反转因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

class IntradayMomentum:
    """日内动量因子"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ):
        """
        初始化日内动量因子
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
    
    def calculate_factors(self) -> pd.DataFrame:
        """计算日内动量因子"""
        # 获取股票数据
        try:
            # 尝试直接获取指数数据
            stock_data = D.features([self.instruments], ["$open", "$close", "$high", "$low"], 
                                  start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            stock_data = D.features(D.instruments(self.instruments), ["$open", "$close", "$high", "$low"], 
                                  start_time=self.start_time, end_time=self.end_time)
        
        # 计算因子
        factors = pd.DataFrame(index=stock_data.index)
        
        # 日内动量
        factors["intraday_momentum"] = (stock_data["$close"] - stock_data["$open"]) / stock_data["$open"]
        
        # 日内波动率
        factors["intraday_volatility"] = (stock_data["$high"] - stock_data["$low"]) / stock_data["$open"]
        
        return factors
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 计算因子
        factors = self.calculate_factors()
        
        # 计算因子收益
        factor_returns = pd.DataFrame(index=factors.index)
        
        for factor in ["intraday_momentum", "intraday_volatility"]:
            # 计算因子收益率
            factor_returns[factor] = factors[factor].pct_change()
        
        # 计算组合收益
        portfolio_returns = factor_returns.mean(axis=1)
        
        return pd.DataFrame({"returns": portfolio_returns})

class MomentumReversal:
    """动量反转因子"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ):
        """
        初始化动量反转因子
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
    
    def calculate_factors(self) -> pd.DataFrame:
        """计算动量反转因子"""
        # 获取股票数据
        try:
            # 尝试直接获取指数数据
            stock_data = D.features([self.instruments], ["$close"], 
                                  start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            stock_data = D.features(D.instruments(self.instruments), ["$close"], 
                                  start_time=self.start_time, end_time=self.end_time)
        
        # 计算因子
        factors = pd.DataFrame(index=stock_data.index)
        
        # 短期动量
        factors["short_term_momentum"] = stock_data["$close"].pct_change(5)
        
        # 中期动量
        factors["medium_term_momentum"] = stock_data["$close"].pct_change(20)
        
        # 长期动量
        factors["long_term_momentum"] = stock_data["$close"].pct_change(60)
        
        return factors
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 计算因子
        factors = self.calculate_factors()
        
        # 计算因子收益
        factor_returns = pd.DataFrame(index=factors.index)
        
        for factor in ["short_term_momentum", "medium_term_momentum", "long_term_momentum"]:
            # 计算因子收益率
            factor_returns[factor] = factors[factor].pct_change()
        
        # 计算组合收益
        portfolio_returns = factor_returns.mean(axis=1)
        
        return pd.DataFrame({"returns": portfolio_returns}) 