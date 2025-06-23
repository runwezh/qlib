"""
价量关系模型实现
包含量价背离和资金流向因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

class PriceVolumeDivergence:
    """量价背离因子实现"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        atr_period: int = 20
    ):
        """
        初始化量价背离因子
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        atr_period : int
            ATR计算周期
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.atr_period = atr_period
        
    def calculate_factors(self) -> pd.DataFrame:
        """计算价量背离因子"""
        # 获取股票数据
        try:
            # 尝试直接获取指数数据
            stock_data = D.features([self.instruments], ["$close", "$volume"], 
                                  start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            stock_data = D.features(D.instruments(self.instruments), ["$close", "$volume"], 
                                  start_time=self.start_time, end_time=self.end_time)
        
        # 计算因子
        factors = pd.DataFrame(index=stock_data.index)
        
        # 价格变化
        factors["price_change"] = stock_data["$close"].pct_change()
        
        # 成交量变化
        factors["volume_change"] = stock_data["$volume"].pct_change()
        
        # 价量背离
        factors["divergence"] = factors["price_change"] * factors["volume_change"]
        
        return factors
    
    def calculate_atr(self) -> pd.Series:
        """计算ATR"""
        # 获取价格数据
        price_data = D.features(self.instruments, ["$high", "$low", "$close"], 
                              start_time=self.start_time, end_time=self.end_time)
        
        # 计算TR
        tr1 = price_data["$high"] - price_data["$low"]
        tr2 = abs(price_data["$high"] - price_data["$close"].shift(1))
        tr3 = abs(price_data["$low"] - price_data["$close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_divergence(self) -> pd.DataFrame:
        """计算量价背离"""
        # 获取价格和成交量数据
        data = D.features(self.instruments, ["$close", "$volume"], 
                         start_time=self.start_time, end_time=self.end_time)
        
        # 计算价格和成交量的变化
        price_change = data["$close"].pct_change()
        volume_change = data["$volume"].pct_change()
        
        # 计算背离信号
        divergence = pd.DataFrame({
            "price_change": price_change,
            "volume_change": volume_change,
            "divergence": np.where(
                (price_change > 0) & (volume_change < 0),
                -1,  # 顶背离
                np.where(
                    (price_change < 0) & (volume_change > 0),
                    1,  # 底背离
                    0  # 无背离
                )
            )
        })
        
        return divergence
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 计算因子
        factors = self.calculate_factors()
        
        # 计算因子收益
        factor_returns = pd.DataFrame(index=factors.index)
        
        for factor in ["price_change", "volume_change", "divergence"]:
            # 计算因子收益率
            factor_returns[factor] = factors[factor].pct_change()
        
        # 计算组合收益
        portfolio_returns = factor_returns.mean(axis=1)
        
        return pd.DataFrame({"returns": portfolio_returns})

class CapitalFlow:
    """资金流向因子实现"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        large_order_threshold: float = 0.15,
        north_money_threshold: float = 0.005,
        margin_threshold: float = 0.2
    ):
        """
        初始化资金流向因子
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        large_order_threshold : float
            大单成交占比阈值
        north_money_threshold : float
            北向资金增持比例阈值
        margin_threshold : float
            融资买入占比阈值
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.large_order_threshold = large_order_threshold
        self.north_money_threshold = north_money_threshold
        self.margin_threshold = margin_threshold
        
    def calculate_factors(self) -> pd.DataFrame:
        """计算资金流向因子"""
        # 获取股票数据
        try:
            # 尝试直接获取指数数据
            stock_data = D.features([self.instruments], ["$close", "$volume", "$amount"], 
                                  start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            stock_data = D.features(D.instruments(self.instruments), ["$close", "$volume", "$amount"], 
                                  start_time=self.start_time, end_time=self.end_time)
        
        # 计算因子
        factors = pd.DataFrame(index=stock_data.index)
        
        # 资金流入
        factors["inflow"] = stock_data["$amount"] * (stock_data["$close"] > stock_data["$close"].shift(1))
        
        # 资金流出
        factors["outflow"] = stock_data["$amount"] * (stock_data["$close"] < stock_data["$close"].shift(1))
        
        # 净流入
        factors["net_inflow"] = factors["inflow"] - factors["outflow"]
        
        return factors
    
    def calculate_large_order_flow(self) -> pd.Series:
        """计算大单资金流向"""
        # 获取成交量和成交额数据
        data = D.features(self.instruments, ["$volume", "$amount"], 
                         start_time=self.start_time, end_time=self.end_time)
        
        # 计算大单成交占比
        large_order_ratio = data["$amount"] / data["$volume"]
        
        # 计算大单资金流向
        large_order_flow = np.where(
            large_order_ratio > self.large_order_threshold,
            data["$amount"],
            0
        )
        
        return pd.Series(large_order_flow, index=data.index)
    
    def calculate_north_money_flow(self) -> pd.Series:
        """计算北向资金流向"""
        # 获取北向资金数据
        north_data = D.features(self.instruments, ["$north_money", "$float_share"], 
                              start_time=self.start_time, end_time=self.end_time)
        
        # 计算北向资金增持比例
        north_ratio = north_data["$north_money"] / north_data["$float_share"]
        
        # 计算连续增持
        north_flow = north_ratio.rolling(window=3).sum()
        
        return north_flow
    
    def calculate_margin_flow(self) -> pd.Series:
        """计算融资资金流向"""
        # 获取融资数据
        margin_data = D.features(self.instruments, ["$margin_buy", "$amount"], 
                               start_time=self.start_time, end_time=self.end_time)
        
        # 计算融资买入占比
        margin_ratio = margin_data["$margin_buy"] / margin_data["$amount"]
        
        # 计算融资资金流向
        margin_flow = np.where(
            margin_ratio > self.margin_threshold,
            margin_data["$margin_buy"],
            0
        )
        
        return pd.Series(margin_flow, index=margin_data.index)
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 计算因子
        factors = self.calculate_factors()
        
        # 计算因子收益
        factor_returns = pd.DataFrame(index=factors.index)
        
        for factor in ["inflow", "outflow", "net_inflow"]:
            # 计算因子收益率
            factor_returns[factor] = factors[factor].pct_change()
        
        # 计算组合收益
        portfolio_returns = factor_returns.mean(axis=1)
        
        return pd.DataFrame({"returns": portfolio_returns}) 