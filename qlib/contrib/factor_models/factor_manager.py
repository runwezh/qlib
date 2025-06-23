"""
因子组合管理器
用于协调和组合多个因子模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

from .classic_factors import FamaFrenchModel, SDFModel
from .momentum_factors import IntradayMomentum, MomentumReversal
from .price_volume_factors import PriceVolumeDivergence, CapitalFlow
from .composite_scenario import CompositeScenarioModel, MarketScenario

class FactorManager:
    """因子组合管理器"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        scenario_weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化因子组合管理器
        
        Parameters
        ----------
        instruments : str
            股票池，支持以下格式：
            - "csi500": 中证500成分股
            - "csi300": 沪深300成分股
            - "csi100": 中证100成分股
            - "all": 所有股票
            - 指数代码：如 "SH000905"
        start_time : str
            开始时间
        end_time : str
            结束时间
        factor_weights : Dict[str, float]
            因子权重配置
        scenario_weights : Dict[str, float]
            场景权重配置
        """
        # 处理股票池参数
        if instruments.startswith("SH") or instruments.startswith("SZ"):
            # 如果是指数代码，获取其成分股
            self.instruments = D.instruments(instruments)
        else:
            # 如果是预定义的股票池名称，获取其成分股
            self.instruments = D.instruments(instruments)
            
        self.start_time = start_time
        self.end_time = end_time
        
        # 初始化因子模型
        self.factor_models = {
            "fama_french": FamaFrenchModel(self.instruments, start_time, end_time),
            "sdf": SDFModel(self.instruments, start_time, end_time),
            "intraday": IntradayMomentum(self.instruments, start_time, end_time),
            "momentum_reversal": MomentumReversal(self.instruments, start_time, end_time),
            "price_volume": PriceVolumeDivergence(self.instruments, start_time, end_time),
            "capital_flow": CapitalFlow(self.instruments, start_time, end_time),
            "scenario": CompositeScenarioModel(self.instruments, start_time, end_time)
        }
        
        # 设置默认因子权重
        self.factor_weights = factor_weights or {
            "fama_french": 0.2,
            "sdf": 0.2,
            "intraday": 0.15,
            "momentum_reversal": 0.15,
            "price_volume": 0.1,
            "capital_flow": 0.1,
            "scenario": 0.1
        }
        
        # 设置默认场景权重
        self.scenario_weights = scenario_weights or {
            MarketScenario.BULL: 0.4,
            MarketScenario.BEAR: 0.3,
            MarketScenario.SIDEWAYS: 0.2,
            MarketScenario.VOLATILE: 0.1
        }
    
    def calculate_factor_returns(self) -> Dict[str, pd.DataFrame]:
        """计算各因子收益"""
        factor_returns = {}
        
        for name, model in self.factor_models.items():
            try:
                factor_returns[name] = model.optimize_portfolio()
            except Exception as e:
                print(f"Error calculating returns for {name}: {str(e)}")
                continue
        
        return factor_returns
    
    def calculate_scenario_returns(self) -> pd.DataFrame:
        """计算场景收益"""
        scenario_model = self.factor_models["scenario"]
        scenario = scenario_model.identify_market_scenario()
        
        # 获取基础收益数据
        try:
            # 尝试直接获取指数数据
            base_returns = D.features([self.instruments], ["$close"], 
                                    start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            base_returns = D.features(D.instruments(self.instruments), ["$close"], 
                                    start_time=self.start_time, end_time=self.end_time)
            # 计算指数收益率
            base_returns = base_returns.groupby(level=0)["$close"].mean()
            base_returns = pd.DataFrame({"$close": base_returns})
        
        base_returns = base_returns["$close"].pct_change()
        
        # 计算各场景收益
        scenario_returns = pd.DataFrame(index=base_returns.index)
        
        for scenario_type in MarketScenario.__dict__.values():
            if isinstance(scenario_type, str):
                # 获取当前场景的数据
                scenario_mask = scenario == scenario_type
                if scenario_mask.any():
                    # 计算场景收益
                    scenario_returns[scenario_type] = base_returns[scenario_mask].mean()
        
        return scenario_returns
    
    def optimize_weights(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """优化因子和场景权重"""
        # 计算因子收益
        factor_returns = self.calculate_factor_returns()
        
        # 计算场景收益
        scenario_returns = self.calculate_scenario_returns()
        
        # 计算因子协方差矩阵
        factor_cov = pd.DataFrame({name: returns["returns"] 
                                 for name, returns in factor_returns.items()}).cov()
        
        # 使用风险平价方法优化因子权重
        inv_vol = 1 / np.sqrt(np.diag(factor_cov))
        factor_weights = dict(zip(factor_returns.keys(), inv_vol / inv_vol.sum()))
        
        # 计算场景协方差矩阵
        scenario_cov = scenario_returns.cov()
        
        # 使用风险平价方法优化场景权重
        inv_vol = 1 / np.sqrt(np.diag(scenario_cov))
        scenario_weights = dict(zip(scenario_returns.columns, inv_vol / inv_vol.sum()))
        
        return factor_weights, scenario_weights
    
    def generate_portfolio(self) -> pd.DataFrame:
        """生成组合投资组合"""
        try:
            # 计算各因子收益
            factor_returns = self.calculate_factor_returns()
            
            # 如果没有成功的因子，返回空DataFrame
            if not factor_returns:
                print("Warning: No factor returns were successfully calculated")
                return pd.DataFrame()
            
            # 计算场景收益
            scenario_returns = self.calculate_scenario_returns()
            
            # 确保所有因子收益具有相同的索引
            common_index = None
            for name, returns in factor_returns.items():
                if isinstance(returns, pd.DataFrame) and "returns" in returns.columns:
                    if common_index is None:
                        common_index = returns.index
                    else:
                        common_index = common_index.intersection(returns.index)
                elif isinstance(returns, pd.Series):
                    if common_index is None:
                        common_index = returns.index
                    else:
                        common_index = common_index.intersection(returns.index)
            
            if common_index is None or len(common_index) == 0:
                print("Warning: No common index found among factor returns")
                return pd.DataFrame()
            
            # 生成组合收益
            portfolio_returns = pd.Series(0, index=common_index)
            
            # 计算加权收益
            for name, returns in factor_returns.items():
                if isinstance(returns, pd.DataFrame) and "returns" in returns.columns:
                    # 对齐索引
                    aligned_returns = returns.loc[common_index, "returns"]
                    portfolio_returns += aligned_returns * self.factor_weights[name]
                elif isinstance(returns, pd.Series):
                    # 对齐索引
                    aligned_returns = returns.loc[common_index]
                    portfolio_returns += aligned_returns * self.factor_weights[name]
            
            # 添加场景调整
            if not scenario_returns.empty:
                # 对齐场景收益的索引
                scenario_returns = scenario_returns.loc[common_index]
                for scenario, weight in self.scenario_weights.items():
                    if scenario in scenario_returns.columns:
                        portfolio_returns += scenario_returns[scenario] * weight
            
            return pd.DataFrame({
                "portfolio_returns": portfolio_returns,
                "scenario": scenario_returns.idxmax(axis=1) if not scenario_returns.empty else None
            })
            
        except Exception as e:
            print(f"Error in generate_portfolio: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_exposures(self) -> pd.DataFrame:
        """获取因子暴露"""
        exposures = {}
        
        for name, model in self.factor_models.items():
            try:
                if hasattr(model, "calculate_factor_exposures"):
                    exposures[name] = model.calculate_factor_exposures()
                else:
                    # 使用默认方法计算因子暴露
                    returns = model.optimize_portfolio()
                    exposures[name] = returns["returns"]
            except Exception as e:
                print(f"Error calculating exposures for {name}: {str(e)}")
                continue
        
        return pd.DataFrame(exposures)
    
    def get_scenario_exposures(self) -> pd.DataFrame:
        """获取场景暴露"""
        scenario_model = self.factor_models["scenario"]
        scenario = scenario_model.identify_market_scenario()
        
        # 计算各场景的暴露
        exposures = pd.DataFrame(index=scenario.index)
        
        for scenario_type in MarketScenario.__dict__.values():
            if isinstance(scenario_type, str):
                exposures[scenario_type] = (scenario == scenario_type).astype(float)
        
        return exposures 