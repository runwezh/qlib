"""
经典多因子模型实现
包含 Fama-French 三因子模型和随机贴现因子(SDF)模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

class FamaFrenchModel:
    """Fama-French 三因子模型实现"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        momentum_filter: bool = True,
        quality_filter: bool = True,
        beta_target: float = 0.3
    ):
        """
        初始化 Fama-French 三因子模型
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        momentum_filter : bool
            是否使用动量过滤
        quality_filter : bool
            是否使用质量过滤
        beta_target : float
            目标市场beta值
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.momentum_filter = momentum_filter
        self.quality_filter = quality_filter
        self.beta_target = beta_target
        
    def calculate_hml(self) -> pd.Series:
        """计算价值因子(HML)"""
        try:
            # 获取账面市值比数据
            bm_data = D.features(self.instruments, ["$bm"], start_time=self.start_time, end_time=self.end_time)
            
            if bm_data.empty:
                print("Warning: No book-to-market data available")
                return pd.Series()
            
            # 按账面市值比分组
            bm_quantiles = bm_data.groupby(level=0)["$bm"].transform(lambda x: pd.qcut(x, 3, labels=["L", "M", "H"]))
            
            # 计算HML因子收益
            hml_returns = bm_data.groupby([bm_quantiles, bm_data.index.get_level_values(0)])["$bm"].mean()
            
            # 验证数据
            if "H" not in hml_returns or "L" not in hml_returns:
                print("Warning: Missing H or L group in HML calculation")
                return pd.Series()
            
            return hml_returns["H"] - hml_returns["L"]
            
        except Exception as e:
            print(f"Error in calculate_hml: {str(e)}")
            return pd.Series()
    
    def calculate_smb(self) -> pd.Series:
        """计算规模因子(SMB)"""
        try:
            # 获取市值数据
            mkt_cap = D.features(self.instruments, ["$mkt_cap"], start_time=self.start_time, end_time=self.end_time)
            
            if mkt_cap.empty:
                print("Warning: No market cap data available")
                return pd.Series()
            
            # 按市值分组
            size_quantiles = mkt_cap.groupby(level=0)["$mkt_cap"].transform(lambda x: pd.qcut(x, 3, labels=["S", "M", "B"]))
            
            # 计算SMB因子收益
            smb_returns = mkt_cap.groupby([size_quantiles, mkt_cap.index.get_level_values(0)])["$mkt_cap"].mean()
            
            # 验证数据
            if "S" not in smb_returns or "B" not in smb_returns:
                print("Warning: Missing S or B group in SMB calculation")
                return pd.Series()
            
            return smb_returns["S"] - smb_returns["B"]
            
        except Exception as e:
            print(f"Error in calculate_smb: {str(e)}")
            return pd.Series()
    
    def calculate_mkt(self) -> pd.Series:
        """计算市场因子(MKT)"""
        try:
            # 获取市场收益率
            market_returns = D.features("csi300", ["$close"], start_time=self.start_time, end_time=self.end_time)
            
            if market_returns.empty:
                print("Warning: No market return data available")
                return pd.Series()
            
            market_returns = market_returns["$close"].pct_change()
            return market_returns
            
        except Exception as e:
            print(f"Error in calculate_mkt: {str(e)}")
            return pd.Series()
    
    def apply_momentum_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用动量过滤"""
        if not self.momentum_filter:
            return data
        # 计算6个月动量
        momentum = data["$close"].pct_change(periods=126)
        # 过滤掉后30%的股票
        momentum_quantiles = momentum.groupby(level=0).transform(lambda x: pd.qcut(x, 10, labels=False))
        return data[momentum_quantiles < 7]
    
    def apply_quality_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用质量过滤"""
        if not self.quality_filter:
            return data
        # 获取ROE和资产负债率数据
        quality_data = D.features(self.instruments, ["$roe", "$debt_to_assets"], 
                                start_time=self.start_time, end_time=self.end_time)
        # 应用质量过滤条件
        quality_mask = (quality_data["$roe"] > 0.08) & (quality_data["$debt_to_assets"] < 0.6)
        return data[quality_mask]
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        try:
            # 计算因子收益
            hml_returns = self.calculate_hml()
            smb_returns = self.calculate_smb()
            mkt_returns = self.calculate_mkt()
            
            # 检查是否有有效的因子收益
            if hml_returns.empty or smb_returns.empty or mkt_returns.empty:
                print("Warning: One or more factor returns are empty")
                return pd.DataFrame()
            
            # 获取基础数据
            base_data = D.features(self.instruments, ["$close", "$volume"], 
                                 start_time=self.start_time, end_time=self.end_time)
            
            if base_data.empty:
                print("Warning: No base data available")
                return pd.DataFrame()
            
            # 应用过滤条件
            filtered_data = self.apply_momentum_filter(base_data)
            filtered_data = self.apply_quality_filter(filtered_data)
            
            if filtered_data.empty:
                print("Warning: No data after filtering")
                return pd.DataFrame()
            
            # 计算因子暴露
            factor_exposures = pd.DataFrame({
                "HML": hml_returns,
                "SMB": smb_returns,
                "MKT": mkt_returns
            })
            
            # 优化因子权重
            weights = self._optimize_weights(factor_exposures)
            
            # 生成投资组合
            portfolio = self._generate_portfolio(filtered_data, weights)
            
            return portfolio
            
        except Exception as e:
            print(f"Error in optimize_portfolio: {str(e)}")
            return pd.DataFrame()
    
    def _optimize_weights(self, factor_exposures: pd.DataFrame) -> Dict[str, float]:
        """优化因子权重"""
        # 计算因子协方差矩阵
        cov_matrix = factor_exposures.cov()
        
        # 使用风险平价方法优化权重
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        
        return dict(zip(factor_exposures.columns, weights))
    
    def _generate_portfolio(self, data: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """生成投资组合"""
        # 计算组合收益
        portfolio_returns = pd.Series(0, index=data.index.get_level_values(0).unique())
        for factor, weight in weights.items():
            if factor == "MKT":
                portfolio_returns += weight * self.calculate_mkt()
            elif factor == "HML":
                portfolio_returns += weight * self.calculate_hml()
            elif factor == "SMB":
                portfolio_returns += weight * self.calculate_smb()
        
        return pd.DataFrame({"returns": portfolio_returns})

class SDFModel:
    """随机贴现因子(SDF)模型实现"""
    
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        macro_factors: List[str] = None,
        micro_factors: List[str] = None
    ):
        """
        初始化 SDF 模型
        
        Parameters
        ----------
        instruments : str
            股票池
        start_time : str
            开始时间
        end_time : str
            结束时间
        macro_factors : List[str]
            宏观因子列表
        micro_factors : List[str]
            微观因子列表
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.macro_factors = macro_factors or ["$pmi", "$cpi"]
        self.micro_factors = micro_factors or ["$cash_flow", "$peg"]
        
    def calculate_macro_factors(self) -> pd.DataFrame:
        """计算宏观因子"""
        macro_data = D.features(self.instruments, self.macro_factors, 
                              start_time=self.start_time, end_time=self.end_time)
        return macro_data
    
    def calculate_micro_factors(self) -> pd.DataFrame:
        """计算微观因子"""
        micro_data = D.features(self.instruments, self.micro_factors,
                              start_time=self.start_time, end_time=self.end_time)
        return micro_data
    
    def calculate_northbound_flow(self) -> pd.Series:
        """计算北向资金净流入"""
        # 获取北向资金数据
        northbound_data = D.features(self.instruments, ["$north_money"], 
                                   start_time=self.start_time, end_time=self.end_time)
        return northbound_data["$north_money"]
    
    def calculate_bond_spread(self) -> pd.Series:
        """计算国债收益率利差"""
        # 获取国债收益率数据
        bond_data = D.features(self.instruments, ["$bond_yield"], 
                             start_time=self.start_time, end_time=self.end_time)
        return bond_data["$bond_yield"]
    
    def optimize_portfolio(self) -> pd.DataFrame:
        """优化投资组合"""
        # 计算各类因子
        macro_factors = self.calculate_macro_factors()
        micro_factors = self.calculate_micro_factors()
        northbound_flow = self.calculate_northbound_flow()
        bond_spread = self.calculate_bond_spread()
        
        # 合并因子数据
        factor_data = pd.concat([
            macro_factors,
            micro_factors,
            northbound_flow.to_frame("northbound_flow"),
            bond_spread.to_frame("bond_spread")
        ], axis=1)
        
        # 优化因子权重
        weights = self._optimize_weights(factor_data)
        
        # 生成投资组合
        portfolio = self._generate_portfolio(factor_data, weights)
        
        return portfolio
    
    def _optimize_weights(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """优化因子权重"""
        # 计算因子协方差矩阵
        cov_matrix = factor_data.cov()
        
        # 使用风险平价方法优化权重
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        
        return dict(zip(factor_data.columns, weights))
    
    def _generate_portfolio(self, factor_data: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """生成投资组合"""
        # 计算组合收益
        portfolio_returns = pd.Series(0, index=factor_data.index.get_level_values(0).unique())
        for factor, weight in weights.items():
            portfolio_returns += weight * factor_data[factor]
        
        return pd.DataFrame({"returns": portfolio_returns}) 