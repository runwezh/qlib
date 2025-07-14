"""
回测评估器
用于评估因子组合的表现
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

from .factor_manager import FactorManager

class BacktestEvaluator:
    """回测评估器"""
    
    def __init__(
        self,
        factor_manager: FactorManager,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ):
        """
        初始化回测评估器
        
        Parameters
        ----------
        factor_manager : FactorManager
            因子管理器
        start_time : str
            开始时间
        end_time : str
            结束时间
        """
        self.factor_manager = factor_manager
        self.start_time = start_time
        self.end_time = end_time
    
    def run_backtest(self) -> pd.DataFrame:
        """运行回测"""
        # 生成投资组合
        portfolio = self.factor_manager.generate_portfolio()
        
        if portfolio.empty:
            return pd.DataFrame()
        
        # 计算累积收益
        portfolio["cumulative_returns"] = (1 + portfolio["portfolio_returns"]).cumprod()
        
        # 计算基准收益
        try:
            # 尝试直接获取指数数据
            benchmark_data = D.features([self.factor_manager.instruments], ["$close"], 
                                      start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            benchmark_data = D.features(D.instruments(self.factor_manager.instruments), ["$close"], 
                                      start_time=self.start_time, end_time=self.end_time)
            # 计算指数收益率
            benchmark_data = benchmark_data.groupby(level=0)["$close"].mean()
            benchmark_data = pd.DataFrame({"$close": benchmark_data})
        
        benchmark_returns = benchmark_data["$close"].pct_change()
        portfolio["benchmark_returns"] = benchmark_returns
        portfolio["benchmark_cumulative_returns"] = (1 + benchmark_returns).cumprod()
        
        return portfolio
    
    def calculate_metrics(self, portfolio: pd.DataFrame) -> Dict[str, float]:
        """计算评估指标"""
        if portfolio.empty:
            return {}
        
        metrics = {}
        
        # 计算年化收益率
        total_days = (portfolio.index[-1] - portfolio.index[0]).days
        total_returns = portfolio["cumulative_returns"].iloc[-1] - 1
        metrics["annual_return"] = (1 + total_returns) ** (365 / total_days) - 1
        
        # 计算年化波动率
        daily_returns = portfolio["portfolio_returns"]
        metrics["annual_volatility"] = daily_returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        metrics["sharpe_ratio"] = (metrics["annual_return"] - risk_free_rate) / metrics["annual_volatility"]
        
        # 计算最大回撤
        cumulative_returns = portfolio["cumulative_returns"]
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        metrics["max_drawdown"] = drawdowns.min()
        
        # 计算信息比率
        excess_returns = portfolio["portfolio_returns"] - portfolio["benchmark_returns"]
        metrics["information_ratio"] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return metrics
    
    def generate_report(self) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """生成评估报告"""
        # 运行回测
        portfolio = self.run_backtest()
        
        if portfolio.empty:
            return {
                "portfolio": pd.DataFrame(),
                "metrics": {}
            }
        
        # 计算评估指标
        metrics = self.calculate_metrics(portfolio)
        
        return {
            "portfolio": portfolio,
            "metrics": metrics
        }
    
    def optimize_factor_weights(self) -> Dict:
        """优化因子权重配置"""
        # 获取因子暴露
        factor_exposures = self.factor_manager.get_factor_exposures()
        scenario_exposures = self.factor_manager.get_scenario_exposures()
        
        # 计算因子收益
        factor_returns = self.factor_manager.calculate_factor_returns()
        scenario_returns = self.factor_manager.calculate_scenario_returns()
        
        # 计算因子协方差矩阵
        factor_cov = pd.DataFrame({name: returns["returns"] 
                                 for name, returns in factor_returns.items()}).cov()
        
        # 计算场景协方差矩阵
        scenario_cov = scenario_returns.cov()
        
        # 使用风险平价方法优化因子权重
        inv_vol = 1 / np.sqrt(np.diag(factor_cov))
        factor_weights = dict(zip(factor_returns.keys(), inv_vol / inv_vol.sum()))
        
        # 使用风险平价方法优化场景权重
        inv_vol = 1 / np.sqrt(np.diag(scenario_cov))
        scenario_weights = dict(zip(scenario_returns.columns, inv_vol / inv_vol.sum()))
        
        # 生成优化后的配置
        config = {
            "factor_weights": factor_weights,
            "scenario_weights": scenario_weights,
            "optimization_date": datetime.now().strftime("%Y-%m-%d"),
            "target_returns": {
                "short_term": self.target_short_return,
                "mid_term": self.target_mid_return
            },
            "risk_metrics": {
                "max_drawdown_threshold": self.max_drawdown_threshold,
                "volatility_threshold": 0.25  # 年化波动率阈值
            }
        }
        
        return config
    
    def _analyze_drawdown_periods(self, portfolio: pd.DataFrame) -> List[Dict]:
        """分析回撤期间"""
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, row in portfolio.iterrows():
            if row["drawdown"] <= -self.max_drawdown_threshold and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif row["drawdown"] > -self.max_drawdown_threshold and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    "start_date": start_date,
                    "end_date": date,
                    "max_drawdown": portfolio.loc[start_date:date, "drawdown"].min()
                })
        
        return drawdown_periods
    
    def save_report(self, filepath: str):
        """保存分析报告"""
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4) 