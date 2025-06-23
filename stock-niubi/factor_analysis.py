"""
因子分析主程序
用于分析因子表现并生成优化配置
"""

import os
import json
from datetime import datetime, timedelta
import qlib
from qlib.constant import REG_CN
from qlib.contrib.factor_models import FactorManager
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union

# 初始化 qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)

class BacktestEvaluator:
    """回测评估器"""
    
    def __init__(
        self,
        factor_manager: FactorManager,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        short_term_days: int = 10,
        mid_term_days: int = 60,
        target_short_return: float = 0.10,
        target_mid_return: float = 0.30,
        max_drawdown_threshold: float = 0.15
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
        short_term_days : int
            短期目标天数
        mid_term_days : int
            中期目标天数
        target_short_return : float
            短期目标收益率
        target_mid_return : float
            中期目标收益率
        max_drawdown_threshold : float
            最大回撤阈值
        """
        self.factor_manager = factor_manager
        self.start_time = start_time
        self.end_time = end_time
        self.short_term_days = short_term_days
        self.mid_term_days = mid_term_days
        self.target_short_return = target_short_return
        self.target_mid_return = target_mid_return
        self.max_drawdown_threshold = max_drawdown_threshold
    
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
            benchmark_data = qlib.data.D.features([self.factor_manager.instruments], ["$close"], 
                                      start_time=self.start_time, end_time=self.end_time)
        except ValueError:
            # 如果失败，尝试获取成分股数据
            benchmark_data = qlib.data.D.features(qlib.data.D.instruments(self.factor_manager.instruments), ["$close"], 
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
        
        # 计算短期和中期收益
        if len(portfolio) >= self.short_term_days:
            short_term_returns = portfolio["cumulative_returns"].iloc[-1] / portfolio["cumulative_returns"].iloc[-self.short_term_days] - 1
            metrics["short_term_return"] = short_term_returns
        
        if len(portfolio) >= self.mid_term_days:
            mid_term_returns = portfolio["cumulative_returns"].iloc[-1] / portfolio["cumulative_returns"].iloc[-self.mid_term_days] - 1
            metrics["mid_term_return"] = mid_term_returns
        
        return metrics
    
    def evaluate_performance(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """评估性能是否满足目标"""
        evaluation = {}
        
        # 评估短期收益目标
        if "short_term_return" in metrics:
            evaluation["short_term_target_met"] = metrics["short_term_return"] >= self.target_short_return
        
        # 评估中期收益目标
        if "mid_term_return" in metrics:
            evaluation["mid_term_target_met"] = metrics["mid_term_return"] >= self.target_mid_return
        
        # 评估最大回撤
        if "max_drawdown" in metrics:
            evaluation["drawdown_target_met"] = abs(metrics["max_drawdown"]) <= self.max_drawdown_threshold
        
        return evaluation
    
    def generate_report(self) -> Dict[str, Union[pd.DataFrame, Dict[str, float], Dict[str, bool]]]:
        """生成评估报告"""
        # 运行回测
        portfolio = self.run_backtest()
        
        if portfolio.empty:
            return {
                "portfolio": pd.DataFrame(),
                "metrics": {},
                "evaluation": {}
            }
        
        # 计算评估指标
        metrics = self.calculate_metrics(portfolio)
        
        # 评估性能
        evaluation = self.evaluate_performance(metrics)
        
        # 优化因子权重
        optimized_config = self.optimize_factor_weights()
        
        return {
            "portfolio": portfolio,
            "metrics": metrics,
            "evaluation": evaluation,
            "optimized_config": optimized_config
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
    
    def save_report(self, filepath: str, report: Dict = None):
        """保存分析报告"""
        if report is None:
            report = self.generate_report()
        
        # 确保报告中的所有数据都是可序列化的
        serializable_report = {}
        for k, v in report.items():
            if isinstance(v, pd.DataFrame):
                serializable_report[k] = v.to_dict()
            elif isinstance(v, pd.Series):
                serializable_report[k] = v.to_dict()
            elif isinstance(v, np.generic):
                serializable_report[k] = v.item()
            else:
                serializable_report[k] = v
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=4)

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_factor_analysis(
    start_time: str,
    end_time: str,
    instruments: str = "csi500",
    output_path: str = "stock-niubi/factor_analysis_results"
):
    """
    运行因子分析
    
    Parameters
    ----------
    start_time : str
        分析开始时间
    end_time : str
        分析结束时间
    instruments : str
        股票池，支持以下格式：
        - "csi500": 中证500成分股
        - "csi300": 沪深300成分股
        - "csi100": 中证100成分股
        - "all": 所有股票
    output_path : str
        输出路径
    """
    # 确保输出目录存在
    ensure_dir(output_path)
    
    # 获取股票池
    print(f"正在处理股票池: {instruments}")
    # 直接使用预定义的股票池名称，不需要转换为指数代码
    print(f"处理后的股票池: {instruments}")
    
    # 创建因子管理器
    print("正在创建因子管理器...")
    factor_manager = FactorManager(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time
    )
    print("因子管理器创建完成")
    
    # 创建回测评估器
    print("正在创建回测评估器...")
    evaluator = BacktestEvaluator(
        factor_manager=factor_manager,
        start_time=start_time,
        end_time=end_time,
        short_term_days=10,  # 短期目标：10天
        mid_term_days=60,    # 中期目标：60天
        target_short_return=0.10,  # 短期目标收益：10%
        target_mid_return=0.30,    # 中期目标收益：30%
        max_drawdown_threshold=0.15  # 最大回撤阈值：15%
    )
    print("回测评估器创建完成")
    
    # 生成分析报告
    print("正在生成分析报告...")
    report = evaluator.generate_report()
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_path}/factor_analysis_report_{timestamp}.json"
    evaluator.save_report(report_path, report)
    
    # 保存最新配置
    latest_config_path = f"{output_path}/latest_config.json"
    if isinstance(report, dict) and "optimized_config" in report:
        with open(latest_config_path, 'w', encoding='utf-8') as f_config:
            json.dump(report["optimized_config"], f_config, ensure_ascii=False, indent=4)
    
    return report_path, latest_config_path

if __name__ == "__main__":
    # 设置分析时间范围
    end_time = datetime.now().strftime("%Y-%m-%d")
    start_time = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # 运行因子分析
    report_path, config_path = run_factor_analysis(
        start_time=start_time,
        end_time=end_time,
        instruments="csi500",
        output_path="stock-niubi/factor_analysis_results"
    )
    
    print(f"因子分析报告已保存至：{report_path}")
    print(f"最新配置已保存至：{config_path}")