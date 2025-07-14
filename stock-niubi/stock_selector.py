"""
选股程序
基于因子分析结果进行选股
"""

import os
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd
from qlib.data import D
from qlib.contrib.factor_models import FactorManager

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

class StockSelector:
    """选股器"""
    
    def __init__(
        self,
        config_path: str,
        instruments: str = "csi500",
        max_stocks: int = 20
    ):
        """
        初始化选股器
        
        Parameters
        ----------
        config_path : str
            因子配置路径
        instruments : str
            股票池
        max_stocks : int
            最大选股数量
        """
        self.config_path = config_path
        self.instruments = instruments
        self.max_stocks = max_stocks
        
        # 加载因子配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def select_stocks(self) -> Dict[str, List[str]]:
        """
        选股
        
        Returns
        -------
        Dict[str, List[str]]
            选股结果，包含短期和中期选股列表
        """
        # 创建因子管理器
        factor_manager = FactorManager(
            instruments=self.instruments,
            start_time=None,  # 使用最新数据
            end_time=None,
            factor_weights=self.config["factor_weights"],
            scenario_weights=self.config["scenario_weights"]
        )
        
        # 获取因子暴露
        factor_exposures = factor_manager.get_factor_exposures()
        scenario_exposures = factor_manager.get_scenario_exposures()
        
        # 计算综合得分
        scores = pd.Series(0, index=factor_exposures.index)
        
        # 合并因子得分
        for factor, weight in self.config["factor_weights"].items():
            if factor in factor_exposures.columns:
                scores += weight * factor_exposures[factor]
        
        # 合并场景得分
        for scenario, weight in self.config["scenario_weights"].items():
            if scenario in scenario_exposures.columns:
                scores += weight * scenario_exposures[scenario]
        
        # 选择得分最高的股票
        selected_stocks = scores.nlargest(self.max_stocks)
        
        # 生成选股结果
        result = {
            "selection_date": datetime.now().strftime("%Y-%m-%d"),
            "short_term_stocks": selected_stocks.index.tolist()[:10],  # 短期选股
            "mid_term_stocks": selected_stocks.index.tolist(),        # 中期选股
            "stock_scores": selected_stocks.to_dict()
        }
        
        return result
    
    def save_selection(self, output_path: str) -> str:
        """
        保存选股结果
        
        Parameters
        ----------
        output_path : str
            输出路径
            
        Returns
        -------
        str
            输出文件路径
        """
        # 确保输出目录存在
        ensure_dir(output_path)
        
        # 选股
        selection = self.select_stocks()
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}/stock_selection_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selection, f, ensure_ascii=False, indent=4)
        
        return output_file

if __name__ == "__main__":
    # 创建选股器
    selector = StockSelector(
        config_path="factor_analysis_results/latest_config.json",
        instruments="csi500",
        max_stocks=20
    )
    
    # 保存选股结果
    output_file = selector.save_selection("stock_selection_results")
    
    print(f"选股结果已保存至：{output_file}") 