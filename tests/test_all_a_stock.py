from datetime import datetime
import os
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


def save_markdown_result(dataset, recorder, result_path):
    """
    输出分析结果到 markdown 文件，并附详细解释
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Qlib 量化分析结果（{now}）\n")

    # 1. 样本数据展示
    try:
        df = dataset.prepare("train")
        lines.append("## 1. 样本数据展示（前5行）\n")
        lines.append("```")
        lines.append(str(df.head()))
        lines.append("```")
        lines.append("字段说明：特征为因子，LABEL0为预测标签（如未来收益）\n")
    except Exception as e:
        lines.append(f"样本数据展示失败: {e}\n")

    # 2. 模型预测分数
    try:
        pred = recorder.load_object("pred.pkl")
        lines.append("## 2. 模型预测分数（前5行）\n")
        lines.append("```")
        lines.append(str(pred.head()))
        lines.append("```")
        lines.append("score 含义：模型预测该股票未来收益的分数，越高越看涨\n")
    except Exception as e:
        lines.append(f"模型预测分数展示失败: {e}\n")

    # 3. 因子分析
    try:
        ic = recorder.load_object("sig_analysis/ic.pkl")
        lines.append("## 3. 因子分析\n")
        lines.append("```")
        lines.append(str(ic))
        lines.append("```")
        lines.append(
            "**因子分析解释：**\n"
            "- **IC（信息系数）**：衡量模型预测值与真实收益的相关性，越高越好，0.05 以上为可用，0.1 以上为优秀。\n"
            "- **ICIR**：IC 的稳定性，越高越好。\n"
            "- **Rank IC**：预测排名与真实收益排名的相关性，抗异常值能力更强。\n"
            "- **Rank ICIR**：Rank IC 的稳定性。\n"
            "- **一般判断标准**：IC > 0.03，Rank IC > 0.03，说明模型有一定预测能力；ICIR、Rank ICIR > 0.3，说明预测能力较为稳定。\n"
        )
    except Exception as e:
        lines.append(f"因子分析展示失败: {e}\n")

    # 4. 回测绩效指标
    try:
        report = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        lines.append("## 4. 回测绩效指标\n")
        lines.append("```")
        lines.append(str(report))
        lines.append("```")
        lines.append(
            "**回测绩效指标解释：**\n"
            "- **mean**：日均收益率，正值表示整体赚钱。\n"
            "- **std**：日收益率标准差，衡量波动性，越小越稳。\n"
            "- **annualized_return**：年化收益率，0.15 表示年化收益 15%。\n"
            "- **information_ratio**：信息比率，>1 表示策略表现优异。\n"
            "- **max_drawdown**：最大回撤，负数，绝对值越小越好。\n"
        )
    except Exception as e:
        lines.append(f"回测绩效展示失败: {e}\n")

    # 5. 指标分析
    try:
        indicator = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
        lines.append("## 5. 指标分析\n")
        lines.append("```")
        lines.append(str(indicator))
        lines.append("```")
        lines.append(
            "**指标分析解释：**\n"
            "- **ffr**：持仓比例，1.0 表示全仓持有。\n"
            "- **pa**、**pos**：其他策略相关指标，具体含义可参考 Qlib 文档。\n"
        )
    except Exception as e:
        lines.append(f"指标分析展示失败: {e}\n")

    # 6. 股票排名分析
    try:
        pred = recorder.load_object("pred.pkl")
        pred_reset = pred.reset_index()
        # 每期top5
        topk = 5
        top_stocks = pred_reset.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime").head(topk)
        best_stocks = top_stocks["instrument"].value_counts().head(10)
        lines.append("## 6. 股票排名分析\n")
        lines.append("### 6.1 每期Top5出现次数最多的股票\n")
        lines.append("```")
        lines.append(str(best_stocks))
        lines.append("```")
        lines.append("出现次数越多，说明该股票在回测期间被模型持续看好。\n")

        # 全期平均分最高
        mean_score = pred_reset.groupby("instrument")["score"].mean().sort_values(ascending=False)
        lines.append("### 6.2 全期平均预测分数最高的股票\n")
        lines.append("```")
        lines.append(str(mean_score.head(10)))
        lines.append("```")
        lines.append("分数越高，模型越看好该股票未来表现。\n")
    except Exception as e:
        lines.append(f"股票排名分析失败: {e}\n")

    lines.append("\n---\n本文件由 Qlib 自动生成，便于理解和归档。\n")
    # 确保目录存在
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("开始初始化qlib。。。",flush=True)
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    print("with 语句块开始",flush=True)
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    print("with 语句块结束",flush=True)
    print("信号生成完成",flush=True)

    # 5. 保存分析结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/qlib_all_a_stock_analysis_{ts}.md"
    save_markdown_result(dataset, recorder, result_path)
    print(f"分析结果已保存到: {result_path}",flush=True)

if __name__ == "__main__":
    try:
        print("开始运行主流程",flush=True)
        main()
        print("主流程运行完成",flush=True)
    except Exception as e:
        print(f"主流程异常终止: {e}",flush=True)
        import traceback
        traceback.print_exc()
