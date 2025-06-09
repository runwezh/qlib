# Qlib

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![PypI Versions](https://img.shields.io/pypi/v/pyqlib)](https://pypi.org/project/pyqlib/#history)
[![Upload Python Package](https://github.com/microsoft/qlib/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/pyqlib/)
[![Github Actions Test Status](https://github.com/microsoft/qlib/workflows/Test/badge.svg?branch=main)](https://github.com/microsoft/qlib/actions)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/qlib](https://badges.gitter.im/Microsoft/qlib.svg)](https://gitter.im/Microsoft/qlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## :newspaper: **最新动态！** &nbsp;   :sparkling_heart: 

最近发布的功能特性

### 介绍 [RD-Agent](https://github.com/microsoft/RD-Agent)：基于LLM的工业数据驱动研发自主进化智能体

我们很高兴宣布发布 **RD-Agent**📢，这是一个强大的工具，支持量化投资研发中的自动化因子挖掘和模型优化。

RD-Agent 现已在 [GitHub](https://github.com/microsoft/RD-Agent) 上发布，欢迎您的星标🌟！

要了解更多信息，请访问我们的 [♾️演示页面](https://rdagent.azurewebsites.net/)。在这里，您将找到中英文演示视频，帮助您更好地理解RD-Agent的应用场景和使用方法。

我们为您准备了几个演示视频：
| 应用场景 | 演示视频（英文） | 演示视频（中文） |
| --                      | ------    | ------    |
| 量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=zh) |
| 基于报告的量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/report_factor?lang=en) | [链接](https://rdagent.azurewebsites.net/report_factor?lang=zh) |
| 量化模型优化 | [链接](https://rdagent.azurewebsites.net/model_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/model_loop?lang=zh) |

- 📃**论文**: [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
- 👾**代码**: https://github.com/microsoft/RD-Agent/
```BibTeX
@misc{li2025rdagentquant,
    title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
    author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Weiqing Liu and Jiang Bian},
    year={2025},
    eprint={2505.15155},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
![image](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

***

| 功能特性 | 状态 |
| --                      | ------    |
| [R&D-Agent-Quant](https://arxiv.org/abs/2505.15155) 已发布 | 将R&D-Agent应用于Qlib进行量化交易 | 
| 端到端学习的BPQP | 📈即将推出！([审核中](https://github.com/microsoft/qlib/pull/1863)) |
| 🔥LLM驱动的自动量化工厂🔥 | 🚀 已在 [♾️RD-Agent](https://github.com/microsoft/RD-Agent) 中发布，2024年8月8日 |
| KRNN和Sandwich模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/1414/)，2023年5月26日 |
| 发布Qlib v0.9.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.9.0)，2022年12月9日 |
| 强化学习框架 | :hammer: :chart_with_upwards_trend: 2022年11月10日发布。[#1332](https://github.com/microsoft/qlib/pull/1332), [#1322](https://github.com/microsoft/qlib/pull/1322), [#1316](https://github.com/microsoft/qlib/pull/1316),[#1299](https://github.com/microsoft/qlib/pull/1299),[#1263](https://github.com/microsoft/qlib/pull/1263), [#1244](https://github.com/microsoft/qlib/pull/1244), [#1169](https://github.com/microsoft/qlib/pull/1169), [#1125](https://github.com/microsoft/qlib/pull/1125), [#1076](https://github.com/microsoft/qlib/pull/1076)|
| HIST和IGMTF模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/1040)，2022年4月10日 |
| Qlib [笔记本教程](https://github.com/microsoft/qlib/tree/main/examples/tutorial) | 📖 [已发布](https://github.com/microsoft/qlib/pull/1037)，2022年4月7日 | 
| Ibovespa指数数据 | :rice: [已发布](https://github.com/microsoft/qlib/pull/990)，2022年4月6日 |
| 时点数据库 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/343)，2022年3月10日 |
| Arctic提供商后端和订单簿数据示例 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/744)，2022年1月17日 |
| 基于元学习的框架和DDG-DA  | :chart_with_upwards_trend:  :hammer: [已发布](https://github.com/microsoft/qlib/pull/743)，2022年1月10日 | 
| 基于规划的投资组合优化 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/754)，2021年12月28日 | 
| 发布Qlib v0.8.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.8.0)，2021年12月8日 |
| ADD模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/704)，2021年11月22日 |
| ADARNN模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/689)，2021年11月14日 |
| TCN模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/668)，2021年11月4日 |
| 嵌套决策框架 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/438)，2021年10月1日。[示例](https://github.com/microsoft/qlib/blob/main/examples/nested_decision_execution/workflow.py) 和 [文档](https://qlib.readthedocs.io/en/latest/component/highfreq.html) |
| 时间路由适配器(TRA) | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/531)，2021年7月30日 |
| Transformer和Localformer | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/508)，2021年7月22日 |
| 发布Qlib v0.7.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.7.0)，2021年7月12日 |
| TCTS模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/491)，2021年7月1日 |
| 在线服务和自动模型滚动 | :hammer:  [已发布](https://github.com/microsoft/qlib/pull/290)，2021年5月17日 | 
| DoubleEnsemble模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/286)，2021年3月2日 | 
| 高频数据处理示例 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/257)，2021年2月5日  |
| 高频交易示例 | :chart_with_upwards_trend: [部分代码已发布](https://github.com/microsoft/qlib/pull/227)，2021年1月28日  | 
| 高频数据(1分钟) | :rice: [已发布](https://github.com/microsoft/qlib/pull/221)，2021年1月27日 |
| Tabnet模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/205)，2021年1月22日 |

2021年之前发布的功能特性未在此列出。

<p align="center">
  <img src="docs/_static/img/logo/1.png" />
</p>

Qlib是一个开源的、面向AI的量化投资平台，旨在通过AI技术在量化投资中实现潜力、赋能研究并创造价值，从探索想法到实施生产。Qlib支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。

越来越多的SOTA量化研究工作/论文在不同范式中被发布到Qlib中，以协作解决量化投资中的关键挑战。例如，1）使用监督学习从丰富和异构的金融数据中挖掘市场复杂的非线性模式，2）使用自适应概念漂移技术建模金融市场的动态性质，3）使用强化学习建模连续投资决策并协助投资者优化其交易策略。

它包含数据处理、模型训练、回测的完整ML流水线；并涵盖量化投资的整个链条：阿尔法挖掘、风险建模、投资组合优化和订单执行。
更多详情，请参考我们的论文 ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189)。

<table>
  <tbody>
    <tr>
      <th>框架、教程、数据和DevOps</th>
      <th>量化研究中的主要挑战和解决方案</th>
    </tr>
    <tr>
      <td>
        <li><a href="#计划"><strong>计划</strong></a></li>
        <li><a href="#qlib框架">Qlib框架</a></li>
        <li><a href="#快速开始">快速开始</a></li>
          <ul dir="auto">
            <li type="circle"><a href="#安装">安装</a> </li>
            <li type="circle"><a href="#数据准备">数据准备</a></li>
            <li type="circle"><a href="#自动量化研究工作流">自动量化研究工作流</a></li>
            <li type="circle"><a href="#通过代码构建定制化量化研究工作流">通过代码构建定制化量化研究工作流</a></li></ul>
        <li><a href="#量化数据集动物园"><strong>量化数据集动物园</strong></a></li>
        <li><a href="#学习框架">学习框架</a></li>
        <li><a href="#更多关于qlib">更多关于Qlib</a></li>
        <li><a href="#离线模式和在线模式">离线模式和在线模式</a>
        <ul>
          <li type="circle"><a href="#qlib数据服务器性能">Qlib数据服务器性能</a></li></ul>
        <li><a href="#相关报告">相关报告</a></li>
        <li><a href="#联系我们">联系我们</a></li>
        <li><a href="#贡献">贡献</a></li>
      </td>
      <td valign="baseline">
        <li><a href="#量化研究中的主要挑战和解决方案">量化研究中的主要挑战和解决方案</a>
          <ul>
            <li type="circle"><a href="#预测寻找有价值的信号模式">预测：寻找有价值的信号/模式</a>
              <ul>
                <li type="disc"><a href="#量化模型论文动物园"><strong>量化模型（论文）动物园</strong></a>
                  <ul>
                    <li type="circle"><a href="#运行单个模型">运行单个模型</a></li>
                    <li type="circle"><a href="#运行多个模型">运行多个模型</a></li>
                  </ul>
                </li>
              </ul>
            </li>
          <li type="circle"><a href="#适应市场动态">适应市场动态</a></li>
          <li type="circle"><a href="#强化学习建模连续决策">强化学习：建模连续决策</a></li>
          </ul>
        </li>
      </td>
    </tr>
  </tbody>
</table>

# 计划
正在开发的新功能（按预计发布时间排序）。
您对这些功能的反馈非常重要。

# Qlib框架

<div style="align: center">
<img src="docs/_static/img/framework-abstract.jpg" />
</div>

Qlib的高级框架如上所示（用户可以在深入了解细节时找到Qlib设计的[详细框架](https://qlib.readthedocs.io/en/latest/introduction/introduction.html#framework)）。
组件被设计为松耦合模块，每个组件都可以独立使用。

Qlib提供强大的基础设施来支持量化研究。[数据](https://qlib.readthedocs.io/en/latest/component/data.html)始终是重要的组成部分。
设计了强大的学习框架来支持不同的学习范式（例如[强化学习](https://qlib.readthedocs.io/en/latest/component/rl.html)、[监督学习](https://qlib.readthedocs.io/en/latest/component/workflow.html#model-section)）和不同层次的模式（例如[市场动态建模](https://qlib.readthedocs.io/en/latest/component/meta.html)）。
通过建模市场，[交易策略](https://qlib.readthedocs.io/en/latest/component/strategy.html)将生成将被执行的交易决策。不同层次或粒度的多个交易策略和执行器可以[嵌套优化并一起运行](https://qlib.readthedocs.io/en/latest/component/highfreq.html)。
最后，将提供全面的[分析](https://qlib.readthedocs.io/en/latest/component/report.html)，模型可以以低成本[在线服务](https://qlib.readthedocs.io/en/latest/component/online.html)。

# 快速开始

本快速入门指南试图演示：
1. 使用 _Qlib_ 构建完整的量化研究工作流并尝试您的想法非常容易。
2. 尽管使用*公共数据*和*简单模型*，机器学习技术在实际量化投资中**工作得非常好**。

这里有一个快速**[演示](https://terminalizer.com/view/3f24561a4470)**展示如何安装``Qlib``，并使用``qrun``运行LightGBM。**但是**，请确保您已经按照[说明](#数据准备)准备了数据。

## 安装

此表展示了`Qlib`支持的Python版本：
|               | 使用pip安装      | 从源码安装  |        绘图        |
| ------------- |:---------------------:|:--------------------:|:------------------:|
| Python 3.8    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.10   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.11   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.12   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |

**注意**: 
1. 建议使用**Conda**来管理您的Python环境。在某些情况下，在`conda`环境之外使用Python可能会导致缺少头文件，从而导致某些包的安装失败。
2. 请注意，在Python 3.6中安装cython在从源码安装``Qlib``时会出现一些错误。如果用户在其机器上使用Python 3.6，建议*升级*Python到3.8或更高版本，或使用`conda`的Python从源码安装``Qlib``。

### 使用pip安装
用户可以根据以下命令通过pip轻松安装``Qlib``。

```bash
  pip install pyqlib
```

**注意**: pip将安装最新的稳定版qlib。但是，qlib的主分支正在积极开发中。如果您想测试主分支中的最新脚本或功能，请使用下面的方法安装qlib。

### 从源码安装
此外，用户可以根据以下步骤通过源代码安装最新的开发版本``Qlib``：

* 在从源码安装``Qlib``之前，用户需要安装一些依赖项：

  ```bash
  pip install numpy
  pip install --upgrade cython
  ```

* 克隆仓库并按如下方式安装``Qlib``。
    ```bash
    git clone https://github.com/microsoft/qlib.git && cd qlib
    pip install .  # 推荐开发使用 `pip install -e .[dev]`。详情请查看 docs/developer/code_standard_and_dev_guide.rst
    ```

**提示**: 如果您在环境中安装`Qlib`或运行示例失败，比较您的步骤和[CI工作流](.github/workflows/test_qlib_from_source.yml)可能会帮助您找到问题。

**Mac提示**: 如果您使用带有M1的Mac，您可能会在构建LightGBM的wheel时遇到问题，这是由于缺少OpenMP的依赖项。要解决这个问题，首先使用``brew install libomp``安装openmp，然后运行``pip install .``成功构建它。

## 数据准备
❗ 由于更严格的数据安全政策，官方数据集暂时被禁用。您可以尝试社区贡献的[这个数据源](https://github.com/chenditc/investment_data/releases)。
这里是下载最新数据的示例。
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

下面的官方数据集将在不久的将来恢复。

----

通过运行以下代码加载和准备数据：

### 使用模块获取
  ```bash
  # 获取1日数据
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取1分钟数据
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

### 从源码获取

  ```bash
  # 获取1日数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取1分钟数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

此数据集由[爬虫脚本](scripts/data_collector/)收集的公共数据创建，这些脚本已在同一仓库中发布。
用户可以使用它创建相同的数据集。[数据集描述](https://github.com/microsoft/qlib/tree/main/scripts/data_collector#description-of-dataset)

*请**注意**数据是从[Yahoo Finance](https://finance.yahoo.com/lookup)收集的，数据可能不完美。
如果用户有高质量的数据集，我们建议用户准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*。

### 日频数据的自动更新（来自yahoo finance）
  > 如果用户只想在历史数据上尝试他们的模型和策略，这一步是*可选的*。
  > 
  > 建议用户手动更新一次数据（--trading_date 2021-05-25），然后设置为自动更新。
  >
  > **注意**: 用户无法基于Qlib提供的离线数据增量更新数据（一些字段被删除以减少数据大小）。用户应该使用[yahoo收集器](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)从头下载Yahoo数据，然后增量更新它。
  > 
  > 更多信息，请参考：[yahoo收集器](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)

  * 每个交易日自动更新数据到"qlib"目录（Linux）
      * 使用*crontab*: `crontab -e`
      * 设置定时任务：

        ```
        * * * * 1-5 python <脚本路径> update_data_to_bin --qlib_data_1d_dir <用户数据目录>
        ```
        * **脚本路径**: *scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <用户数据目录> --trading_date <开始日期> --end_date <结束日期>
      ```
      * *trading_date*: 交易日开始
      * *end_date*: 交易日结束（不包括）

### 检查数据健康状况
  * 我们提供了一个脚本来检查数据的健康状况，您可以运行以下命令来检查数据是否健康。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
    ```
  * 当然，您也可以添加一些参数来调整测试结果，比如这样。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --missing_data_num 30055 --large_step_threshold_volume 94485 --large_step_threshold_price 20
    ```
  * 如果您想了解更多关于`check_data_health`的信息，请参考[文档](https://qlib.readthedocs.io/en/latest/component/data.html#checking-the-health-of-the-data)。

## Docker镜像
1. 从docker hub仓库拉取docker镜像
    ```bash
    docker pull pyqlib/qlib_image_stable:stable
    ```
2. 启动新的Docker容器
    ```bash
    docker run -it --name <容器名称> -v <挂载的本地目录>:/app qlib_image_stable
    ```
3. 此时您在docker环境中，可以运行qlib脚本。示例：
    ```bash
    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
4. 退出容器
    ```bash
    >>> exit
    ```
5. 重启容器
    ```bash
    docker start -i -a <容器名称>
    ```
6. 停止容器
    ```bash
    docker stop <容器名称>
    ```
7. 删除容器
    ```bash
    docker rm <容器名称>
    ```
8. 如果您想了解更多信息，请参考[文档](https://qlib.readthedocs.io/en/latest/developer/how_to_build_image.html)。

## 自动量化研究工作流
Qlib提供了一个名为`qrun`的工具来自动运行整个工作流（包括构建数据集、训练模型、回测和评估）。您可以根据以下步骤启动自动量化研究工作流并获得图形报告分析：

1. 量化研究工作流：使用lightgbm工作流配置（[workflow_config_lightgbm_Alpha158.yaml](examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)）运行`qrun`，如下所示。
    ```bash
      cd examples  # 避免在包含`qlib`的目录下运行程序
      qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    如果用户想在调试模式下使用`qrun`，请使用以下命令：
    ```bash
    python -m pdb qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    `qrun`的结果如下，请参考[文档](https://qlib.readthedocs.io/en/latest/component/strategy.html#result)了解更多关于结果的解释。

    ```bash

    '以下是无成本超额收益的分析结果。'
                           risk
    mean               0.000708
    std                0.005626
    annualized_return  0.178316
    information_ratio  1.996555
    max_drawdown      -0.081806
    '以下是有成本超额收益的分析结果。'
                           risk
    mean               0.000512
    std                0.005626
    annualized_return  0.128982
    information_ratio  1.444287
    max_drawdown      -0.091078
    ```
    这里有`qrun`和[工作流](https://qlib.readthedocs.io/en/latest/component/workflow.html)的详细文档。

2. 图形报告分析：首先，运行`python -m pip install .[analysis]`安装所需的依赖项。然后使用`jupyter notebook`运行`examples/workflow_by_code.ipynb`获得图形报告。
    - 预测信号（模型预测）分析
      - 分组累积收益
      ![累积收益](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_cumulative_return.png)
      - 收益分布
      ![多空](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_long_short.png)
      - 信息系数(IC)
      ![信息系数](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_IC.png)
      ![月度IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_monthly_IC.png)
      ![IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_NDQ.png)
      - 预测信号（模型预测）的自相关
      ![自相关](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_auto_correlation.png)

    - 投资组合分析
      - 回测收益
      ![报告](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/report.png)
   - [上述结果的解释](https://qlib.readthedocs.io/en/latest/component/report.html)

## 通过代码构建定制化量化研究工作流
自动工作流可能不适合所有量化研究人员的研究工作流。为了支持灵活的量化研究工作流，Qlib还提供了模块化接口，允许研究人员通过代码构建自己的工作流。[这里](examples/workflow_by_code.ipynb)是通过代码定制量化研究工作流的演示。

# 量化研究中的主要挑战和解决方案
量化投资是一个非常独特的场景，有很多关键挑战需要解决。
目前，Qlib为其中几个提供了一些解决方案。

## 预测：寻找有价值的信号/模式
准确预测股价趋势是构建盈利投资组合的重要组成部分。
然而，金融市场中大量不同格式的数据使得构建预测模型具有挑战性。

越来越多的SOTA量化研究工作/论文专注于构建预测模型以在复杂的金融数据中挖掘有价值的信号/模式，这些工作在`Qlib`中发布。

### [量化模型（论文）动物园](examples/benchmarks)

这里是基于`Qlib`构建的模型列表。
- [基于XGBoost的GBDT (Tianqi Chen, et al. KDD 2016)](examples/benchmarks/XGBoost/)
- [基于LightGBM的GBDT (Guolin Ke, et al. NIPS 2017)](examples/benchmarks/LightGBM/)
- [基于Catboost的GBDT (Liudmila Prokhorenkova, et al. NIPS 2018)](examples/benchmarks/CatBoost/)
- [基于pytorch的MLP](examples/benchmarks/MLP/)
- [基于pytorch的LSTM (Sepp Hochreiter, et al. Neural computation 1997)](examples/benchmarks/LSTM/)
- [基于pytorch的GRU (Kyunghyun Cho, et al. 2014)](examples/benchmarks/GRU/)
- [基于pytorch的ALSTM (Yao Qin, et al. IJCAI 2017)](examples/benchmarks/ALSTM)
- [基于pytorch的GATs (Petar Velickovic, et al. 2017)](examples/benchmarks/GATs/)
- [基于pytorch的SFM (Liheng Zhang, et al. KDD 2017)](examples/benchmarks/SFM/)
- [基于tensorflow的TFT (Bryan Lim, et al. International Journal of Forecasting 2019)](examples/benchmarks/TFT/)
- [基于pytorch的TabNet (Sercan O. Arik, et al. AAAI 2019)](examples/benchmarks/TabNet/)
- [基于LightGBM的DoubleEnsemble (Chuheng Zhang, et al. ICDM 2020)](examples/benchmarks/DoubleEnsemble/)
- [基于pytorch的TCTS (Xueqing Wu, et al. ICML 2021)](examples/benchmarks/TCTS/)
- [基于pytorch的Transformer (Ashish Vaswani, et al. NeurIPS 2017)](examples/benchmarks/Transformer/)
- [基于pytorch的Localformer (Juyong Jiang, et al.)](examples/benchmarks/Localformer/)
- [基于pytorch的TRA (Hengxu, Dong, et al. KDD 2021)](examples/benchmarks/TRA/)
- [基于pytorch的TCN (Shaojie Bai, et al. 2018)](examples/benchmarks/TCN/)
- [基于pytorch的ADARNN (YunTao Du, et al. 2021)](examples/benchmarks/ADARNN/)
- [基于pytorch的ADD (Hongshun Tang, et al.2020)](examples/benchmarks/ADD/)
- [基于pytorch的IGMTF (Wentao Xu, et al.2021)](examples/benchmarks/IGMTF/)
- [基于pytorch的HIST (Wentao Xu, et al.2021)](examples/benchmarks/HIST/)
- [基于pytorch的KRNN](examples/benchmarks/KRNN/)
- [基于pytorch的Sandwich](examples/benchmarks/Sandwich/)

非常欢迎您提交新量化模型的PR。

每个模型在`Alpha158`和`Alpha360`数据集上的性能可以在[这里](examples/benchmarks/README.md)找到。

### 运行单个模型
上面列出的所有模型都可以使用``Qlib``运行。用户可以通过[benchmarks](examples/benchmarks)文件夹找到我们提供的配置文件和关于模型的一些详细信息。更多信息可以在上面列出的模型文件中检索。

`Qlib`提供三种不同的方式来运行单个模型，用户可以选择最适合他们情况的方式：
- 用户可以使用上面提到的工具`qrun`基于配置文件运行模型的工作流。
- 用户可以基于`examples`文件夹中列出的[一个](examples/workflow_by_code.py)创建`workflow_by_code` python脚本。

- 用户可以使用`examples`文件夹中列出的脚本[`run_all_model.py`](examples/run_all_model.py)来运行模型。这里是要使用的特定shell命令的示例：`python run_all_model.py run --models=lightgbm`，其中`--models`参数可以接受上面列出的任意数量的模型（可用模型可以在[benchmarks](examples/benchmarks/)中找到）。更多用例，请参考文件的[文档字符串](examples/run_all_model.py)。
    - **注意**: 每个基线都有不同的环境依赖项，请确保您的python版本与要求一致（例如，由于`tensorflow==1.15.0`的限制，TFT仅支持Python 3.6~3.7）

### 运行多个模型
`Qlib`还提供了一个脚本[`run_all_model.py`](examples/run_all_model.py)，可以运行多个模型进行多次迭代。（**注意**: 脚本目前仅支持*Linux*。其他操作系统将在未来得到支持。此外，它也不支持并行运行同一模型多次，这将在未来的开发中修复。）

脚本将为每个模型创建一个唯一的虚拟环境，并在训练后删除环境。因此，只会生成和存储实验结果，如`IC`和`回测`结果。

这里是运行所有模型10次迭代的示例：
```python
python run_all_model.py run 10
```

它还提供API来一次运行特定模型。更多用例，请参考文件的[文档字符串](examples/run_all_model.py)。

### 重大变更
在`pandas`中，`group_key`是`groupby`方法的参数之一。从`pandas`的1.5版本到2.0版本，`group_key`的默认值从`no default`更改为`True`，这将导致qlib在操作期间报错。所以我们设置`group_key=False`，但它不能保证某些程序能正确运行，包括：
* qlib\examples\rl_order_execution\scripts\gen_training_orders.py
* qlib\examples\benchmarks\TRA\src\dataset.MTSDatasetH.py
* qlib\examples\benchmarks\TFT\tft.py

## [适应市场动态](examples/benchmarks_dynamic)

由于金融市场环境的非平稳性质，数据分布可能在不同时期发生变化，这使得基于训练数据构建的模型在未来测试数据中的性能下降。
因此，使预测模型/策略适应市场动态对模型/策略的性能非常重要。

这里是基于`Qlib`构建的解决方案列表。
- [滚动重训练](examples/benchmarks_dynamic/baseline/)
- [基于pytorch的DDG-DA (Wendi, et al. AAAI 2022)](examples/benchmarks_dynamic/DDG-DA/)

## 强化学习：建模连续决策
Qlib现在支持强化学习，这是一个旨在建模连续投资决策的功能。此功能通过从与环境的交互中学习来协助投资者优化其交易策略，以最大化某种累积奖励的概念。

这里是基于`Qlib`构建的按场景分类的解决方案列表。

### [订单执行的强化学习](examples/rl_order_execution)
[这里](https://qlib.readthedocs.io/en/latest/component/rl/overall.html#order-execution)是此场景的介绍。下面的所有方法都在[这里](examples/rl_order_execution)进行了比较。
- [TWAP](examples/rl_order_execution/exp_configs/backtest_twap.yml)
- [PPO: "An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization", IJCAL 2020](examples/rl_order_execution/exp_configs/backtest_ppo.yml)
- [OPDS: "Universal Trading for Order Execution with Oracle Policy Distillation", AAAI 2021](examples/rl_order_execution/exp_configs/backtest_opds.yml)

# 量化数据集动物园
数据集在量化中起着非常重要的作用。这里是基于`Qlib`构建的数据集列表：

| 数据集                                    | 美国市场 | 中国市场 |
| --                                         | --        | --           |
| [Alpha360](./qlib/contrib/data/handler.py) |  √        |  √           |
| [Alpha158](./qlib/contrib/data/handler.py) |  √        |  √           |

[这里](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)是使用`Qlib`构建数据集的教程。
非常欢迎您提交构建新量化数据集的PR。

# 学习框架
Qlib高度可定制，其许多组件都是可学习的。
可学习组件是`预测模型`和`交易智能体`的实例。它们基于`学习框架`层学习，然后应用于`工作流`层中的多个场景。
学习框架也利用了`工作流`层（例如共享`信息提取器`，基于`执行环境`创建环境）。

基于学习范式，它们可以分为强化学习和监督学习。
- 对于监督学习，详细文档可以在[这里](https://qlib.readthedocs.io/en/latest/component/model.html)找到。
- 对于强化学习，详细文档可以在[这里](https://qlib.readthedocs.io/en/latest/component/rl.html)找到。Qlib的RL学习框架利用`工作流`层中的`执行环境`来创建环境。值得注意的是，也支持`嵌套执行器`。这使用户能够一起优化不同级别的策略/模型/智能体（例如，为特定的投资组合管理策略优化订单执行策略）。

# 更多关于Qlib
如果您想快速了解qlib最常用的组件，可以尝试[这里](examples/tutorial/)的笔记本。

详细文档组织在[docs](docs/)中。
需要[Sphinx](http://www.sphinx-doc.org)和readthedocs主题来构建html格式的文档。
```bash
cd docs/
conda install sphinx sphinx_rtd_theme -y
# 或者，您可以使用pip安装它们
# pip install sphinx sphinx_rtd_theme
make html
```
您也可以直接在线查看[最新文档](http://qlib.readthedocs.io/)。

Qlib正在积极持续开发中。我们的计划在路线图中，作为[github项目](https://github.com/microsoft/qlib/projects/1)管理。

# 离线模式和在线模式
Qlib的数据服务器可以部署为`离线`模式或`在线`模式。默认模式是离线模式。

在`离线`模式下，数据将在本地部署。

在`在线`模式下，数据将作为共享数据服务部署。数据及其缓存将被所有客户端共享。由于更高的缓存命中率，数据检索性能预计会得到改善。它也会消耗更少的磁盘空间。在线模式的文档可以在[Qlib-Server](https://qlib-server.readthedocs.io/)中找到。在线模式可以使用[基于Azure CLI的脚本](https://qlib-server.readthedocs.io/en/latest/build.html#one-click-deployment-in-azure)自动部署。在线数据服务器的源代码可以在[Qlib-Server仓库](https://github.com/microsoft/qlib-server)中找到。

## Qlib数据服务器性能
数据处理的性能对于像AI技术这样的数据驱动方法很重要。作为面向AI的平台，Qlib为数据存储和数据处理提供了解决方案。为了演示Qlib数据服务器的性能，我们将其与其他几种数据存储解决方案进行比较。

我们通过完成相同的任务来评估几种存储解决方案的性能，该任务从股票市场的基本OHLCV日数据（从2007年到2020年每天800只股票）创建数据集（14个特征/因子）。该任务涉及数据查询和处理。

|                         | HDF5      | MySQL     | MongoDB   | InfluxDB  | Qlib -E -D  | Qlib +E -D   | Qlib +E +D  |
| --                      | ------    | ------    | --------  | --------- | ----------- | ------------ | ----------- |
| 总计（1CPU）（秒）  | 184.4±3.7 | 365.3±7.5 | 253.6±6.7 | 368.2±3.6 | 147.0±8.8   | 47.6±1.0     | **7.4±0.3** |
| 总计（64CPU）（秒） |           |           |           |           | 8.8±0.6     | **4.2±0.2**  |             |
* `+(-)E`表示有（无）`ExpressionCache`
* `+(-)D`表示有（无）`DatasetCache`

大多数通用数据库加载数据需要太多时间。在查看底层实现后，我们发现在通用数据库解决方案中，数据经过太多层接口和不必要的格式转换。
这种开销大大减慢了数据加载过程。
Qlib数据以紧凑格式存储，这对于组合成科学计算数组是高效的。

# 相关报告
- [Guide To Qlib: Microsoft's AI Investment Platform](https://analyticsindiamag.com/qlib/)
- [微软也搞AI量化平台？还是开源的！](https://mp.weixin.qq.com/s/47bP5YwxfTp2uTHjUBzJQQ)
- [微矿Qlib：业内首个AI量化投资开源平台](https://mp.weixin.qq.com/s/vsJv7lsgjEi-ALYUz4CvtQ)

# 联系我们
- 如果您有任何问题，请在[这里](https://github.com/microsoft/qlib/issues/new/choose)创建issue或在[gitter](https://gitter.im/Microsoft/qlib)中发送消息。
- 如果您想为`Qlib`做贡献，请[创建pull request](https://github.com/microsoft/qlib/compare)。
- 出于其他原因，欢迎您通过电子邮件联系我们([qlib@microsoft.com](mailto:qlib@microsoft.com))。
  - 我们正在招聘新成员（全职员工和实习生），欢迎您的简历！

加入即时通讯讨论群：
|[Gitter](https://gitter.im/Microsoft/qlib)|
|----|
|![image](https://github.com/microsoft/qlib/blob/main/docs/_static/img/qrcode/gitter_qr.png)|

# 贡献
我们感谢所有贡献并感谢所有贡献者！
<a href="https://github.com/microsoft/qlib/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/qlib" /></a>

在我们于2020年9月在Github上发布Qlib作为开源项目之前，Qlib是我们组内的内部项目。不幸的是，内部提交历史没有保留。我们组的许多成员也为Qlib贡献了很多，包括Ruihua Wang、Yinda Zhang、Haisu Yu、Shuyu Wang、Bochen Pang和[Dong Zhou](https://github.com/evanzd/evanzd)。特别感谢[Dong Zhou](https://github.com/evanzd/evanzd)，因为他的Qlib初始版本。

## 指导

这个项目欢迎贡献和建议。
**这里有一些[代码标准和开发指导](docs/developer/code_standard_and_dev_guide.rst)用于提交pull request。**

做贡献并不是一件困难的事情。解决一个问题（也许只是回答[问题列表](https://github.com/microsoft/qlib/issues)或[gitter](https://gitter.im/Microsoft/qlib)中提出的问题），修复/提出错误，改进文档甚至修复拼写错误都是对Qlib的重要贡献。

例如，如果您想为Qlib的文档/代码做贡献，可以按照下图中的步骤操作。
<p align="center">
  <img src="https://github.com/demon143/qlib/blob/main/docs/_static/img/change%20doc.gif" />
</p>

如果您不知道如何开始贡献，可以参考以下示例。
| 类型 | 示例 |
| -- | -- |
| 解决问题 | [回答问题](https://github.com/microsoft/qlib/issues/749)；[提出](https://github.com/microsoft/qlib/issues/765)或[修复](https://github.com/microsoft/qlib/pull/792)错误 |
| 文档 | [改进文档质量](https://github.com/microsoft/qlib/pull/797/files)；[修复拼写错误](https://github.com/microsoft/qlib/pull/774) | 
| 功能 | 实现[请求的功能](https://github.com/microsoft/qlib/projects)，如[这个](https://github.com/microsoft/qlib/pull/754)；[重构接口](https://github.com/microsoft/qlib/pull/539/files) |
| 数据集 | [添加数据集](https://github.com/microsoft/qlib/pull/733) | 
| 模型 | [实现新模型](https://github.com/microsoft/qlib/pull/689)，[贡献模型的一些说明](https://github.com/microsoft/qlib/tree/main/examples/benchmarks#contributing) |

[Good first issues](https://github.com/microsoft/qlib/labels/good%20first%20issue)被标记为表示它们很容易开始您的贡献。

您可以通过`rg 'TODO|FIXME' qlib`在Qlib中找到一些不完善的实现

如果您想成为Qlib的维护者之一以贡献更多（例如帮助合并PR，分类问题），请通过电子邮件联系我们([qlib@microsoft.com](mailto:qlib@microsoft.com))。我们很乐意帮助升级您的权限。

## 许可证
大多数贡献要求您同意贡献者许可协议（CLA），声明您有权利并实际上授予我们使用您贡献的权利。详情请访问https://cla.opensource.microsoft.com。

当您提交pull request时，CLA机器人将自动确定您是否需要提供CLA并适当地装饰PR（例如，状态检查，评论）。只需按照机器人提供的说明操作。您只需要在使用我们CLA的所有仓库中执行一次此操作。

此项目采用了[Microsoft开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
更多信息请参见[行为准则FAQ](https://opensource.microsoft.com/codeofconduct/faq/)或联系[opencode@microsoft.com](mailto:opencode@microsoft.com)获取任何其他问题或评论。