"""mobo —— 多目标锻造工艺优化工具包。

分层结构：
- ``mobo.common``：集中式路径解析与全局日志。
- ``mobo.surrogate``：代理模型（DNN/多项式/SVR/随机森林/Kriging）训练与评估。
- ``mobo.optimization``：NSGA-II 遗传算法与强化学习优化器。
- ``mobo.extraction``：按工件类型分派的 KEY 文件目标提取原子能力层。
- ``mobo.automation``：DEFORM 采样/求解/提取自动化流水线。
- ``mobo.cli``：命令行入口。
"""

__version__ = "1.1.0"

__all__ = ["__version__"]
