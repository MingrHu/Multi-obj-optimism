"""NSGA-II 遗传算法优化：代理模型问题定义、自适应算子与运行入口。"""

from .problem import ConstraintSpec, ObjectiveSpec, SurrogateOptimizationProblem
from .operators import AdaptiveSBX, save_pareto_solutions

__all__ = [
    "ConstraintSpec",
    "ObjectiveSpec",
    "SurrogateOptimizationProblem",
    "AdaptiveSBX",
    "save_pareto_solutions",
]
