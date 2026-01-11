import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.sbx import SBX

import numpy as np
from pymoo.operators.crossover.sbx import SBX


def save_pareto_solutions(
    res,
    filename="../../data/pareto_solutions.txt",
    *,
    var_names=None,
    obj_names=None,
):
    X = np.asarray(res.X)
    F = np.asarray(res.F)

    n_var = int(X.shape[1])
    n_obj = int(F.shape[1])

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n_var)]
    if obj_names is None:
        obj_names = [f"f{i+1}" for i in range(n_obj)]

    if len(var_names) != n_var:
        raise ValueError("var_names 长度必须等于 res.X 的列数")
    if len(obj_names) != n_obj:
        raise ValueError("obj_names 长度必须等于 res.F 的列数")

    has_constraints = getattr(res, "G", None) is not None
    if has_constraints:
        G = np.asarray(res.G)
        constraint_violations = np.sum(G > 0, axis=1)
    else:
        constraint_violations = np.zeros(X.shape[0], dtype=int)

    with open(filename, "a", encoding="utf-8") as f:
        f.write("Pareto前沿最优解集\n")
        f.write("=" * 80 + "\n")
        f.write(f"解数量: {len(X)}\n")
        f.write(f"变量数: {n_var}, 目标数: {n_obj}\n")
        f.write("=" * 80 + "\n\n")

        header_cols = ["序号"] + list(var_names) + list(obj_names) + ["约束违反"]
        f.write(" | ".join(header_cols) + "\n")
        f.write("-" * max(100, len(" | ".join(header_cols)) + 10) + "\n")

        for i in range(len(X)):
            row = [f"{i+1:4d}"]
            row += [f"{X[i, j]:.6g}" for j in range(n_var)]
            row += [f"{F[i, j]:.6g}" for j in range(n_obj)]
            row += [f"{int(constraint_violations[i])}"]
            f.write(" | ".join(row) + "\n")

class AdaptiveSBX(SBX):
    def __init__(self, eta_c_min=2, eta_c_max=20, prob=0.9, **kwargs):
        super().__init__(prob=prob, eta=eta_c_min, **kwargs)
        self.eta_c_min = eta_c_min
        self.eta_c_max = eta_c_max

    def _do(self, problem, X, **kwargs):
        # 获取当前代数信息
        algorithm = kwargs.get("algorithm")
        if algorithm is None:
            # 如果没有算法信息，使用默认SBX
            return super()._do(problem, X, **kwargs)
        
        # 获取当前代数和总代数
        gen = algorithm.n_gen
        total_gen = algorithm.termination.n_max_gen
        
        # # 自适应调整eta_c
        # progress = min(gen / total_gen, 1.0)
        # eta_c = self.eta_c_min + (self.eta_c_max - self.eta_c_min) * progress
        # 非线性自适应策略 - 更激进的收敛
        progress = min(gen / total_gen, 1.0)
        # 使用指数衰减，前期变化快，后期稳定
        eta_c = self.eta_c_max + (self.eta_c_min - self.eta_c_max) * (progress ** 0.5)
        self.eta = eta_c  # 更新eta值
        
        # 调用父类的SBX实现
        return super()._do(problem, X, **kwargs)

def eps_record():
    with open("../../data/eps_origion_more.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]  # 修复点

    # 提取表头后的数据行（跳过前3行：表头和分隔线）
    data_rows = lines[3:]  

    eps_list = []

    for row in data_rows:
        # 按 | 分割每行
        parts = [part.strip() for part in row.split('|')]
        
        # 提取eps（第6列）
        try:
            eps = parts[5].strip()  # 第六列是eps
            eps_value = None if eps == '-' else float(eps)
            eps_list.append(eps_value)
        except (IndexError, ValueError):
            continue  

    # 保存结果到文件
    with open('../../data/eps_array_more.txt', 'w') as f:
        temp = np.inf
        for i,val in enumerate(eps_list):
            if i == 0:
                continue
            temp = min(temp,val)
            f.write(f"{i}\t{temp}\n")

def get_paretodata(input_path,save_path):
    with open(input_path,"r") as file:
        lines = [line.strip() for line in file if line.strip()]  
    data_rows = lines
    res = []
    for row in data_rows:
        # 按 | 分割每行
        parts = [part.strip() for part in row.split('|')]
        try:
            stdv = parts[4].strip() 
            load = parts[5].strip() 
            res.append([float(stdv),float(load)])
        except (IndexError, ValueError):
            continue  

    with open(save_path, 'w') as f:
        for vec in res:  
            f.write(f"{vec[0]:.1f}\t{vec[1] / 1000:.2f}\n")

