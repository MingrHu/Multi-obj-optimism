import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.sbx import SBX

def save_pareto_solutions(res, filename="../../data/pareto_solutions.txt"):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("Pareto前沿最优解集\n")
        f.write("=" * 80 + "\n")
        f.write(f"解数量: {len(res.X)}\n")
        f.write(f"变量数: {res.X.shape[1]}, 目标数: {res.F.shape[1]}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入表头
        f.write("序号 | 工件温度(°C) | 模具温度(°C) | 速度(mm/s) | 标准差 | 载荷 | 约束违反\n")
        f.write("-" * 100 + "\n")
        
        # 计算约束违反程度
        constraint_violations = np.sum(res.G > 0, axis=1)
        
        # 写入每个解
        for i in range(len(res.X)):
            f.write(f"{i+1:4d} | {res.X[i,0]:11.1f} | {res.X[i,1]:11.1f} | {res.X[i,2]:9.1f} | "
                   f"{res.F[i,0]:7.2f} | {res.F[i,1]:7.2f} | {constraint_violations[i]:8d}\n")

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

