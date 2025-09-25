import numpy as np
import os
from joblib import load
from utils import save_pareto_solutions,AdaptiveSBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from GAproblem import FormingProcessOptimization  # 确保此类已正确定义

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def NSGA2_run():

    # 加载模型和标准化器
    stdv_model = load('../../data/models/SVR/svr_res_stdv_model.pkl')
    load_model = load('../../data/models/SVR/svr_res_load_model.pkl')
    scalers = load('../../data/models/SVR/svr_res_scalers.pkl') 


    # 定义优化问题
    problem = FormingProcessOptimization(
        stdv_model=stdv_model,
        load_model=load_model,
        scalers=scalers  # 传入标准化器
    )

    # 配置 NSGA-II
    algorithm = NSGA2(
        pop_size=50, # 初始种群大小
        n_offsprings=50, # 每代的子代数
        # sampling=LHS(),
        sampling = FloatRandomSampling(),
        # crossover = SBX(prob=0.9, eta=15), # 二进制交叉变异
        crossover=AdaptiveSBX(eta_c_min=20, eta_c_max=5, prob=0.95),
        mutation=PM(eta=20), # 多项式变异
        eliminate_duplicates=True # 是否去重
    )

    # 运行优化
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=42, # LHS无需启用
        verbose=True,
        save_history = True
    )


    plot = Scatter()
    plot.add(res.F, color="red")
    plot.save("../../data/pareto_front.png")

    save_pareto_solutions(res)

#*************************TEST*************************************
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

if __name__ == "__main__":
    # NSGA2_run()
    eps_record()