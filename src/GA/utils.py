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


class HybridPMG(Mutation):
    def __init__(self, eta_m_min=10, eta_m_max=50, p_poly=0.7, sigma_max=0.1, sigma_min=0.01, lambda_=3):
        super().__init__()
        self.eta_m_min = eta_m_min
        self.eta_m_max = eta_m_max
        self.p_poly = p_poly
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.lambda_ = lambda_

    def _do(self, problem, X, t, T, **kwargs):
        n_var, n_offsprings = problem.n_var, X.shape[0]
        eta_m = self.eta_m_min + (self.eta_m_max - self.eta_m_min) * (t / T)
        sigma = self.sigma_max * np.exp(-self.lambda_ * (t / T)) + self.sigma_min
        
        for i in range(n_offsprings):
            for j in range(n_var):
                if np.random.rand() < (1 / n_var):  # Mutation probability
                    if np.random.rand() < self.p_poly:  # Polynomial mutation
                        u = np.random.rand()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (1 / (2 * (1 - u))) ** (1 / (eta_m + 1))
                        X[i, j] += delta * (problem.xu[j] - problem.xl[j])
                    else:  # Gaussian mutation
                        X[i, j] += sigma * np.random.randn() * (problem.xu[j] - problem.xl[j])
                    
                    # Bound check
                    X[i, j] = np.clip(X[i, j], problem.xl[j], problem.xu[j])
        return X
