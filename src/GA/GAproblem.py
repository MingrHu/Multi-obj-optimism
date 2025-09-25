from pymoo.core.problem import ElementwiseProblem
import numpy as np

class FormingProcessOptimization(ElementwiseProblem):
    def __init__(self, stdv_model, load_model, scalers):
        self.stdv_model = stdv_model
        self.load_model = load_model
        self.scalers = scalers

        # 定义变量范围
        param_bounds = [
            (875, 965),   # 工件温度范围 [°C]
            (300, 700),   # 模具温度范围 [°C]
            (10, 50)       # 上模速度范围 [mm/s]
        ]
        n_var = len(param_bounds)
        xl = np.array([b[0] for b in param_bounds])
        xu = np.array([b[1] for b in param_bounds])
        # n_obj 目标函数个数
        super().__init__(n_var=n_var, n_obj=2, n_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 标准化输入（注意：x是1D数组，需reshape为(1, n_var)）
        x_scaled = self.scalers['scaler_X'].transform(np.array(x).reshape(1, -1))

        # 分别用两个模型预测
        stdv_scaled = self.stdv_model.predict(x_scaled)[0]  # 输出是2D数组，取第一个值
        load_scaled = self.load_model.predict(x_scaled)[0]

        stdv = self.scalers['scaler_y_stdv'].inverse_transform([[stdv_scaled]])[0][0]
        load = self.scalers['scaler_y_load'].inverse_transform([[load_scaled]])[0][0]

        # 设置目标函数
        out["F"] = [stdv , load] # 最小化stdv和load

        # 设置约束条件
        max_load = 350000
        min_stdv = 5
        out["G"] = [
            load - max_load,  # load ≤ 350000 → G ≤ 0
            min_stdv - stdv   # stdv ≥ 5 → G ≤ 0
        ]

