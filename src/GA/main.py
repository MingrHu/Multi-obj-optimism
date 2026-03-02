import os
from joblib import load as joblib_load
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from GAproblem import ConstraintSpec, ObjectiveSpec, SurrogateOptimizationProblem
from utils import AdaptiveSBX, save_pareto_solutions

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def _load_model(model_dir: str, target_name: str):
    pkl_path = os.path.join(model_dir, f"{target_name}_model.pkl")
    keras_path = os.path.join(model_dir, f"{target_name}_model.keras")

    if os.path.exists(pkl_path):
        return joblib_load(pkl_path)

    if os.path.exists(keras_path):
        from keras.models import load_model as keras_load_model

        return keras_load_model(keras_path)

    raise FileNotFoundError(f"找不到模型文件：{pkl_path} 或 {keras_path}")


#  @brief  遗传算法运行示例
#  @return None
#  @author Hu Mingrui
#  @date   2026/02/06
#  @about  选择部分输入输出变量进行优化 例如选择1-3为输入变量，res1-res2为输出变量
def NSGA2_run():
    model_family = "PRG"
    model_dir = f"../../data/models/{model_family}"

    # 1-3为输入变量 grain和load是输出变量
    vars_out = ["1", "2", "3", "grain", "load"]
    n_vars = 3

    # 目标函数对象为res1 res2
    objective_names = ["grain", "load"]

    # 加载标准化器
    scalers = joblib_load(os.path.join(model_dir, f"{objective_names[0]}_scalers.pkl"))

    output_names = vars_out[n_vars:]
    objective_specs = []
    for name in objective_names:
        y_index = output_names.index(name)
        model = _load_model(model_dir, name)
        objective_specs.append(ObjectiveSpec(name=name, model=model, y_index=y_index, minimize=True))

    # 选择的输入变量为1-3
    decision_var_indices = [0, 1, 2]
    # 输入变量的取值范围
    decision_bounds = [
        (875, 965),   # 工件温度范围 [°C]
        (300, 700),   # 模具温度范围 [°C]
        (10, 50)      # 上模速度范围 [mm/s]
    ]

    # 约束条件
    constraints = [
        ConstraintSpec(objective="grain", kind="upper", value=30),
        ConstraintSpec(objective="load", kind="upper", value=330000),
    ]

    problem = SurrogateOptimizationProblem(
        objectives=objective_specs,
        scalers=scalers,
        decision_var_indices=decision_var_indices,
        bounds=decision_bounds,
        x_base=None,
        fixed_values=None,
        constraints=constraints,
    )

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=FloatRandomSampling(),
        crossover=AdaptiveSBX(eta_c_min=20, eta_c_max=5, prob=0.95),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 200),
        seed=42,
        verbose=True,
        save_history=True,
    )

    plot = Scatter()
    plot.add(res.F, color="red")
    plot.save("../../data/pareto_front.png")

    input_names = vars_out[:n_vars]
    var_names = [input_names[i] for i in decision_var_indices]
    save_pareto_solutions(
        res,
        filename="../../data/pareto_solutions.txt",
        var_names=var_names,
        obj_names=objective_names,
    )


if __name__ == "__main__":
    NSGA2_run()