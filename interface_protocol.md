# 接口参数文档

> 本文件保留 Python 内部服务的历史协议和字段背景，不再作为 HTTP 调用依据
> 当前 Flask 路径、请求字段和响应格式以 [DOE_HTTP_API.md](DOE_HTTP_API.md) 为准
> 后端安装、启动和完整 Demo 测试见
> [src/mobo/api/BACKEND_STARTUP.md](src/mobo/api/BACKEND_STARTUP.md)
>
> 实现对齐说明（2026-07）：本协议已在仓库落地为三个服务层，均以任务 ID 为主键把
> 请求/状态/结果持久化到 `data/tasks/<id>/state.json`（见 `mobo.common.task_store`），
> 从而支持「仅凭 ID 续跑」：
>
> | 流程 | 服务模块 | 主键 | 主要函数 |
> |------|----------|------|----------|
> | 代理模型 | `mobo.surrogate.service` | `model_id`（`tr_` 前缀） | `train_surrogate` / `query_model_status` |
> | 多目标优化 | `mobo.optimization.service` | `task_id`（`opt_` 前缀） | `run_optimization` / `query_optimization_status` |
> | DEFORM 自动化 | `mobo.automation.service` | `task_id` | `init_execution_task` / `run_execution_step` / `run_extract_data` / `query_execution_status` |
>
> `state.json` 统一结构：`{task_id, kind, created_at, updated_at, status, stage, req, data}`，
> 其中 `req` 保存续跑所需的全部输入，`data` 保存阶段产物与结果（文件路径、指标等）。

## 1 代理模型部分

​ 本**微服务方内部调用方式和具体输入参数如下**，下面依次解释各部分输入参数

```python
# 代理模型调用示例
def TEST_CALL_SURR_MODEL():
    vars_f = ["1", "2", "3", "4", "5", "6", "7","8","9","10",
                "11","12","13","14","15","16","17","18","19","20", "res1", "res2", "res3"]
    data_file = '/Users/bytedance/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'

    selectItem = SurrogateModel(
        data_file=data_file,
        vars_f=vars_f,
        n_vars=7,
        model_params={
            "PRG": {"degree": 2},
            "SVR": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "RF": {"n_estimators": 300, "n_jobs": -1},
            "KM": {"alpha": 0.1, "n_restarts_optimizer": 10},
            "DNN": {"epochs": 300, "batch_size": 16, "verbose": 0, "patience": 30},
        },
    )
```

### 1.1 用户输入参数Req

​ 用户输入的参数主要包括工艺参数和目标变量的输入，**核心在于SurrogateModel类的使用**，为了方便端上能够直接调用，因此**本次需要端上传入的格式为json字符串，参考传入示例如下**

```json
{
	"model_index": 0, //模型索引映射：0-PRG、1-SVR、2-RF、3-KM、4-DNN，仅对应索引参数生效
        "biz_params": {	// 用户输入的参数和超参数设置
            	// 仅在PRG时为request
                "degree": 2, // PRG-多项式最高阶数 
            
            	// 仅在SVR时为request
                "C": 1.0, // SVR-惩罚系数	
                "epsilon": 0.1, // SVR-不敏感损失阈值 
            
            	// 仅在RF时为request
                "n_estimators": 300, // RF-决策树数量 仅在RF时为request
                "n_jobs": -1, // RF-并行核心数(-1=全核心)
            
            	// 仅在KM时为request
                "alpha": 0.1, // KM-正则化系数
                "n_restarts_optimizer": 10, // KM-优化器重启次数
            
            	// 仅在DNN时为request
                "epochs": 300, // DNN-训练迭代轮数
                "batch_size": 16, // DNN-训练批次大小
                "verbose": 0, // DNN-日志打印等级(0=静默)
                "patience": 30 // DNN-早停等待轮数
        }
}
```

​ **传参调用示例req**

```json
// 多项式回归
{
	"model_index": 0,
	"params": {
		"degree": 2
	}
}

// 支持向量回归
{
	"model_index": 1,
	"params": {
		"kernel": "rbf",
		"C": 1.0,
		"epsilon": 0.1
	}
}

// 随机森林
{
	"model_index": 2,
	"params": {
		"n_estimators": 300,
		"n_jobs": -1
	}
}

// 克里金法
{
	"model_index": 3,
	"params": {
		"alpha": 0.1,
		"n_restarts_optimizer": 10
	}
}

// 神经网络
{
	"model_index": 4,
	"params": {
		"epochs": 300,
		"batch_size": 16,
		"verbose": 0,
		"patience": 30
	}
}
```

### 1.2 返回给端上的Resp

​ 本协议是需要返回时端上所需要的一些值，包含对训练代理模型返回响应值

```json
// 代理模型返回响应值resp
{
    "code": 0, // 状态码 0成功 非0失败
    "msg": "训练完成", // 结果描述文本
    "model_id": "tr_20260619_0823_1256", // 代理模型唯实例一id 落盘 + 复用唯一
    "data": {
        "model_index": 0, // 当前训练模型索引 0-PRG/1-SVR/2-RF/3-KM/4-DNN
        "train_status": "finished", // 训练状态：running/finished/failed
        "model_save_path": "/model/proxy/prg_xxxx.pkl", // 模型文件存储路径
        "train_cost_sec": 12.68, // 训练耗时(秒)
        "hyper_params": { // 本次训练使用的完整超参（入参+缺失补默认）
            "degree": 2
        },
        "metrics": { // 模型评估指标（回归通用）
            "r2": 0.962,
            "rmse": 0.036,
            "mae": 0.021,
            "mse": 0.0013
        },
        "train_info": {
            "train_sample_num": 1200, // 训练集样本量
            "test_sample_num": 300, // 测试集样本量
            "early_stop_epoch": 246 // DNN专属，其余模型返回null
        }
    }
}

```

## 2 优化算法部分

​ 本**微服务方内部调用方式逻辑和具体输入参数如下**，目前项目所使用的优化算法默认为NSGA2

```python
def TEST_CALL_OPT():
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
```

### 2.1 用户输入参数Req

​ 本次要求端上在优化算法传递的参数字段如下，

```json
{
    "model_id": "tr_20260619_0823_1256", // 前面落盘的代理模型标识 用于优化算法
    "objective_names": ["grain", "load"], // 优化目标输出变量名
    "input_var_count": 3, // 输入决策变量数量 n_vars
    "all_var_list": ["1", "2", "3", "grain", "load"], // vars_out 全部变量列表
    "decision_var_indices": [0, 1, 2], // 参与优化的输入变量下标
    "decision_var_names": ["1", "2", "3"], // 决策变量名称，用于结果导出
    "decision_bounds": [ // 各决策变量上下限，和indices一一对应
        {"lower": 875, "upper": 965, "desc": "工件温度[°C]"},
        {"lower": 300, "upper": 700, "desc": "模具温度[°C]"},
        {"lower": 10, "upper": 50, "desc": "上模速度[mm/s]"}
    ],
    "constraints": [ // 约束条件列表 ConstraintSpec映射
        {
            "target_obj": "grain", // 约束绑定的目标变量
            "constraint_kind": "upper", // upper/lower 上下限约束
            "limit_value": 30 // 约束阈值
        },
        {
            "target_obj": "load",
            "constraint_kind": "upper",
            "limit_value": 330000
        }
    ],
    "objective_config": [ // 每个目标的配置 ObjectiveSpec映射
        {
            "name": "grain",
            "minimize": true // true最小化/false最大化
        },
        {
            "name": "load",
            "minimize": true
        }
    ],
    "optimizer_config": { // NSGA2算法完整超参
        "pop_size": 100, // 种群规模
        "n_offsprings": 100, // 每代子代数量
        "eliminate_duplicates": true, // 是否去重重复个体
        "n_gen": 200, // 迭代总代数
        "seed": 42, // 随机种子
    },
    "output_config": { // 输出文件路径配置
        "pareto_txt_path": "../../data/pareto_solutions.txt"
    }
}
```

### 2.2 返回给端上的Resp

​ 本协议是需要返回时端上所需要的一些值，**多目标优化写入文件的结果字段可能较多，因此先存在文件里面，后续再具体协商文件内部的结果字段**

```json
{
    "code": 0,
    "msg": "多目标优化计算完成",
    "task_id": "opt_20260619_1522_0041",	// 用于追溯多目标优化任务
    "data": {
        "task_info": {
            "model_id": "tr_20260619_0823_1256",
            "decision_var_names": ["1", "2", "3"],
            "objective_names": ["grain", "load"],
            "total_generation": 200,
            "pop_size": 100,
            "run_time_sec": 28.62
        },
        "file_resource": {
            "solution_txt_path": "../../data/pareto_solutions.txt" // 多目标优化的输出位置
        },
        "constraint_check": {
            "grain_max_limit": 30,
            "load_max_limit": 330000,
            "all_solution_feasible": true
        }
    }
}
```

## 3 TODO LIST

​ 后续接口协商都以这个文件更新为准。

已落地：
- 三个流程的服务层与 `data/tasks/<id>/state.json` 持久化（仅凭 ID 续跑）。
- 代理模型：`model_index`（0-PRG/1-SVR/2-RF/3-KM/4-DNN）到底层 `which_model` 的映射，
  `biz_params` 按各模型约定顺序转换；训练结果（模型路径/超参/耗时）落盘。

已补充：
- 稳定对外入口为 `mobo.api` Flask 服务，完整协议与调用示例见 `DOE_HTTP_API.md`。
- 优化请求中的 `optimizer_config` / `decision_bounds` / `constraints` / `objective_config`
  已透传到新增参数化 NSGA-II 编排器；历史无参 `NSGA2_run` 示例保持不变。
- 代理模型请求实际还必须提供 `data_file`、`all_var_list`、`input_var_count`。每个
  `model_id` 的产物快照到自身任务目录，避免共享模型族目录覆盖实例。

仍有限制：
- 历史代理训练函数尚未消费 `model_par`，所以 `mobo.api` 只接受与当前代码一致的固定
  超参数，避免静默忽略自定义值。文档原示例中的 KM/DNN 参数与代码真实固定值不一致，
  HTTP 调用应以 `DOE_HTTP_API.md` 为准。
- 代理模型 resp 的 `metrics`（r2/rmse/mae/mse）需底层训练函数返回后才能落盘，
  当前 `data` 暂不含 metrics。
