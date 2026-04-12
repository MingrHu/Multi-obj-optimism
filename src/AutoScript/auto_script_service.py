import os
from auto_script_method import(Doe_sample_generate,Doe_execute)
from typing import List, Dict

# 全局任务管理字典 方便后续查询任务状态和结果
exec_task_manager: Dict[str, Doe_execute] = {}
smpgen_task_manager: Dict[str, Doe_sample_generate] = {}

def CreateSmpGenTask(task_id:str,
                     smp_save_path:str,
                     gen_method:str,
                     param_ranges:dict[str, tuple[float, float]],
                     n_samples:int = 0,
                     level_nums:List[int] = [])->Dict[str, str]:
    """
    创建并执行抽样任务
    返回结构规范示例：
    {
        "task_id": "xxx",
        "status": "success", # "success" | "failed"
        "message": "抽样完成，共生成 10 个样本"
    }
    """
    if n_samples == 0:
        return {}
    try:
        smpgen_task_manager[task_id] = smpgen_task_manager.get(task_id, 
        Doe_sample_generate(gen_method, param_ranges, smp_save_path, n_samples, level_nums))
        return {
            "task_id": task_id,
            "status": "success",
            "message": f"成功使用 {gen_method} 方法生成样本",
        }
    except Exception as e:
        print(f"抽样任务创建失败：{e}")
        return {
            "task_id": task_id,
            "status": "failed",
            "message": f"使用 {gen_method} 方法生成样本失败",
        }

# 使用示例参考 auto_script_test.py
def InitExecutionTask(
    task_id: str,
    paths_config: Dict[str, str], # 包含 res_db, res_key 等路径和文件名称
    par: List[List[str]],
    tar: List[List[str]],
    is_progress: List[bool],
    max_step: int
)->Dict[str, str]:
    """
    初始化执行任务
    返回结构规范示例：
    {
        "task_id": "xxx",
        "status": "success", # "success" | "failed"
        "message": "执行任务初始化成功"
    }
    """
    try:
        if paths_config["smp_file"] == "" or paths_config["std_key_file"] == "" or paths_config["temp_key_path"] == "" \
            or paths_config["res_db_path"] == "" or paths_config["res_key_path"] == "" or paths_config["res_txt_path"] == "":
            return {
                "task_id": task_id,
                "status": "failed",
                "message": "未指定样本、标准键、临时键文件、结果路径或结果文件名",
            }
        for f_path in paths_config.values():
            target_dir = f_path if not os.path.splitext(f_path)[1] else os.path.dirname(f_path)
            os.makedirs(target_dir, exist_ok=True)

        exc = Doe_execute(paths_config["smp_file"],
                          paths_config["std_key_file"],
                          paths_config["temp_key_path"],
                          paths_config["res_db_path"],
                          paths_config["res_key_path"],
                          paths_config["res_txt_path"],
                          par,tar,is_progress,max_step)
        exec_task_manager[task_id] = exec_task_manager.get(task_id, exc)
        exc.generate_key_file()

        return {
            "task_id": task_id,
            "status": "success",
            "message": "执行任务初始化成功",
        }
    except Exception as e:
        print(f"执行任务初始化失败：{e}")
        return {
            "task_id": task_id,
            "status": "failed",
            "message": f"执行任务初始化失败：{e}",
        }

def RunExecutionStep(task_id: str) -> Dict[str, str]:
    """
    - "generate_keys"
    - "run_deform"
    - "extract_data"
    """
    # 获取 TaskManager[task_id] 对应的实例
    if task_id not in exec_task_manager:
        return {
            "task_id": task_id,
            "status": "failed",
            "message": "执行任务不存在",
        }
    exc = exec_task_manager[task_id]
    exc.process_run()
    return {
        "task_id": task_id,
        "status": "success",
        "message": "计算任务开始运行",
    }
    # 返回执行结果：{"status": "success", "message": "KEY文件生成完毕"}
    
def QueryExecutionStatus(task_id: str) -> Dict[str, str]:
    """
    查询执行任务状态
    """
    if task_id not in exec_task_manager:
        return {
            "task_id": task_id,
            "status": "failed",
            "message": "执行任务不存在",
        }
    exc = exec_task_manager[task_id]
    return {
        "task_id": task_id,
        "status": f"{exc.pre_status}",
        "message": f"执行任务状态：{exc.pre_status}",
    }
    
def RunExtractData(task_id: str) -> Dict[str, str]:
    """
    提取数据
    """
    if task_id not in exec_task_manager:
        return {
            "task_id": task_id,
            "status": "failed",
            "message": "执行任务不存在",
        }
    exc = exec_task_manager[task_id]
    exc.extract()
    return {
        "task_id": task_id,
        "status": "success",
        "message": "开始提取数据",
    }
    