import sys,time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from auto_script_service import (CreateSmpGenTask,InitExecutionTask,RunExtractData,
                                 RunExecutionStep,QueryExecutionStatus)
from Common.tools import (PROJECT_DIR)

# if windows
# par = [["temp","temp","temp","speed"],["workpiece","topdie","butdie","topdie"]] 
# tar = [["grain","load"],["workpiece","topdie"]]
# is_progress = [False,True]
# # 请输入绝对路径 相对路径在deform里面处理会有问题
# sample_file = "../../data/AUTO/smp.txt"
# std_key_file = "../../data/AUTO/MODEL.KEY"
# temp_key_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\temp_key"   # "../../data/AUTO/temp_key"  case 1    
# res_db_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\res_db"       # "../../data/AUTO/res_db"   case 2
# res_key_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\res_key"     # case 3
# res_txt = "../../data/AUTO/res_txt"

def sample_generate_test():
    # 定义参数范围
    param_ranges = {
        'temp1': (875.0, 965.0),    # 工件温度范围 (℃)
        'temp2': (300.0, 700.0),    # 上模具温度范围 (℃)
        'temp3':(300.0,700.0),      # 下模具温度范围 (℃)
        'speed':(10.0, 50.0) ,      # 锻造速度范围 (mm/s)
    }
    save_path = f"{PROJECT_DIR}/data/TEST"
    
    CreateSmpGenTask("1001",save_path,"lhs",param_ranges,1000,)

def generate_keyfile_test():
    par = [["temp","temp","temp","speed"],
           ["workpiece","topdie","butdie","topdie"]] 
    tar = [["grain","load"],
           ["workpiece","topdie"]]
    is_progress = [False,True]
    paths_config = {
        "smp_file": f"{PROJECT_DIR}/data/AUTO/smp.txt",
        "std_key_file": f"{PROJECT_DIR}/data/AUTO/MODEL.KEY",
        "temp_key_path": f"{PROJECT_DIR}/data/AUTO/temp_key",
        "res_db_path": f"{PROJECT_DIR}/data/AUTO/res_db",
        "res_key_path": f"{PROJECT_DIR}/data/AUTO/res_key",
        "res_txt_path": f"{PROJECT_DIR}/data/AUTO/res_txt",
    }
    
    msg = InitExecutionTask("2026-04-10-1zsacazwwzs/12s",paths_config,par,tar,is_progress,100)
    print(msg)
    while True:
        status = QueryExecutionStatus("2026-04-10-1zsacazwwzs/12s")
        time.sleep(3)
        if status["status"] == "0":
            break


def run_process_test():
    msg = RunExecutionStep("2026-04-10-1zsacazwwzs/12s")
    print(msg)
    while True:
        status = QueryExecutionStatus("2026-04-10-1zsacazwwzs/12s")
        time.sleep(3)
        if status["status"] == "0":
            break

def extra_data_test():
    msg = RunExtractData("2026-04-10-1zsacazwwzs/12s")
    print(msg)
    while True:
        status = QueryExecutionStatus("2026-04-10-1zsacazwwzs/12s")
        time.sleep(3)
        if status["status"] == "0":
            break
    
if __name__ == "__main__":
    # sample_generate_test()
    generate_keyfile_test()
    run_process_test()
    extra_data_test()
