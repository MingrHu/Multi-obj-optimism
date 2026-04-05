from .auto_script_method import (Doe_sample_generate,Doe_execute)

def sample_generate_test():
    # 定义参数范围
    param_ranges = {
        'temp1': (875.0, 965.0),    # 工件温度范围 (℃)
        'temp2': (300.0, 700.0),    # 上模具温度范围 (℃)
        'temp3':(300.0,700.0),      # 下模具温度范围 (℃)
        'speed':(10.0, 50.0) ,      # 锻造速度范围 (mm/s)
    }

    # 生成样本
    lhs = Doe_sample_generate(
        sample_method = "lhs",
        param_ranges = param_ranges,
        save_path="../../data/sample",
        n_samples=10,
    )

    full = Doe_sample_generate(
        sample_method = 'full',
        param_ranges = param_ranges,
        save_path = "../../data/sample",
        n_samples = 0,
        level_nums = [5,2,2,2]
    )

def generate_keyfile_test():
    par = [["temp","temp","temp","speed"],["workpiece","topdie","butdie","topdie"]] 
    tar = [["grain","load"],["workpiece","topdie"]]
    is_progress = []
    exc = Doe_execute("../../data/AUTO/smp.txt",
                      "../../data/AUTO/MODEL.KEY",
                      "../../data/AUTO/temp_key",
                      "../../data/AUTO/res_db",
                      "../../data/AUTO/res_key",
                      "../../data/AUTO/res_txt",
                      par,tar,is_progress,880)
    exc.generate_key_file()

def run_factory_test():
    par = [["temp","temp","temp","speed"],["workpiece","topdie","butdie","topdie"]] 
    tar = [["grain","load"],["workpiece","topdie"]]
    is_progress = [False,True]
    # 请输入绝对路径 相对路径在deform里面处理会有问题
    sample_file = "../../data/AUTO/smp.txt"
    std_key_file = "../../data/AUTO/MODEL.KEY"
    temp_key_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\temp_key"   # "../../data/AUTO/temp_key"  case 1    
    res_db_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\res_db"       # "../../data/AUTO/res_db"   case 2
    res_key_path = "C:\\Users\\16969\\Desktop\\Multi-obj-optimism\\data\\AUTO\\res_key"     # case 3
    res_txt = "../../data/AUTO/res_txt"

    exc = Doe_execute(sample_file,
                      std_key_file,
                      temp_key_path,
                      res_db_path,
                      res_key_path,
                      res_txt,
                      par,tar,is_progress,880)
    exc.generate_key_file()
    exc.process_run()        
    exc.extract()

    input("Press Enter to exit...")  # 添加这一行
    