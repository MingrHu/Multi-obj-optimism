from utils import LHSSampleGenerate
import os
def sample_generate(samples_num:int,param_ranges:dict,sample_save_path:str):
    LHS_n_samples = samples_num  # 初始样本数量
    LHSSampleGenerate(LHS_n_samples, param_ranges,sample_save_path)  # 生成拉丁超立方采样数据



if __name__ =="__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 切换工作目录
    os.chdir(script_dir)
    # 参数配置
    param_ranges = {
        'workpiece_temp': (875, 965),    # 工件温度范围 (℃)
        'mold_temp': (300, 700),         # 模具温度范围 (℃)
        'forging_speed': (10, 50)        # 锻造速度范围 (mm/s)
    }
    sample_generate(100,param_ranges,"../../data")
    # FullSampleGenerate(param_ranges)  # 生成全因子采样数据