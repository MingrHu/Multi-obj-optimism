# ********************************************
#  创建时间： 2025/11/24
#  名   称： 自动化DEFORM求解模块定义
#  版   本： 1.0
#  @author    Hu Mingrui
#  说   明： 本功能负责批量产生KEY文件
#  将KEY文件保存为DB文件 运行模拟求解器
#  求解完成后批量提取对应的数据到指定位置
#  ！注意！输入的所有路径请符合windows规范为了
#  避免转义字符的影响 统一采用双斜线表示路径分隔 "\\"
# *******************************************
import os
from typing import List
from utils import(_extractMaxStress,_extractMaxLoad,_extractGrainStdv,
GetNewFilePath,FormatFloat,ProcessKEY_TO_DB,ProcessDB_TO_KEY,
ProcessRun_CALDB,LHSSampleGenerate,FullSampleGenerate)
# *********************SOME VAR DEF***********************
# KEY文件关键字变量
KEYFILE_VAR_DIC = {
    'temp':"NDTMP",
    'speed':"MOVCTL",

}
# 指定对象
OBJ_DEF = {
    'workpiece':"1",
    'topdie':"1",
    'butdie':"1"
}
# 
TAR_FUNC = {
    'stress':_extractMaxStress,
    'load':_extractMaxLoad,
    'grain':_extractGrainStdv
}

class Doe_sample_generate:
    def __init__(self,sample_method:str,param_ranges:dict[str, tuple[float, float]],n_samples:int,save_path:str) -> None:
        # 采样方法
        SAMPLE = {
            'lhs':LHSSampleGenerate,
            'full':FullSampleGenerate
        }
        if sample_method in SAMPLE:
            SAMPLE[sample_method](n_samples,param_ranges,save_path)
        else:
            raise ValueError(f"Unsupported sample method: {sample_method}")


#  @brief  求解类
#  @return None
#  @author Hu Mingrui
#  @date   2025/11/27
#  @about  根据指定的方法完成数据驱动操作
class Doe_execute:    

    def __init__(self, 
                 sample_file: str,  # 样本文件
                 std_key_file:str,  # 模板key文件
                 temp_key_path:str, # 批量生成的输入key路径
                 res_db_path:str,   # 最终的结果db路径
                 res_key_path:str,  # 最终的结果key路径
                 res_txt_path:str,  # 最终的数据集位置
                 parmeter:List[List[str]],     # 工艺参数固定项 2 × n 
                 target_var:List[List[str]],   # 目标变量固定项 2 × m
                 is_inprogress:List[bool],     # 目标值是否进行全过程提取 1 × m
                 max_step:int
                 ):
        self.sp_path = sample_file
        self.std_path = std_key_file
        self.tmp_key_path = temp_key_path
        self.res_db_path = res_db_path
        self.res_key_path = res_key_path
        self.res_txt_path = res_txt_path
        self.par = parmeter     # [["temp","speed"],["workpiece","topdie"]] 初始的格式
        self.var = target_var   # [["temp","load"],["workpiece","topdie"]]
        self.in_progress = is_inprogress
        self.max_step = max_step
        # 一些中间路径
        self.tmp_key_file = []
        self.res_db_file = []


    def generate_key_file(self) -> None:
        with open(self.sp_path,'r',encoding = 'utf-8') as file:
            org_lines = file.readlines()
            for line in org_lines:
                self.par.append(line.split())
            
        with open(self.std_path,'r',encoding = 'utf-8') as key_file:
            std_key_file = key_file.readlines()
            for i in range(len(self.par[0])):
                if i == 0 or i == 1:
                    continue
                new_key_file = []
                for line in std_key_file:
                    line_list = line.split()
                    if len(line) >= 2:
                        for pos,data in enumerate(self.par[i]):
                            par = self.par[0][pos],tar_obj = self.par[1][pos]
                            if KEYFILE_VAR_DIC[par] == line_list[0] and line_list[1] == OBJ_DEF[tar_obj]:
                                format_data = FormatFloat(data)
                                line = line.replace(line_list[-1],format_data)
                    new_key_file.append(line)

                save_file = GetNewFilePath(self.std_path,self.tmp_key_path,str(i - 2),"KEY")
                with open(save_file, 'w',encoding = 'utf-8') as f:
                    f.writelines(new_key_file)
                self.tmp_key_file.append(save_file)
                print(f"第{i - 1}个新生成的KEY文件已经保存!")
    
    def process_run(self) -> None:
        for i,tmp_key in enumerate(self.tmp_key_file):
            os.makedirs(f"{self.res_db_path}\\{i}",exist_ok = True)
            save_file = GetNewFilePath(tmp_key,f"{self.res_db_path}\\{i}","","DB")
            self.res_db_file.append(save_file)
            if os.path.exists(save_file):
                continue
            ProcessKEY_TO_DB(tmp_key,save_file)
        ProcessRun_CALDB(self.res_db_file)

    def extract(self,max_step:int) -> None:
        res_lines = []
        for i,dbfile in enumerate(self.res_db_file):
            print(f"当前提取的文件为：{dbfile}")
            # 创建目录
            res_save_path = f"{self.res_key_path}{i}"
            os.makedirs(self.res_key_path,exist_ok = True)
            list_key = []
            for step in range(0,self.max_step):
                key_file = GetNewFilePath(dbfile,res_save_path,str(step),"KEY")
                list_key.append(key_file)
                while not os.path.exists(key_file):
                    ProcessDB_TO_KEY(dbfile,key_file,str(step))
            res_line = self.par[i + 2]
            key_lines = []
            # 以目标值顺序进行搜索
            # res_line = [temp:398,speed:20,tar1:10.2,tar2:0.9]
            for key_file in list_key:
                with open(key_file,'r',encoding = 'utf-8') as f:
                    key_lines.append(f.readlines())

            for idx in range(len(self.var[0])):
                tar_info = self.var[0][idx]
                obj_info = self.var[1][idx]
                in_progress = self.in_progress[idx]
                res_line.append(TAR_FUNC[tar_info](key_lines,obj_info,in_progress))
            # 收集最终结果
            res_lines.append(res_line)
        with open("output.txt", "w", encoding="utf-8") as f:
            for row_idx, row in enumerate(res_lines, start = 1):  
                row_str = "\t".join(map(str, row))  
                f.write(f"{row_idx}\t{row_str}\n")  
                


# TEST
if __name__ == "__main__":
    # 定义参数范围
    param_ranges = {
        'temp1': (875.0, 965.0),    # 工件温度范围 (℃)
        'temp2': (300.0, 700.0),    # 模具温度范围 (℃)
        'temp3': (300.0, 700.0),    # 模具温度范围 (℃)
        'temp4': (300.0, 700.0),    # 模具温度范围 (℃)
        'temp5': (300.0, 700.0),    # 模具温度范围 (℃)
        'speed':(10.0, 50.0) ,     # 锻造速度范围 (mm/s)
    }

    # 生成样本
    sampler = Doe_sample_generate(
        sample_method = "lhs",
        param_ranges = param_ranges,
        n_samples=10000,
        save_path="./data/sample"
    )