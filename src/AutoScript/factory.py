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
from utils import(KEYFILE_VAR_DIC,OBJ_DEF,
GetNewFilePath,FormatFloat,ProcessKEY_TO_DB,
ProcessRun_CALDB)


class Doe_sample_generate:
    def __init__(self,) -> None:
        


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
                 parmeter:List,     # 输入的工艺参数
                 target_var:List    # 优化的目标变量
                 ):
        self.sp_path = sample_file
        self.std_path = std_key_file
        self.tmp_key_path = temp_key_path
        self.res_db_path = res_db_path
        self.res_key_path = res_key_path
        self.res_txt_path = res_txt_path
        self.par = parmeter     # [["temp","speed"],["workpiece","topdie"]] 初始的格式
        self.var = target_var
        # 一些中间路径
        self.tmp_key_file = []
        self.res_db_file = []


    def generate_key_file(self) -> None:
        with open(self.sp_path,'r',encoding = 'utf-8') as file:
            org_lines = file.readlines()
            for line in org_lines:
                line_list = line.split()
                val = [float(val) for val in line_list]
                self.par.append(val)
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




                            
        





# TEST
# if __name__ == "__main__":