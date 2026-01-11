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
    'topdie':"2",
    'butdie':"3"
}
# 目标函数
TAR_FUNC = {
    'stress':_extractMaxStress,
    'load':_extractMaxLoad,
    'grain':_extractGrainStdv
}

class Doe_sample_generate:
    def __init__(self,sample_method:str,
                 param_ranges:dict[str, tuple[float, float]],
                 save_path:str,
                 n_samples:int = 0,
                 level_nums:List[int] = []) -> None:
        # 采样方法
        SAMPLE = {
            'lhs':LHSSampleGenerate,
            'full':FullSampleGenerate
        }
        if sample_method == 'lhs':
            LHSSampleGenerate(n_samples,param_ranges,save_path)
        elif sample_method == 'full':
            FullSampleGenerate(param_ranges,save_path,level_nums)
        else:
            raise ValueError(f"Unsupported sample method: {sample_method}")


#  @brief  求解类
#  @return None
#  @author Hu Mingrui
#  @date   2025/11/27
#  @param  sample_file      采样生成的txt文件 例如../../data/AUTO/smp.txt
#  @param  std_key_file     模板key文件 例如../../data/AUTO/key_modle.key
#  @param  temp_key_path    中间key文件路径 例如../../data/AUTO/temp_key  
#  @param  res_db_path      计算结果DB文件 例如../../data/AUTO/res_db
#  @param  res_key_path     最终结果key文件位置 例如../../data/AUTO/res_key
#  @param  res_txt_path     提取的数据集位置 例如../../data/AUTO/res_txt
#  @param  parmeter         工艺参数输入组合 固定 2 × n 第一行为n个工艺参数 第二行为n个工艺参数对应的部件序号
#  @param  target_val       目标函数的变量 固定 2 × m 第一行为m个目标变量 第二行为m个变量对应的部件序号
#  @param  is_inprogress    对应每个目标变量是否通过全过程计算提取（例如有的变量需要整个工艺流程来提取 有的只需要最后一步）
#  @param  max_step         输入设定的key文件求解过程的最大步数（用于确定和生成中间key文件）
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
        self.var = target_var   # [["grain","load"],["workpiece","topdie"]]
        self.in_progress = is_inprogress
        self.max_step = max_step
        # 一些中间路径
        self.tmp_key_file = []
        self.res_db_file = []

# PS:
# temp  temp    temp    speed
#   1     2      3       2
# 915.0	560.0   560.0	26.0
# 877.0	370.0   370.0	45.0
# 930.0	686.0   686.0	10.0
# 963.0	593.0   593.0	24.0
# 875.0	487.0   487.0	34.0
# 923.0	668.0   668.0	19.0
# 899.0	306.0   306.0	34.0
    def generate_key_file(self) -> None:
        with open(self.sp_path,'r',encoding = 'utf-8') as file:
            org_lines = file.readlines()
            for line in org_lines:
                self.par.append(line.split())
            
        with open(self.std_path,'r',encoding = 'utf-8') as key_file:
            std_key_file = key_file.readlines()
            for i in range(len(self.par)):
                # 先跳过前两行
                if i == 0 or i == 1:
                    continue
                new_key_file = []
                # 产生新key file
                for line in std_key_file:
                    line_list = line.split()
                    if len(line_list) >= 2:
                        for pos,data in enumerate(self.par[i]):
                            # 对于每一行满足条件的key_file 遍历匹配工艺参数和部件对象
                            # PS:注意 标准KEY文件的目标行格式必须满足每行第一个标识参数名称
                            # 第二个标识工艺参数所属的对象 最后一个标识工艺参数值
                            gy_par,tar_obj = self.par[0][pos],self.par[1][pos]
                            # 寻找匹配的工艺参数和对象 必须完全匹配 匹配后才能进行工艺参数修改
                            if KEYFILE_VAR_DIC[gy_par] == line_list[0] and line_list[1] == OBJ_DEF[tar_obj]:
                                # 生成这行的参数值
                                format_data = FormatFloat(data)
                                # 替换
                                line = line.replace(line_list[-1],format_data)
                                break
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

    def extract(self) -> None:
        res_lines = []
        for i,dbfile in enumerate(self.res_db_file):
            print(f"当前提取的文件为：{dbfile}")
            # 创建目录
            res_save_path = f"{self.res_key_path}\\{i}"
            os.makedirs(res_save_path,exist_ok = True)
            # 主要存放当前db生成的所有key文件
            list_key = []
            for step in range(0,self.max_step):
                key_file = GetNewFilePath(dbfile,res_save_path,str(step),"KEY")
                list_key.append(key_file)
                while not os.path.exists(key_file):
                    ProcessDB_TO_KEY(dbfile,key_file,str(step))
            res_line = self.par[i + 2]
            key_lines = []
            # 遍历获取每个key的所有内容 这里可以优化一下内存使用
            # TODO(MingrHu)
            for key_file in list_key:
                with open(key_file,'r',encoding = 'utf-8') as f:
                    key_lines.append(f.readlines())

            for idx in range(len(self.var[0])):
                # var = [["grain","load"],["workpiece","topdie"]]
                tar_info = self.var[0][idx]
                obj_info = self.var[1][idx]
                in_progress = self.in_progress[idx]
                # 拿到当前目标值标签和对象等信息后开始抽取值
                res_line.append(TAR_FUNC[tar_info](key_lines,obj_info,in_progress))
            # 收集最终结果
            res_lines.append(res_line)
        with open("output.txt", "w", encoding="utf-8") as f:
            for row_idx, row in enumerate(res_lines, start = 1):  
                row_str = "\t".join(map(str, row))  
                f.write(f"{row_idx}\t{row_str}\n")  
                
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


# TEST
if __name__ == "__main__":
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

