import os,time,threading
from typing import List
from utils import(GetNewFilePath,ProcessKEY_TO_DB_BATCH,ProcessRUN_CALDB,LHSSampleGenerate,
                   FullSampleGenerate,ProcessGEN_KEY_FILE,ProcessEXTRACT_DB_FILE)
from Common.tools import logger

# 任务状态定义
Task_Status_done = 0
Task_Status_running = 1
Task_Status_failed = -1

#  @brief  采样类
#  @return None
#  @author Hu Mingrui
#  @date   2025/11/24
#  @param  sample_method    采样方法 lhs/full
#  @param  param_ranges     参数范围字典 例如{'temp1': (875.0, 965.0), 'temp2': (300.0, 700.0), 'temp3':(300.0,700.0), 'speed':(10.0, 50.0)}
#  @param  save_path        采样生成的txt文件保存路径 例如../../data/sample
#  @param  n_samples        采样的总数量 例如10
#  @param  level_nums       采样等级数量列表如果采用full采样方法 必须输入 例如[5,2,2,2] 例如5个等级 每个等级2个参数
class Doe_sample_generate:
    def __init__(self,sample_method:str,
                 param_ranges:dict[str, tuple[float, float]],
                 save_path:str,
                 n_samples:int = 0,
                 level_nums:List[int] = []) -> None:
        # # 采样方法
        # SAMPLE = {
        #     'lhs':LHSSampleGenerate,
        #     'full':FullSampleGenerate
        # }
        if sample_method == 'lhs':
            LHSSampleGenerate(n_samples,param_ranges,save_path)
        elif sample_method == 'full':
            if  level_nums == []:
                logger.error("level_nums must be provided for full sampling")
                return
            FullSampleGenerate(param_ranges,save_path,level_nums)
        else:
            logger.error(f"Unsupported sample method: {sample_method}")


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
#  @param  target_val       目标值 固定 2 × m 第一行为m个目标变量 第二行为m个变量实际对应的部件序号 比如workpiece对应1 topdie对应2 butdie对应3
#  @param  is_inprogress    对应每个目标值是否通过全过程计算提取（例如有的变量需要整个工艺流程来提取 有的只需要最后一步）
#  @param  max_step         输入设定的key文件求解过程的最大步数（一定要准确 用于确定和生成中间key文件）
#  @about  根据指定的方法完成数据驱动操作
#  @brief 输入的parmeter示例
#  temp      temp    temp    speed
#  workpiece topdie  butdie  topdie
#  @brief 输入的target_var示例
#  grain     load    stress
#  workpiece topdie  butdie
#  @brief 输入的样本文件内容示例:
#  915.0    560.0    560.0    26.0
#  877.0    370.0    370.0    45.0
#  930.0    686.0    686.0    10.0
#  963.0    593.0    593.0    24.0
#  875.0    487.0    487.0    34.0
#  923.0    668.0    668.0    19.0
#  899.0    306.0    306.0    34.0
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
                 max_step:int,
                 is_test:bool = True
                ):
        self.smp_path = sample_file
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
        self.tmp_key_file:list[str] = []
        self.res_db_file:list[str] = []

        # TODO: 任务状态管理
        # 每个方法都有一个前一个任务状态值
        # 状态只有三个值 -1 0 1 分别代表失败 完成 进行中
        self.pre_status:int = 0
        self.is_test = is_test

    def generate_key_file(self) -> None:
        if self.pre_status != Task_Status_done:
            logger.error("can not generate key file because pre_status not done")
            return
        
        # 将样本文件生成的数值填入par
        with open(self.smp_path,'r',encoding = 'utf-8') as file:
            org_lines = file.readlines()
            for line in org_lines:
                self.par.append(line.split())
        # 异步生成key文件
        def async_task():
            try:
                logger.info("Async generate key file start")
                self.pre_status = Task_Status_running  # 状态运行中
                if self.is_test:
                    time.sleep(10)
                    self.tmp_key_file = ["MingrHu"]
                    self.pre_status = Task_Status_done
                    logger.info("✅ generate key file done")
                    return
                
                self.tmp_key_file = ProcessGEN_KEY_FILE(self.std_path,self.par,self.tmp_key_path)
                self.pre_status = Task_Status_done
                logger.info("✅ generate key file done")
            
            except Exception as e:
                self.pre_status = Task_Status_failed
                logger.error(f"❌ generate key file failed: {str(e)}")
        
        # 启动任务
        thread = threading.Thread(target=async_task, daemon=True)
        thread.start() 
    
    def process_run(self) -> None:
        if self.pre_status != Task_Status_done:
            logger.error("can not process run because pre_status not done")
            return
        
        tmp_key_list:list[str] = []
        save_file_list:list[str] = []
        for i,tmp_key in enumerate(self.tmp_key_file):
            os.makedirs(f"{self.res_db_path}\\{i}",exist_ok = True)
            save_file = GetNewFilePath(tmp_key,f"{self.res_db_path}\\{i}","","DB")
            self.res_db_file.append(save_file)
            if os.path.exists(save_file):
                continue
            tmp_key_list.append(tmp_key)
            save_file_list.append(save_file)

        def async_task():
            try:
                self.pre_status = Task_Status_running
                logger.info("Async process run start")
                # 批量转KEY为DB + 计算DB文件
                if self.is_test:
                    time.sleep(10)
                    self.pre_status = Task_Status_done
                    logger.info("✅ process run done")
                    return

                ProcessKEY_TO_DB_BATCH(tmp_key_list,save_file_list)
                ProcessRUN_CALDB(self.res_db_file,24)
                self.pre_status = Task_Status_done
                logger.info("✅ process run done")
            except Exception as e:
                self.pre_status = Task_Status_failed
                logger.error(f"❌ process run failed: {str(e)}")
        
        # 启动任务
        thread = threading.Thread(target=async_task, daemon=True)
        thread.start() 


    def extract(self) -> None:
        if self.pre_status != Task_Status_done:
            logger.error("can not extract because pre_status not done")
            return
        def async_task():
            try:
                logger.info("Async extract start")
                self.pre_status = Task_Status_running
                if self.is_test:
                    time.sleep(10)
                    self.pre_status = Task_Status_done
                    logger.info("✅ extract done")
                    return

                ProcessEXTRACT_DB_FILE(self.res_db_file,self.res_key_path,self.max_step,
                                       self.par,self.var,self.in_progress,self.res_txt_path)
                self.pre_status = Task_Status_done
                logger.info("✅ extract done")
            except Exception as e:
                self.pre_status = Task_Status_failed
                logger.error(f"❌ extract failed: {str(e)}")
        
        # 启动任务
        thread = threading.Thread(target=async_task, daemon=True)
        thread.start() 