from utils import LHSSampleGenerate,merge_files,read_file,write_output
import os
from datetime import datetime
from executor import (InputKEYParameter,GetNewFilePath,ProcessKEY_TO_DB,ProcessRun_CALDB,ProcessDB_TO_KEY,
                      DB,KEY)
from utils import ExtractValueFromKEY,sample_generate
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# if __name__ =="__main__":

    # 1.样本生成
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # 切换工作目录
    # os.chdir(script_dir)
    # # 参数配置
    # param_ranges = {
    #     'workpiece_temp': (875, 965),    # 工件温度范围 (℃)
    #     'mold_temp': (300, 700),         # 模具温度范围 (℃)
    #     'forging_speed': (10, 50)        # 锻造速度范围 (mm/s)
    # }
    # sample_generate(100,param_ranges,"../../data")
    # FullSampleGenerate(param_ranges)  # 生成全因子采样数据

    # 2.数据集合并（如果有多个数据集）
    # file1_data = read_file('../../data/RES-108.txt')
    # file2_data = read_file('../../data/RES-308.txt')
    
    # 处理文件2的工艺参数四舍五入（在merge_files中处理）
    # merged_data = merge_files(file1_data, file2_data)
    
    # 输出结果
    # write_output('../../data/RES.txt', merged_data)
    
    # 3.自动化模拟
    # 输入模板文件位置
    # MODEL_KEY = "D:\\Humingrui\\pyscript\\MODEL.KEY"
    # # 输入要设置的实验参数文件位置
    # # parameter_txt = "D:\\Humingrui\\pyscript\\IN.txt"
    # parameter_txt = "D:\\Humingrui\\pyscript\\AUTO\\check\\IN.txt"

    # # 检查
    # # TEST = os.path.join(os.getcwd(),"TEST")
    # # delete_directory_if_exists(TEST)

    # # 自动创建几个文件夹
    # # 存放批量生成的KEY文件位置
    # SAVE_KEY_PATH = os.path.join(os.getcwd(), "CHECK\\KEY_SAVE")
    # os.makedirs(SAVE_KEY_PATH,exist_ok = True)

    # # 存放由KEY文件转为DB文件的位置
    # SAVE_DB_PATH = os.path.join(os.getcwd(),"CHECK\\DB")
    # os.makedirs(SAVE_DB_PATH,exist_ok = True)

    # # 存放结果KEY文件的位置
    # RES_KEY_PATH = os.path.join(os.getcwd(),"CHECK\\RES_KEY")
    # os.makedirs(RES_KEY_PATH,exist_ok = True)

    # # 存放最后的训练集txt位置
    # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # RES_PATH = os.path.join(os.getcwd(),f"CHECK\\{current_time}-RES.txt")

    # # 初始化必要参数
    # # 工件温度 上模温度 下模温度 模具下压速度
    # num = 0
    # work_tmp = []
    # top_tmp = []
    # button_tmp = []
    # spd_arry = []

    # # 批量生成模板KEY文件
    # with open(parameter_txt, 'r',encoding = 'utf-8') as file:
    #      org_lines = file.readlines()  # 读取所有行
    #      for i,line in enumerate(org_lines):
    #          arry = line.split()
    #          num = arry[0]
    #          work_tmp.append(arry[1])
    #          top_tmp.append(arry[2])
    #          button_tmp.append(arry[2])
    #          spd_arry.append(arry[3])
    # InputKEYParameter(MODEL_KEY,SAVE_KEY_PATH,str(num),work_tmp,top_tmp,button_tmp,spd_arry)
    
    # # KEY转为DB
    # # 必须一个个调用KEY转为DB的前处理
    # for i,keypath in enumerate(KEY):
    #     os.makedirs(f"{SAVE_DB_PATH}\\{i}",exist_ok = True)
    #     path = GetNewFilePath(keypath,f"{SAVE_DB_PATH}\\{i}","","DB")
    #     DB.append(path)
    #     if os.path.exists(path):
    #         continue
    #     ProcessKEY_TO_DB(keypath,path)
    
    # # 启动进程计算DB文件
    # # 可手动调节启动进程个数
    # ProcessRun_CALDB(DB)

    # # 4 数据提取
    # # 必要的初始变量
    # DBPATH =[]
    # res_lines = []
    # num = 10  # 当前文件个数
    
    # # 参数文件位置
    # parameter_txt = "D:\\Humingrui\\pyscript\\IN.txt"
    # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # txt_savepath = os.path.join(os.getcwd(),f"{current_time}-RES.txt")
    
    # # 先获取结果DB文件位置集合
    # root_dir = "D:\\Humingrui\\pyscript\\AUTO\\TEST\\DB"
    # for folder_num in range(num):  
    #     folder_path = os.path.join(root_dir, str(folder_num))  
    #     if os.path.exists(folder_path):  
    #         # 遍历文件夹中的文件 寻找 .DB 文件
    #         for file_name in os.listdir(folder_path):
    #             if file_name.endswith(".DB"):  # 检查是否是 .DB 文件
    #                 full_path = os.path.join(folder_path, file_name)  # 构造完整路径
    #                 print(f"DBPATH = {full_path}\n")
    #                 DBPATH.append(full_path)  # 添加到列表
    #                 break    
    # # 提取参数信息组装成数据集
    # with open(parameter_txt, 'r',encoding = 'utf-8') as file:
    #     # 获取参数文件数据信息
    #     lines = file.readlines() 
    #     # index对应单个DB文件也对应训练集的行
    #     for index,dbpath in enumerate(DBPATH):
    #         print(f"当前提取的文件为：{dbpath}\n")
    #         stress = 0
    #         load = 0
    #         RES_KEY = []
    #         RES_KEY_PATH = os.path.join(os.getcwd(),"D:\\Humingrui\\pyscript\AUTO\\RES_KEY",f"GLOBAL{index}")
    #         os.makedirs(RES_KEY_PATH,exist_ok = True)
    #         for j in range(0,881):
    #             path =  GetNewFilePath(dbpath,f"{RES_KEY_PATH}",str(j),"KEY")
    #             RES_KEY.append(path)
    #             while not os.path.exists(path):
    #                 ProcessDB_TO_KEY(dbpath,path,str(j))
    #         # 分割数据集的行 提取DB文件中所有KEY指定的最值
    #         arry = lines[index].split()
    #         worktmp = arry[1]
    #         dietmp = arry[2]
    #         spd = arry[3]
    #         load = ExtractValueFromKEY(RES_KEY,"FORCE")[0]
    #         stdv = ExtractValueFromKEY(RES_KEY,"USRELM","GRAIN")[0]
    #         max_grain_size = ExtractValueFromKEY(RES_KEY,"USRELM","GRAIN")[1]
    #         avg_grain_size = ExtractValueFromKEY(RES_KEY,"USRELM","GRAIN")[2]

            
    #         print(f"工件晶粒尺寸标准差 = {stdv:.2f}，最大晶粒尺寸 = {max_grain_size:.2f}，平均晶粒尺寸 = {avg_grain_size:.2f}，最大模具载荷 = {load:.2f}\n")
    #         # 插入数据集
    #         line = f"{index}\t{worktmp}\t{dietmp}\t{spd}\t{stdv:.2f}\t{max_grain_size:.2f}\t{avg_grain_size:.2f}\t{load:.2f}\n"
    #         with open(txt_savepath, 'a',encoding = 'utf-8') as file:
    #             file.writelines(line)
            
            
    # print(f"数据集已经制作完成，请查看文件：{txt_savepath}")