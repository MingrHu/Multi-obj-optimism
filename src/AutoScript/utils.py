import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,statistics,subprocess,threading,time,shutil
from pathlib import Path
from pyDOE import lhs
from queue import Queue
from typing import List

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# *********************SOME VAR DEF***********************
# KEY文件关键字变量
KEYFILE_VAR_DIC = {
    'temp':"NDTMP",
    'speed':"MOVCTL",

}
# 指定对象
OBJ_DEF = {
    'workpiece':1,
    'topdie':2,
    'butdie':3
}
# 采样方法
SAMPLE = {
    'lhs':
}

# 全局计数器及锁
solvernum = 0
solvernum_lock = threading.Lock()
# 导入环境变量后可不用输入路径
DEF_PRE_64_path = "DEF_PRE_64.exe" 
DEF_ARM_CTL_path= "DEF_ARM_CTL.COM"

# ************************FUNC DEF**************************
#  @brief  功能函数 1:
#  格式化输入数据为符合DEFROM
#  数据规范的科学计数法格式
#  @return 
#  @author DeepSeek
#  @date   2025/06/05
def FormatFloat(num:str):
    try:
        n = float(num)
    except (ValueError, TypeError):
        return str(num)  # 非数值类型直接返回原字符串
    
    # 处理特殊值
    if n == 0.0:
        return "0.0000000000E+000"
    
    # 格式化为科学计数法
    s = "{0:.10e}".format(n).upper()
    
    # 分割底数和指数
    if 'E' not in s:
        # 对于没有指数的数（如0.0），添加指数部分
        return f"{s}E+000"
    
    l, r = s.split('E')
    
    # 处理指数部分
    sign = '+'  # 默认正号
    if r.startswith('+'):
        exp_str = r[1:]
    elif r.startswith('-'):
        sign = '-'
        exp_str = r[1:]
    else:
        exp_str = r
    
    # 移除前导零并补足3位
    exp_str = exp_str.lstrip('0') or '0'  # 处理全零情况
    exp_num = exp_str.zfill(3)
    
    return f"{l}E{sign}{exp_num}"

#  @brief  功能函数 2: 
#  根据保存位置和原文件全名file_path生成
#  新的保存文件路径res
#  生成规则是取file_path的主体名（无扩展名）
#  加上前面的主体保存路径savepath
#  与设定的标识符tag进行拼接
#  加上最后的文件类型扩展FILETYPE生成新文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
def GetNewFilePath(file_path:str,savepath:str,tag:str,FILETYPE:str):
    predix = Path(file_path).stem
    res = os.path.join(savepath,f"{predix}{tag}.{FILETYPE}")
    return res

#  @brief  功能函数 3: 
#  启动DEF_PRE_64.exe进行KEY文件转为DB文件操作 
#  可批量生成DB文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
#  @about  
def ProcessKEY_TO_DB(KEY_inputpath:str,DB_savepath:str):
    
    # 添加执行命令
    cmd_content = f"E\n2\n1\n{KEY_inputpath}\nE\nE\n7\n2\n{DB_savepath}\nY\nE\nY\n"

    # 将命令写入临时文件
    cmd_file = "TEMP_KEY-DB.txt"
    with open(cmd_file, "w", encoding="utf-8") as f:
        f.write(cmd_content)
    
    # 使用输入重定向执行命令
    # 这也是DEFORM推荐的方法
    command = f'"{DEF_PRE_64_path}" < "{cmd_file}"'
    process = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        shell = True
    )

    # 实时显示输出
    with open("AUTO_OPERATION_LOG.txt", "a", encoding ="utf-8") as log_file:
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            if output:
                log_file.write(output)
    
    # 清理临时文件
    os.remove(cmd_file)

#  @brief  功能函数 4:
#  封装与DEFORM子程序的输入交
#  互操作
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
def OpInput(process, content):
    time.sleep(0.1)
    content += "\n"
    process.stdin.write(content)
    process.stdin.flush()

#  @brief  功能函数 5: 
#  启动DEF_ARM_CTL 提交求解进程
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
def Solve(DB_path:str, path_dir:str):
    """
    异步启动求解进程
    """
    global solvernum

    def _solve():
        global solvernum
        try:
            # 启动子进程，不捕获stdout，直接继承父进程的终端
            Process = subprocess.Popen(
                DEF_ARM_CTL_path,
                stdin = subprocess.PIPE,
                shell = False,
                cwd = path_dir,
                text = True  
            )

            # 输入命令
            OpInput(Process, DB_path)
            OpInput(Process, "B")

            # 等待进程结束
            Process.wait()

            # 关闭输入流
            if Process.stdin:
                Process.stdin.close()

            print(f"当前任务计算完成！请查看{DB_path}结果\n")

        except Exception as e:
            print(f"Error in solver process: {e}")
        finally:
            with solvernum_lock:
                solvernum -= 1

    with solvernum_lock:
        solvernum += 1
        print(f"当前正在计算的任务有：{solvernum}个")

    thread = threading.Thread(target=_solve)
    thread.start()

#  @brief  功能函数6:
#  启动DEF_ARM_CTL.COM 提交计算
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
#  @about  注意！本模块需要规划最大能启动多少个
#  进程进行计算 以免过多导致计算非常缓慢
def ProcessRun_CALDB(DB_inputpath:List,Process_Num = 24):

    task_queue = Queue()
    for i, dbpath in enumerate(DB_inputpath):
        task_queue.put((i+1, dbpath))

    while True:
        try:
            with solvernum_lock:
                current_num = solvernum
            if current_num >= Process_Num:
                time.sleep(3)
                continue
            else:
                # 只有当前队列为空且文件计算完成才停止
                if task_queue.qsize() == 0 and current_num == 0:
                    break
                elif task_queue.qsize() == 0:
                    time.sleep(60)
                    continue
                file_num, db_path = task_queue.get()
                path_dir = os.path.dirname(db_path)
                FN = os.path.splitext(os.path.basename(db_path))[0]
                path = os.path.join(path_dir,FN)

                # 线程启动目标进程
                thread = threading.Thread(target = Solve, args=(path,path_dir,))
                thread.daemon = True
                thread.start()
                print(f"开始计算第{file_num}个 请等待...计算完成后结果DB文件自动保存至{path_dir}\n")
                # 防止过快导致solvernum没更新
                time.sleep(5)
        except Exception as e:
            print(f"Error happend!{e}")

# ************************SAMPLE DEF**************************
#  @brief  样本函数1:
#  拉丁超立方采样方法
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
def LHSSampleGenerate(n_samples:int, param_ranges:dict,sample_save_path:str):

    # 阶段1：生成拉丁超立方采样样本
    def generate_lhs_samples(n_samples, ranges):  # 修正：添加 n_samples 参数
        """生成拉丁超立方采样输入参数"""
        samples = lhs(len(ranges), samples = n_samples)  # 生成 0-1 之间的 LHS 样本
        scaled_samples = np.zeros_like(samples)
        for i, (key, (low, high)) in enumerate(ranges.items()):
            scaled_samples[:, i] = np.round(samples[:, i] * (high - low) + low).astype(int)
        return pd.DataFrame(scaled_samples, columns=ranges.keys())

    # 生成初始样本
    df = generate_lhs_samples(n_samples, param_ranges)

    # 阶段2：添加边界样本（8个角点）
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns = param_ranges.keys())
    for col in boundary_df.columns:
        boundary_df[col] = boundary_df[col].astype(int)

    # 合并样本并去重
    df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates()

    # 保存结果
    SaveResult(boundary_df, df, 'lhs',sample_save_path)

def FullSampleGenerate(param_ranges:dict,sample_save_path:str):
    # 阶段1：全因子采样生成初始样本
    def generate_full_factorial_samples(ranges, levels=5):
        """生成全因子采样输入参数"""
        # 为每个参数生成等间隔的水平值
        param_levels = {}
        for key, (low, high) in ranges.items():
            param_levels[key] = np.round(np.linspace(low, high, levels)).astype(int)
    
        # 生成所有参数组合
        combinations = list(product(*param_levels.values()))
        df = pd.DataFrame(combinations, columns=ranges.keys())
        return df
 
    # 生成初始样本（默认每个参数10个水平）
    df = generate_full_factorial_samples(param_ranges, levels=10)

    # for col in df.columns:
    #     df[col] = df[col].round(4)  # 保留小数点后四位 DEFORM要求
 
    # 阶段2：添加边界样本（8个角点）
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns=param_ranges.keys())
    for col in boundary_df.columns:
        boundary_df[col] = boundary_df[col].astype(int)
 
    # 合并样本并去重
    df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates()

    # 保存并输出结果
    SaveAndprint(boundary_df, df, 'fullfactorial',sample_save_path)
 


# df：生成的全因子采样数据
# boundary_df：边界样本数据
# extype：样本类型（如 'fullfactorial'）
def SaveResult(boundary_df,df,extype,save_path):
        # 可视化（保持与LHS相同的图表结构）
    plt.figure(figsize=(15, 5))
 
    # 子图1：参数空间分布
    plt.subplot(1, 3, 1)
    plt.scatter(
        df['workpiece_temp'],  # X轴：工件温度
        df['mold_temp'],       # Y轴：模具温度
        c=df['forging_speed'], # 颜色：锻造速度（连续值）
        cmap='viridis',        # 颜色映射表
        s=30                   # 点的大小
    )
    plt.xlabel('Workpiece Temperature (℃)')
    plt.ylabel('Mold Temperature (℃)')
    plt.colorbar(label='Forging Speed (mm/s)')
    plt.title('Parameter Space Distribution')
 
    # 子图2：单参数分布直方图
    plt.subplot(1, 3, 2)
    for col in df.columns:
        plt.hist(df[col], bins=20, alpha=0.5, label=col)
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Individual Parameter Distributions')
 
    # 子图3：边界样本高亮
    plt.subplot(1, 3, 3)
    plt.scatter(df['workpiece_temp'], df['mold_temp'], 
                c='blue', s=20, alpha=0.6, label='Normal Samples')
    plt.scatter(boundary_df['workpiece_temp'], boundary_df['mold_temp'],
                c='red', s=50, marker='*', label='Boundary Samples')
    plt.xlabel('Workpiece Temperature (℃)')
    plt.ylabel('Mold Temperature (℃)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    plt.title('Boundary Samples Highlight')
    plt.grid(False)
 
    plt.tight_layout()
    plt.show()
 
    # 数据保存 添加序号
    df_with_id = df.copy()
    df_with_id.insert(0, 'ID', range(1, len(df_with_id) + 1))
    df_with_id.to_csv(f'{save_path}/IN{extype}.txt', sep='\t', index=False, header=False)

    print(f"采样数据已保存至 {save_path}IN{extype}.txt 总计{len(df)} 个样本")
    print("输入参数统计信息：")
    print(df.describe())

def sample_generate(samples_num:int,param_ranges:dict,sample_save_path:str):
    LHS_n_samples = samples_num  # 初始样本数量
    LHSSampleGenerate(LHS_n_samples, param_ranges,sample_save_path)  # 生成拉丁超立方采样数据

def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # 假设列是以空格或制表符分隔的
            parts = line.strip().split()
            if len(parts) >= 5:  # 确保至少有5列（文件1）或6列（文件2）
                data.append(parts)
    return data

def write_output(filename, data):
    with open(filename, 'w') as file:
        for row in data:
            file.write(row) 

def merge_files(file1_data, file2_data):
    merged = {}
    
    # 处理文件1的数据
    for row in file1_data:
        if float(row[4]) < 0:
            continue
        work_temp = f"{float(row[1]):.1f}"
        die_temp = f"{float(row[2]):.1f}"
        speed = f"{float(row[3]):.1f}"

        grain_size_stdv = row[4]
        load = row[5] if len(row) > 5 else None

        key = (work_temp, die_temp, speed)
        if key not in merged:
            merged[key] = {
                'work_temp': work_temp,
                'die_temp':die_temp,
                'speed':speed,
                'grain_size_stdv': grain_size_stdv,
                'load': load
            }  
   
    # 处理文件2的数据
    for row in file2_data:
        if float(row[4]) < 0:
            continue
        work_temp = f"{float(row[1]):.1f}"
        die_temp = f"{float(row[2]):.1f}"
        speed = f"{float(row[3]):.1f}"

        grain_size_stdv = row[4]
        load = row[5] if len(row) > 5 else None
        
        key = (work_temp, die_temp, speed)
        if key not in merged:
            merged[key] = {
                'work_temp': work_temp,
                'die_temp':die_temp,
                'speed':speed,
                'grain_size_stdv': grain_size_stdv,
                'load': load
            }
    
    # 生成合并后的数据
    result = []
    for i,key in enumerate(merged):
        entry = merged[key]
        new_row = f"{i}\t{entry['work_temp']}\t{entry['die_temp']}\t{entry['speed']}\t{entry['grain_size_stdv']}\t{entry['load']}\n"
        result.append(new_row)
    
    return result

#  @brief  提取KEY文件中的目标行
#  当前默认提取锻件的应力情况
#  并且调用等效应力计算最大等效应力值
#  @return 
#  @author Hu Mingrui
#  @date   2025/06/03
#  @about  
def ExtractValueFromKEY(KEY_inputpath,argv = 'STRESS',user_type = 'GRAIN'):

    value = [0] * 3
    # 查找所有的KEY文件
    for i,path in enumerate(KEY_inputpath):
        # 某一步下面的应力值 单个KEY文件
        with open(path, 'r',encoding = 'utf-8') as file:
            lines = file.readlines() 
            # 更新最值
            if argv == 'STRESS':
                value[0] = max(value[0],_extractStress(lines))
            elif argv == 'FORCE':
                value[0] = max(value[0],_extractLoad(lines))
    # 拉取晶粒信息
    if argv == "USRELM" and user_type == 'GRAIN':
        with open(KEY_inputpath[-1], 'r',encoding = 'utf-8') as file:
            lines = file.readlines() 
            value = _extractGrainInfo(lines)

    return value

###################################功能函数部分###############################################
def _extractStress(lines,pos = -1,num = 0):
    # 找首行
    res = 0
    for index,line in enumerate(lines):
        arry = line.split()
        if len(arry) == 4 and arry[0] == 'STRESS' and arry[1] == '1':
            pos = index
            num = int(arry[2])
            break
    # 从首行开始遍历
    if pos != -1 and num > 0:
        cnt = 1
        index += 1
        while cnt <= num:
            arry1 = lines[index].split()
            arry2 = lines[index + 1].split()
            stress = [float(arry1[1]),float(arry1[2]),float(arry1[3]),
                        float(arry1[4]),float(arry1[5]),float(arry2[0])]
            res = max(res,calculate_von_mises(stress))
            cnt += 1
            index += 2
    return res

def _extractLoad(lines):
    # 模具载荷提取
    res = 0
    for index,line in enumerate(lines):
        arry = line.split()
        if len(arry) == 5 and arry[0] == 'FORCE' and arry[1] == '2':
            res = float(arry[4])
    return res

def _extractGrainInfo(lines):
    # 提取锻件晶粒尺寸信息
    pos,num = -1,0
    grainsize = []
    res = [-1]* 3
    for index,line in enumerate(lines):
        arry = line.split()
        if len (arry) == 5 and arry[0] == 'USRELM' and arry[1] == '1':
            pos,num = index + 1,int(arry[2])
            break
    if pos != -1 and num > 0:
        for i in range(num):
            arr = lines[pos + i].split()
            grainsize.append(float(arr[3]))
        res[0] = statistics.stdev(grainsize)
        res[1] = max(grainsize)
        res[2] = statistics.mean(grainsize)
    return res