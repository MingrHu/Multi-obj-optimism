import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,statistics,subprocess,threading,time,shutil
from pathlib import Path
from pyDOE import lhs
from itertools import product
from queue import Queue
from typing import Dict, List, Tuple

# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)


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
#  根据保存位置和原文件全名file生成
#  新的保存文件路径res
#  生成规则是取file的主体名（无扩展名）
#  加上前面的主体保存路径savepath
#  与设定的标识符tag进行拼接
#  加上最后的文件类型扩展FILETYPE生成新文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
def GetNewFilePath(file:str,savepath:str,tag:str,FILETYPE:str):
    predix = Path(file).stem
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
            output = process.stdout.readline() # type: ignore
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

#  @brief  功能函数7：
#  启动DEF_PRE_64.exe 将DB文件的某一步
#  转为KEY文件 用于批量产生结果KEY文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
#  @about  注意！本模块需要输入最后一步的具体
#  步数step 因此需要用户设定好后再输入 否则程序报错
def ProcessDB_TO_KEY(DB_file:str,KEY_file:str,step = ""):

    cmd = f"E\n2\n2\n{DB_file}\n{step}\nE\nE\n8\n{KEY_file}\nE\nY\n"
    # 将命令写入临时文件
    cmd_file = "TEMP_DB-KEY.txt"
    with open(cmd_file, "w", encoding="utf-8") as f:
        f.write(cmd)

    command = f'"{DEF_PRE_64_path}" < "{cmd_file}"'
    process = subprocess.Popen(
    command,
    stdout = subprocess.PIPE,
    stderr = subprocess.STDOUT, 
    text = True,
    shell = True)

    # 实时显示输出
    with open("AUTO_OPERATION_LOG.txt", "a", encoding="utf-8") as log_file:
        while True:
            output = process.stdout.readline() # type: ignore
            if not output and process.poll() is not None:
                break
            if output:
                # print(output.strip())
                log_file.write(output)

    os.remove(cmd_file)

###################################自定义提取函数部分###############################################
def _extractMaxStress(AllLines:List[List[str]],obj:str,inprogress:bool)->str:
    finall_res = -1.0
    # 找首行
    def fun(lines:List[str])->float:
        res,pos,num = 0,-1,0
        for row,line in enumerate(lines):
            arry = line.split()
            if len(arry) == 4 and arry[0] == 'STRESS' and arry[1] == obj:
                pos = row
                num = int(arry[2])
                break
        # 从首行开始遍历
        if pos != -1 and num > 0:
            cnt = 1
            index = pos + 1
            while cnt <= num:
                arry1 = lines[index].split()
                arry2 = lines[index + 1].split()
                stress = [float(arry1[1]),float(arry1[2]),float(arry1[3]),
                            float(arry1[4]),float(arry1[5]),float(arry2[0])]
                res = max(res,calculate_von_mises(stress))
                cnt += 1
                index += 2
        return res
    if inprogress:
        for lines in AllLines:
            finall_res = max(fun(lines),finall_res) # type: ignore
    else:
        finall_res = fun(AllLines[-1])
    return "{:.2f}".format(finall_res)

def _extractMaxLoad(AllLines:List[List[str]],obj:str,inprogress:bool)->str:
    # 模具载荷提取
    finall_res = 0.0
    def fun(lines:List[str])->float:
        res = 0.0
        for index,line in enumerate(lines):
            arry = line.split()
            if len(arry) == 5 and arry[0] == 'FORCE' and arry[1] == '2':
                res = float(arry[4])
        return res
    if inprogress:
        for lines in AllLines:
            finall_res = max(fun(lines),finall_res)
    else:
        finall_res = fun(AllLines[-1])
    return "{:.2f}".format(finall_res)

def _extractGrainStdv(AllLines:List[List[str]],obj:str,inprogress:bool)->str:
    finall_res = 0.0
    # 提取锻件晶粒尺寸信息
    def fun(lines:List[str])->float:
        pos,num = -1,0
        grainsize = []
        res = 0.0
        for index,line in enumerate(lines):
            arry = line.split()
            if len (arry) == 5 and arry[0] == 'USRELM' and arry[1] == '1':
                pos,num = index + 1,int(arry[2])
                break
        if pos != -1 and num > 0:
            for i in range(num):
                arr = lines[pos + i].split()
                grainsize.append(float(arr[3]))
            res = statistics.stdev(grainsize)
        return res
    if inprogress:
        for lines in AllLines:
            finall_res = max(fun(lines),finall_res)
    else:
        finall_res = fun(AllLines[-1])
    return "{:.2f}".format(finall_res)

#  @brief 计算等效应力
#  von-misses准则
#  @return 
#  @author Hu Mingrui
#  @date   2025/06/03
def calculate_von_mises(stress):
    """计算等效应力 (Von Mises Stress)"""
    sxx, syy, szz, sxy, syz, sxz = stress

    return np.sqrt(0.5 * ((sxx - syy)**2 + 
                         (syy - szz)**2 + 
                         (szz - sxx)**2 + 
                         6 * (sxy**2 + syz**2 + sxz**2)))



# ************************SAMPLE DEF**************************
#  @brief  样本函数1:
#  拉丁超立方采样方法
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
def LHSSampleGenerate(n_samples:int, param_ranges,sample_save_path:str):

    # 阶段1：生成拉丁超立方采样样本
    def generate_lhs_samples(n_samples: int, param_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """生成拉丁超立方采样输入参数"""
        samples = lhs(len(param_ranges), samples=n_samples)  # 生成 0-1 之间的 LHS 样本
        scaled_samples = np.zeros_like(samples)
        for i, (key, (low, high)) in enumerate(param_ranges.items()):
            scaled_samples[:, i] = samples[:, i] * (high - low) + low # type: ignore
        df = pd.DataFrame(scaled_samples, columns=param_ranges.keys()) # type: ignore
        return df.round(2)  # 保留小数点后2位

    # 生成初始样本
    df = generate_lhs_samples(n_samples, param_ranges)
    # 阶段2：添加边界样本
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns=param_ranges.keys())

    # 合并样本并去重
    combined_df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates(subset=param_ranges.keys())
    df_formatted = combined_df.applymap(lambda x: f"{x:.2f}") # type: ignore
    SaveResult(df_formatted, 'lhs', sample_save_path)

def FullSampleGenerate(n_samples:int,param_ranges,sample_save_path:str):
    # 阶段1：全因子采样生成初始样本
    def generate_full_factorial_samples(ranges, levels=5):
        """生成全因子采样输入参数"""
        # 为每个参数生成等间隔的水平值
        param_levels = {}
        for key, (low, high) in ranges.items():
            param_levels[key] = np.linspace(low, high, levels)
    
        # 生成所有参数组合
        combinations = list(product(*param_levels.values()))
        df = pd.DataFrame(combinations, columns=ranges.keys())
        return df
 
    # 生成初始样本（默认每个参数5个水平）
    df = generate_full_factorial_samples(param_ranges, levels=10)

    for col in df.columns:
        df[col] = df[col].round(4)  # 保留小数点后四位 DEFORM要求
 
    # 阶段2：添加边界样本（8个角点）
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns = param_ranges.keys())
 
    # 合并样本并去重
    df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates()

    # 保存并输出结果
    SaveResult(df, 'fullfactorial',sample_save_path)

def SaveResult(df: pd.DataFrame, extype: str, save_path: str) -> None:
    """保存采样结果到文件"""
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存文件
    df.to_csv(f'{save_path}/IN{extype}.txt', sep='\t', index=False, header=False)
    print(f"采样数据已保存至 {save_path}IN{extype}.txt 总计{len(df)} 个样本")
    print("输入参数统计信息：")
    print(df.describe())

