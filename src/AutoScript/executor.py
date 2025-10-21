# ********************************************
#  创建时间： 2025/05/14
#  名    称： 自动化模拟脚本
#  版    本： 1.0
#  @author    Hu Mingrui
#  说    明： 本功能负责批量产生KEY文件
#  将KEY文件保存为DB文件 运行模拟求解器
#  求解完成后批量提取对应的数据到指定位置
#  ！注意！输入的所有路径请符合windows规范为了
#  避免转义字符的影响 统一采用双斜线表示路径
# *******************************************
#  ！模块函数：
#  InputKEYParameter()  提取用户的输入
#  Modify_KEY_FILE()    根据模板KEY文件批量产生实验KEY文件
#  ProcessKEY_TO_DB()   将批量的KEY文件转为可用于提交计算的DB文件
#  ProcessRun_CALDB()   将批量准备好的DB文件提交计算
#  ProcessDB_TO_KEY()   将计算完成的DB文件的最后一步结果保存为KEY文件
#
from pathlib import Path
from queue import Queue
import os,os.path,subprocess
import threading
import time,shutil
import numpy as np
from datetime import datetime

MODEL_KEY = '' # 模板文件位置
KEY = [] # 批量生成的KEY文件位置
DB = [] # 批量生成的DB文件位置
RES_KEY = [] # 批量生成的结果KEY文件位置
# 导入环境变量后可不用输入路径
DEF_PRE_64_path = "DEF_PRE_64.exe" 
DEF_ARM_CTL_path= "DEF_ARM_CTL.COM"
# 全局计数器及锁
solvernum = 0
solvernum_lock = threading.Lock()


#  @brief  获取用户的自定义输入 包括：
#  KEY文件位置 path
#  批量产生的KEY文件存储位置  save_path
#  批量生成的KEY文件数量 num
#  批量输入的更改工件温度 work_tmp
#  批量输入的更改上模温度 top_tmp
#  批量输入的更改下模温度 button_tmp
#  批量输入的更改模具压下速度 spd_arry
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
#  @about  目前更改的KEY文件参数只有工件温度和模
#  具压下速率
def InputKEYParameter(path = '',save_path ='',num = '',work_tmp = [],
                      top_tmp =[],button_tmp =[],spd_arry = [],workpiece = 1,top_die = 2,button_die = 3):
     
    while True:
        if not path:
            path = input("请输入模板KEY文件的路径：\n")
            if not os.path.exists(path):
                print("输入的路径非法或文件不存在，请重试！\n")
                continue
        path = FormatPath(path)
        break
    while True:
        if not num:
            num = input("请输入要批处理的数量：\n")
            if not num.isdigit or int(num) < 0:
                print("输入内容或数量不正确，请重试！\n")
                continue
        break

    while True:
        if not save_path:
            save_path = input("请输入需要保存的文件目录：\n")
            if not os.path.exists(save_path):
                print("输入的目录路径非法或不存在，请重试！\n")
                continue
        save_path = FormatPath(save_path)
        break

    for i in range(int(num)):
        if not work_tmp:
            while True:
                t = (input("请依次输入工件的温度℃：\n"))
                if not t.isdigit or float(t) <=0 or len(t) > 12:
                    print("输入内容不正确，请重试！\n")
                    continue
                work_tmp.append(t)
                break
        if not top_tmp:
            while True:
                t = (input("请依次输入模具的温度℃：\n"))
                if not t.isdigit or float(t) <=0 or len(t) > 12:
                    print("输入内容不正确，请重试！\n")
                    continue
                top_tmp.append(t)
                button_tmp.append(t)
                break
        if not spd_arry:
            while True:
                s = (input("请依次输入上模的速度mm/s：\n"))
                if not s.isdigit or float(s) <=0 or len(s) > 12:
                    print("输入内容不正确，请重试！\n")
                    continue
                spd_arry.append(s)
                break
    Modify_KEY_FILE(path,num,work_tmp,top_tmp,button_tmp,
                    spd_arry,workpiece,top_die,button_die,save_path)



#  @brief  根据模板KEY文件批量生成所需的KEY文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
#  @about  目前更改的KEY文件参数只有工件温度和模
#  具温度模以及具压下速率
def  Modify_KEY_FILE(path,num,work_tmp,top_tmp,button_tmp,spd_arry,
                     workpiece,top_die,button_die,save_path):
    with open(path, 'r',encoding = 'utf-8') as file:
        org_lines = file.readlines()  # 读取所有行
    cnt = 0
    for k in range(int(num)):
        modify_lines = []
        for i,line in enumerate(org_lines):
            arry = line.split()
            new_line = line
            if len(arry) >= 2:
                if arry[0] == 'NDTMP' and int(arry[1]) == workpiece:
                    t = FormatFloat(work_tmp[k])
                    new_line = line.replace(arry[-1],t)
                
                if arry[0] == 'NDTMP' and int(arry[1]) == top_die:
                    t = FormatFloat(top_tmp[k])
                    new_line = line.replace(arry[-1],t)

                if arry[0] == 'NDTMP' and int(arry[1]) == button_die:
                    t = FormatFloat(button_tmp[k])
                    new_line = line.replace(arry[-1],t)
                
                if arry[0] == 'MOVCTL' and int(arry[1]) == top_die:
                    s = FormatFloat(spd_arry[k])
                    new_line = line.replace(arry[-1],s)

            modify_lines.append(new_line)
        # tag = f"{work_tmp[k]}-{top_tmp[k]}-{spd_arry[k]}"
        file_path = GetNewFilePath(path,save_path,k,"KEY")
        cnt += 1
        with open(file_path, 'w',encoding = 'utf-8') as file:
            file.writelines(modify_lines)
        # 把生成的KEY文件位置保存
        KEY.append(file_path)
        print(f"第{cnt}个新生成的KEY文件已经保存！\n")


#  @brief  启动DEF_PRE_64.exe进行KEY文件转为DB文
#  件操作 可批量生成DB文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
#  @about  
def ProcessKEY_TO_DB(KEY_inputpath,DB_savepath):
    
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )

    # 实时显示输出
    with open("AUTO_OPERATION_LOG.txt", "a", encoding="utf-8") as log_file:
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            if output:
                # print(output.strip())
                log_file.write(output)
    
    # 清理临时文件
    os.remove(cmd_file)


#  @brief  启动DEF_ARM_CTL.COM 提交计算
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
#  @about  注意！本模块需要规划最大能启动多少个
#  进程进行计算 以免过多导致计算非常缓慢
def ProcessRun_CALDB(DB_inputpath,Process_Num = 24):

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


#  @brief  启动DEF_PRE_64.exe 将DB文件的某一步
#  转为KEY文件 用于批量产生结果KEY文件
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
#  @about  注意！本模块需要输入最后一步的具体
#  步数step 因此需要用户设定好后再输入 否则程序报错
def ProcessDB_TO_KEY(DB_inputpath,KEY_savepath,step = ""):

    cmd = f"E\n2\n2\n{DB_inputpath}\n{step}\nE\nE\n8\n{KEY_savepath}\nE\nY\n"
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
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            if output:
                # print(output.strip())
                log_file.write(output)

    os.remove(cmd_file)




#  @brief  提取KEY文件中的目标行
#  当前默认提取锻件的应力情况
#  并且调用等效应力计算最大等效应力值
#  @return 
#  @author Hu Mingrui
#  @date   2025/06/03
#  @about  
def ExtractEfstressFromKEY(InputPar,KEY_inputpath,txt_savepath,argv = 'STRESS',argc = '1'):

    new_lines = []
    for i,path in enumerate(KEY_inputpath):
        with open(path, 'r',encoding = 'utf-8') as file:
            lines = file.readlines() 
            pos = - 1
            num = 0
            for index,line in enumerate(lines):
                arry = line.split()
                if len(arry) == 4 and arry[0] == argv and arry[1] == argc:
                    pos = index
                    num = int(arry[2])
                    break
            if pos != -1 and num > 0:
                cnt = 1
                index += 1
                von_stress = -1
                while cnt <= num:
                    arry1 = lines[index].split()
                    arry2 = lines[index + 1].split()
                    stress = [float(arry1[1]),float(arry1[2]),float(arry1[3]),
                              float(arry1[4]),float(arry1[5]),float(arry2[0])]
                    von_stress = max(von_stress,calculate_von_mises(stress))
                    cnt += 1
                    index += 2
                # 组装参数
                new_line = f"{i}\t"
                for j in range(len(InputPar[i])):
                    new_line += f"{InputPar[i][j]}\t"
                new_line += f"{von_stress}\n"
                new_lines.append(new_line)
    with open(txt_savepath, 'w',encoding = 'utf-8') as file:
        file.writelines(new_lines)


#**********************************************功能函数部分**********************************************

#  @brief  功能函数 1:
#  格式化输入数据为符合DEFROM
#  数据规范的科学计数法格式
#  @return 
#  @author DeepSeek
#  @date   2025/06/05
def FormatFloat(num):
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
def GetNewFilePath(file_path,savepath,tag,FILETYPE):
    predix = Path(file_path).stem
    res = os.path.join(savepath,f"{predix}{tag}.{FILETYPE}")
    return res

#  @brief  功能函数 3:
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

#  @brief  功能函数 4: 
#  打印输出到脚本控制台并将操
#  作记录到日志文件当中
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/14
def ReadOutput(process):
    # 实时读取并打印输出
    for line in iter(process.stdout.readline, ''):
        # time.sleep(0.1)
        if line:
            print(line, end = '')  # 打印到控制台
            with open("AUTO_OPRATION_LOG.txt", "a", encoding = "utf-8") as log_file:
                log_file.write(line)
        else:
            break

#  @brief  功能函数 5 规范输入路径统一
#  转为双斜线格式的
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
def FormatPath(path):
    # 先标准化路径 
    normal_path = os.path.normpath(path)
    return normal_path.replace("\\","\\\\")

#  @brief  功能函数 6 启动DEF_ARM_CTL
#  提交求解进程
#  @return 
#  @author Hu Mingrui
#  @date   2025/05/15
def Solve(DB_path, path_dir):
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
                stdin=subprocess.PIPE,
                shell=False,
                cwd=path_dir,
                text=True  
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


#  @brief  功能函数 7 计算等效应力
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

#  @brief  功能函数 8 删除目录
#  @return 
#  @author Hu Mingrui
#  @date   2025/06/05
def delete_directory_if_exists(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"目录 {directory_path} 及其所有内容已成功删除。")
        except Exception as e:
            print(f"删除目录时出错: {e}")
    else:
        print(f"目录 {directory_path} 不存在。")

#  @brief  功能函数 9 查找正在运行的 DEF_SIM.exe个数
#  @return 
#  @author Hu Mingrui
#  @date   2025/06/26
def GetNum(process_name = 'DEF_SIM.exe'):
    """
    统计指定进程名的运行实例数量
    """
    try:
        # 调用 tasklist 命令
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
            capture_output=True,
            text=True,
            shell=True
        )

        # 解析输出
        output = result.stdout
        lines = output.split('\n')
        
        # 统计匹配的行数（减去标题行和可能的空行）
        count = 0
        for line in lines:
            if process_name.lower() in line.lower():
                count += 1

        # 减去标题行（例如 "Image Name" 行）
        if count > 0:
            count -= 1  # 减去标题行

        return count

    except Exception as e:
        print(f"Error: {e}")
        return 0


# if __name__ == "__main__":

#     # 输入模板文件位置
#     MODEL_KEY = "D:\\Humingrui\\pyscript\\MODEL.KEY"
#     # 输入要设置的实验参数文件位置
#     # parameter_txt = "D:\\Humingrui\\pyscript\\IN.txt"
#     parameter_txt = "D:\\Humingrui\\pyscript\\AUTO\\check\\IN.txt"

#     # 检查
#     # TEST = os.path.join(os.getcwd(),"TEST")
#     # delete_directory_if_exists(TEST)

#     # 自动创建几个文件夹
#     # 1.存放批量生成的KEY文件位置
#     SAVE_KEY_PATH = os.path.join(os.getcwd(), "CHECK\\KEY_SAVE")
#     os.makedirs(SAVE_KEY_PATH,exist_ok = True)

#     # 2.存放由KEY文件转为DB文件的位置
#     SAVE_DB_PATH = os.path.join(os.getcwd(),"CHECK\\DB")
#     os.makedirs(SAVE_DB_PATH,exist_ok = True)

#     # 3.存放结果KEY文件的位置
#     RES_KEY_PATH = os.path.join(os.getcwd(),"CHECK\\RES_KEY")
#     os.makedirs(RES_KEY_PATH,exist_ok = True)

#     # 4.存放最后的训练集txt位置
#     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     RES_PATH = os.path.join(os.getcwd(),f"CHECK\\{current_time}-RES.txt")

#     # 初始化必要参数
#     # 工件温度 上模温度 下模温度 模具下压速度
#     num = 0
#     work_tmp = []
#     top_tmp = []
#     button_tmp = []
#     spd_arry = []

#     # 批量生成模板KEY文件
#     with open(parameter_txt, 'r',encoding = 'utf-8') as file:
#          org_lines = file.readlines()  # 读取所有行
#          for i,line in enumerate(org_lines):
#              arry = line.split()
#              num = arry[0]
#              work_tmp.append(arry[1])
#              top_tmp.append(arry[2])
#              button_tmp.append(arry[2])
#              spd_arry.append(arry[3])
#     InputKEYParameter(MODEL_KEY,SAVE_KEY_PATH,str(num),work_tmp,top_tmp,button_tmp,spd_arry)
    
#     # KEY转为DB
#     # 必须一个个调用KEY转为DB的前处理
#     for i,keypath in enumerate(KEY):
#         os.makedirs(f"{SAVE_DB_PATH}\\{i}",exist_ok = True)
#         path = GetNewFilePath(keypath,f"{SAVE_DB_PATH}\\{i}","","DB")
#         DB.append(path)
#         if os.path.exists(path):
#             continue
#         ProcessKEY_TO_DB(keypath,path)
    
#     # 启动进程计算DB文件
#     # 可手动调节启动进程个数
#     ProcessRun_CALDB(DB)


#     # DB转为结果KEY文件
#     # for i,dbpath in enumerate(DB):
#     #     path =  GetNewFilePath(dbpath,f"{RES_KEY_PATH}","","KEY")
#     #     RES_KEY.append(path)
#     #     if os.path.exists(path):
#     #         continue
#     #     ProcessDB_TO_KEY(dbpath,path)

#     # 提取KEY文件的等效应力并组装成训练集
#     # ORG_PAR = []
#     # for i in range(len(RES_KEY)):
#     #     temp = [work_tmp[i],top_tmp[i],button_tmp[i],spd_arry[i]]
#     #     ORG_PAR.append(temp)
#     # ExtractEfstressFromKEY(ORG_PAR,RES_KEY,RES_PATH)












    

