


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












    

