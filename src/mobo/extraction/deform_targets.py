"""DEFORM KEY 文件目标提取原子函数。

按目标（应力/载荷/晶粒）从 KEY 文件解析出的多帧文本行中抽取标量结果。统一签名
``fn(AllLines, obj_id, inprogress, select_component)``，其中 ``select_component``
用于选取存在多分量的目标（如载荷方向、晶粒组织分量）。
"""

import numpy as np
import statistics
from typing import List


##########################################################
###################自定义提取函数部分########################
def _extractMaxStress(AllLines:List[List[str]],obj_id:str,inprogress:bool,select_component=None)->str:
    finall_res = -1.0
    # 找首行
    def fun(lines:List[str])->float:
        res,pos,num = 0,-1,0
        for row,line in enumerate(lines):
            arry = line.split()
            if len(arry) == 4 and arry[0] == 'STRESS' and arry[1] == obj_id:
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

def _extractMaxLoad(AllLines:List[List[str]],obj_id:str,inprogress:bool,select_component=0)->str:
    # 载荷提取：FORCE obj fx [fy] [fz]，select_component 选取分量索引(0=x,1=y,2=z)
    idx = int(select_component)
    finall_res = 0.0
    def fun(lines:List[str])->float:
        for line in lines:
            arry=line.split()
            if len(arry)>=3 and arry[0]=='FORCE' and arry[1]==obj_id:
                comps=list(map(float,arry[2:]))
                return comps[idx] if 0<=idx<len(comps) else 0.0
        return 0.0
    if inprogress:
        for lines in AllLines:
            finall_res=max(fun(lines),finall_res)
    else:
        finall_res=fun(AllLines[-1])
    return f"{finall_res:.2f}"

def _extractUsrGrainStdv(AllLines:List[List[str]],obj_id:str,inprogress:bool,select_component=3)->str:
    # 自定义晶粒模型时才用 USRELM；select_component 为每单元行的列索引
    col = int(select_component)
    finall_res = 0.0
    def fun(lines:List[str])->float:
        pos,num = -1,0
        grainsize = []
        res = 0.0
        for index,line in enumerate(lines):
            arry = line.split()
            if len (arry) == 5 and arry[0] == 'USRELM' and arry[1] == obj_id:
                pos,num = index + 1,int(arry[2])
                break
        if pos != -1 and num > 0:
            for i in range(num):
                arr = lines[pos + i].split()
                grainsize.append(float(arr[col]))
            res = statistics.stdev(grainsize)
        return res
    if inprogress:
        for lines in AllLines:
            finall_res = max(fun(lines),finall_res)
    else:
        finall_res = fun(AllLines[-1])
    return "{:.2f}".format(finall_res)

def _extractGrainMorph(AllLines:List[List[str]],obj_id:str,inprogress:bool,select_component=1)->str:
    # GRAIN obj num_units vals_per_unit ...，每单元 vals_per_unit 个晶粒组织信息
    # select_component 选取单元内第几个分量，跨单元求标准差
    comp = int(select_component)
    finall_res = 0.0
    def fun(lines:List[str])->float:
        pos,num,per = -1,0,0
        for index,line in enumerate(lines):
            arry = line.split()
            if len(arry) >= 4 and arry[0] == 'GRAIN' and arry[1] == obj_id:
                pos,num,per = index + 1,int(arry[2]),int(arry[3])
                break
        if pos == -1 or num <= 0 or per <= 0:
            return 0.0
        values = []
        i = pos
        for _ in range(num):
            unit = lines[i].split()[1:]
            i += 1
            while len(unit) < per:
                unit += lines[i].split()
                i += 1
            values.append(float(unit[comp]))
        return statistics.stdev(values) if len(values) > 1 else 0.0
    if inprogress:
        for lines in AllLines:
            finall_res = max(fun(lines),finall_res)
    else:
        finall_res = fun(AllLines[-1])
    return "{:.2f}".format(finall_res)


########################Helper#######################

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