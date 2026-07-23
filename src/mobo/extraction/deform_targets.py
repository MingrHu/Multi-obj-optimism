"""DEFORM KEY 文件目标提取原子函数。

从 ``AutoScript/utils.py`` 原样搬入的自定义提取函数：按目标（应力/载荷/晶粒）
从 KEY 文件解析出的多帧文本行中抽取标量结果。这些函数体保持 byte-for-byte 不变，
由 :mod:`mobo.extraction` 在导入时注册到原子能力层，并由
:class:`mobo.automation.config.DeformConfig` 继续复用。

统一签名（``key_lines`` 约定）：
    ``fn(AllLines: List[List[str]], obj: str, inprogress: bool) -> str``
"""

import numpy as np
import statistics
from typing import List


##########################################################
###################自定义提取函数部分########################
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
        for _,line in enumerate(lines):
            arry = line.split()
            # 根据deform的key文件关键字分布情况
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
            # 有必要讨论一下晶粒的相关情况
            # TODO(MingrHu)
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
